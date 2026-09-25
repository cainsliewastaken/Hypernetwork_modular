"""Minimal data parallelism for torchrun (``torchrun --nproc-per-node=N train.py ...``).

Only the shared-parameter training stages (``base``, ``direct``) are distributed: every rank holds an
identical copy of the networks, draws its own data shuffle and noise, and gradients are averaged with
one flat all-reduce after each backward. Batch sizes in the config are per rank. Rank 0 does all
printing, logging, checkpoints and reports. No DistributedDataParallel wrapper: the wheel is called
through functional_call with per-sample weights, which the wrapper does not see.

Single-process runs are unaffected (every helper is a no-op when the world size is 1).
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def init() -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        dist.init_process_group("nccl")


def shutdown() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def world() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def is_main() -> bool:
    return rank() == 0


def _real(t: torch.Tensor) -> torch.Tensor:
    return torch.view_as_real(t) if t.is_complex() else t


def average_grads(params) -> None:
    """Average .grad over ranks (one flat all-reduce; complex grads through their real view)."""
    if world() == 1:
        return
    grads = [_real(p.grad) for p in params if p.grad is not None]
    if not grads:
        return
    flat = _flatten_dense_tensors(grads)
    dist.all_reduce(flat)
    flat /= world()
    for g, s in zip(grads, _unflatten_dense_tensors(flat, grads)):
        g.copy_(s)


@torch.no_grad()
def broadcast_module(module: torch.nn.Module) -> None:
    """Copy rank 0's parameters and buffers to every rank."""
    if world() == 1:
        return
    for t in list(module.parameters()) + list(module.buffers()):
        dist.broadcast(_real(t.data), 0)


@torch.no_grad()
def average_buffers(module: torch.nn.Module) -> None:
    """Average floating-point buffers (e.g. running feature statistics) over ranks."""
    if world() == 1:
        return
    for b in module.buffers():
        if b.is_floating_point() or b.is_complex():
            x = _real(b.data)
            dist.all_reduce(x)
            x /= world()
