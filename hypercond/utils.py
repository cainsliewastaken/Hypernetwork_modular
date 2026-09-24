from __future__ import annotations

import importlib
import importlib.util
import random
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    return obj


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed) % (2**63 - 1))
    return g


def load_object(spec: str):
    """Load "pkg.mod:Name" or "path/to/file.py:Name"."""
    if ":" not in spec:
        raise ValueError(f"Expected 'module:Name' or 'file.py:Name', got {spec!r}")
    mod_part, name = spec.rsplit(":", 1)
    if mod_part.endswith(".py"):
        path = Path(mod_part).resolve()
        mod_name = f"_hypercond_task_{path.stem}"
        spec_ = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(spec_)
        sys.modules[mod_name] = module
        spec_.loader.exec_module(module)
    else:
        module = importlib.import_module(mod_part)
    return getattr(module, name)


@contextmanager
def frozen(module: torch.nn.Module):
    """Hold a module exactly still (no grads) — required in any phase that consumes cached targets."""
    flags = [p.requires_grad for p in module.parameters()]
    for p in module.parameters():
        p.requires_grad_(False)
    try:
        yield module
    finally:
        for p, f in zip(module.parameters(), flags):
            p.requires_grad_(f)


def cycle(loader):
    while True:
        for b in loader:
            yield b


def rel_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Relative distance ||a-b|| / mean(||a||, ||b||), row-wise."""
    return (a - b).norm(dim=-1) / (0.5 * (a.norm(dim=-1) + b.norm(dim=-1))).clamp_min(1e-12)


def to_float(x):
    if torch.is_tensor(x):
        return x.item() if x.numel() == 1 else x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {k: to_float(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_float(v) for v in x]
    return x
