"""Anchored proximal oracle solves (landscape §3).

    minimize_theta  task(base + ΔW(theta); x_i, bank) + lam * ||theta - anchor_i||^2

Short, identical budgets from the anchor, on a fixed common-random-number bank, with the wheel held
exactly still. Batched: samples are independent (per-sample parameters, element-wise Adam), so one
optimizer over the [B, dim] tensor is B independent solves.
"""
from __future__ import annotations

import torch

from .targets import TargetStore
from .utils import frozen, make_generator, to_device


def anchored_solve(model, task, batch, bank, anchor, lam, steps, lr, init=None):
    with frozen(model.wheel):
        theta = (anchor if init is None else init).detach().clone().requires_grad_(True)
        anchor = anchor.detach()
        opt = torch.optim.Adam([theta], lr=lr)
        for _ in range(int(steps)):
            opt.zero_grad(set_to_none=True)
            obj = task.loss(model, theta, batch, bank) + lam * (theta - anchor).pow(2).sum(1)
            obj.sum().backward()
            opt.step()
        with torch.no_grad():
            final = task.loss(model, theta, batch, bank)
    return theta.detach(), final.detach()


def generate_targets(pipe, indices, anchor_fn=None, lam=None, steps=None, lr=None, jitter=0.0, seed=0,
                     round_idx=0, init_store: TargetStore | None = None) -> TargetStore:
    """Solve targets for train `indices`.

    anchor_fn(batch) -> [B, dim] anchor (None = the base, i.e. zeros). Solves are warm-started at the
    anchor, or at `init_store` rows when given (annealing ladders), plus optional jitter (twin test).
    """
    oc = pipe.cfg.oracle
    lam = oc.lam if lam is None else lam
    steps = oc.steps if steps is None else steps
    lr = oc.lr if lr is None else lr
    gen = make_generator(seed)
    init_rows = init_store.rows() if init_store is not None else None
    was_training = pipe.wheel.training
    pipe.wheel.eval()
    pipe.hyper.eval()
    idx_all, th_all, ls_all, le_all = [], [], [], []
    for batch in pipe.loader(pipe.train_ds, oc.batch_size, indices=indices, shuffle=False):
        idx = batch["_idx"].clone()
        batch = to_device(batch, pipe.device)
        B = idx.numel()
        with torch.no_grad():
            anchor = anchor_fn(batch).detach() if anchor_fn is not None else torch.zeros(B, pipe.space.dim, device=pipe.device)
        init = anchor
        if init_rows is not None:
            init = init_store.theta[[init_rows[int(i)] for i in idx]].to(pipe.device)
        if jitter > 0:
            init = init + jitter * torch.randn(init.shape, generator=gen).to(pipe.device)
        theta, l_solve = anchored_solve(pipe.cw, pipe.task, batch, pipe.solve_bank, anchor, lam, steps, lr, init=init)
        with torch.no_grad():
            l_eval = pipe.task.loss(pipe.cw, theta, batch, pipe.eval_bank)
        idx_all.append(idx); th_all.append(theta.cpu()); ls_all.append(l_solve.cpu()); le_all.append(l_eval.cpu())
    pipe.wheel.train(was_training)
    idx = torch.cat(idx_all)
    order = torch.argsort(idx)
    return TargetStore(idx[order], torch.cat(th_all)[order], torch.cat(ls_all)[order], torch.cat(le_all)[order],
                       round=round_idx, meta={"lam": lam, "steps": steps, "lr": lr, "jitter": jitter})
