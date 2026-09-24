"""Shared-parameter training stages: base (§7a), meta/bi-level (§7b), joint two-timescale (§7f)."""
from __future__ import annotations

import copy

import torch

from .oracle import anchored_solve
from .utils import cycle, frozen, to_device


def _select_score(pipe, cond: bool):
    if not pipe.select_split:
        return None
    ev = pipe.evaluate(split=pipe.select_split)
    return ev["cond"] if cond else ev["base"]


def train_base(pipe):
    """Plain wheel training (theta=None). Regime allocation belongs in Task.make_bank(purpose="train")."""
    cfg = pipe.cfg.base
    if cfg.steps <= 0:
        return
    opt = torch.optim.AdamW(pipe.wheel.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps)
    it = cycle(pipe.loader(pipe.train_ds, cfg.batch_size, shuffle=True, drop_last=True))
    best, best_state = float("inf"), copy.deepcopy(pipe.wheel.state_dict())
    for step in range(1, cfg.steps + 1):
        pipe.wheel.train()
        b = to_device(next(it), pipe.device)
        loss = pipe.task.loss(pipe.cw, None, b, pipe.train_bank(cfg.bank_size, step)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(pipe.wheel.parameters(), cfg.grad_clip)
        opt.step(); sched.step()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            score = _select_score(pipe, cond=False)
            pipe.logger.log("base", step=step, train_loss=loss.item(), val_base=score)
            if score is None or score < best:
                best, best_state = (score if score is not None else best), copy.deepcopy(pipe.wheel.state_dict())
    pipe.wheel.load_state_dict(best_state)
    pipe.space.rebuild(pipe.wheel)


def train_meta(pipe):
    """First-order bi-level refinement: make the base the best ANCHOR for adapted models.
    Inner: short anchored solve from the base. Outer: gradient of the adapted loss w.r.t. base
    weights with the adapted coefficients held fixed (first-order). Anchor-optimal != standalone-
    optimal: expect the standalone score to possibly degrade while the adapted score improves."""
    cfg = pipe.cfg.meta
    if cfg.steps <= 0:
        return
    opt = torch.optim.Adam(pipe.wheel.parameters(), lr=cfg.lr)
    it = cycle(pipe.loader(pipe.train_ds, cfg.batch_size, shuffle=True, drop_last=True))
    for step in range(1, cfg.steps + 1):
        b = to_device(next(it), pipe.device)
        bank = pipe.train_bank(cfg.bank_size, 10_000_000 + step)
        zero = torch.zeros(pipe.task.condition(b).shape[0], pipe.space.dim, device=pipe.device)
        pipe.wheel.eval()
        theta, adapted = anchored_solve(pipe.cw, pipe.task, b, bank, zero, cfg.lam, cfg.inner_steps, cfg.inner_lr)
        pipe.wheel.train()
        loss = pipe.task.loss(pipe.cw, theta, b, bank).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            pipe.logger.log("meta", step=step, adapted_loss=loss.item(), standalone_val=_select_score(pipe, cond=False))
    pipe.space.rebuild(pipe.wheel)  # valid: no targets exist yet


def train_joint(pipe):
    """Two-timescale joint finisher. Both networks train on the task loss; wheel lr is 3-10x below
    the hypernet's; H is L2-anchored to its own distilled predictions; save-on-best. Monitor the
    wheel's relative displacement (too low = not engaged) and its standalone score (should barely move)."""
    cfg = pipe.cfg.joint
    if cfg.steps <= 0:
        return
    anchor_h = copy.deepcopy(pipe.hyper).eval()
    for p in anchor_h.parameters():
        p.requires_grad_(False)
    w0 = [p.detach().clone() for p in pipe.wheel.parameters()]
    w0_norm = torch.sqrt(sum(p.pow(2).sum() for p in w0)).clamp_min(1e-12)
    opt = torch.optim.Adam([
        {"params": pipe.hyper.parameters(), "lr": cfg.hyper_lr},
        {"params": pipe.wheel.parameters(), "lr": cfg.hyper_lr / cfg.wheel_lr_ratio},
    ])
    it = cycle(pipe.loader(pipe.train_ds, cfg.batch_size, shuffle=True, drop_last=True))
    best = _select_score(pipe, cond=True)
    best = float("inf") if best is None else best
    best_state = (copy.deepcopy(pipe.wheel.state_dict()), copy.deepcopy(pipe.hyper.state_dict()))
    pipe.logger.log("joint", step=0, val_cond=best)
    params = list(pipe.hyper.parameters()) + list(pipe.wheel.parameters())
    for step in range(1, cfg.steps + 1):
        pipe.wheel.train(); pipe.hyper.train()
        b = to_device(next(it), pipe.device)
        x = pipe.task.condition(b)
        th = pipe.hyper(x)
        with torch.no_grad():
            th0 = anchor_h(x)
        task_l = pipe.task.loss(pipe.cw, th, b, pipe.train_bank(cfg.bank_size, 20_000_000 + step)).mean()
        loss = task_l + cfg.anchor_weight * (th - th0).pow(2).sum(1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            disp = torch.sqrt(sum((p.detach() - q).pow(2).sum() for p, q in zip(pipe.wheel.parameters(), w0))) / w0_norm
            ev = pipe.evaluate(split=pipe.select_split) if pipe.select_split else None
            pipe.logger.log("joint", step=step, task_loss=task_l.item(), wheel_displacement=disp,
                            val_cond=ev and ev["cond"], val_wheel_standalone=ev and ev["base"])
            score = ev["cond"] if ev else -step
            if score < best:
                best = score
                best_state = (copy.deepcopy(pipe.wheel.state_dict()), copy.deepcopy(pipe.hyper.state_dict()))
    pipe.wheel.load_state_dict(best_state[0])
    pipe.hyper.load_state_dict(best_state[1])
