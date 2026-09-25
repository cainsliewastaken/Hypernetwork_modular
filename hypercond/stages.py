"""Shared-parameter training stages: base (§7a), meta/bi-level (§7b), joint two-timescale (§7f)."""
from __future__ import annotations

import copy
import math
import time

import torch

from . import distributed as D
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
    wu = max(0, int(cfg.warmup_steps))

    def lr_factor(i):  # i = number of scheduler steps taken
        if wu and i < wu:
            return (i + 1) / wu
        if cfg.schedule == "constant":
            return 1.0
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (i - wu) / max(1, cfg.steps - wu))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    it = cycle(pipe.loader(pipe.train_ds, cfg.batch_size, shuffle=True, drop_last=True))
    best, best_state = float("inf"), copy.deepcopy(pipe.wheel.state_dict())
    score = _select_score(pipe, cond=False)
    print(f"[base] step 0: val={score}", flush=True)
    run_loss, n_run, t0 = 0.0, 0, time.time()
    for step in range(1, cfg.steps + 1):
        pipe.wheel.train()
        b = to_device(next(it), pipe.device)
        loss = pipe.task.loss(pipe.cw, None, b, pipe.train_bank(cfg.bank_size, step)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        D.average_grads(pipe.wheel.parameters())
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(pipe.wheel.parameters(), cfg.grad_clip)
        opt.step(); sched.step()
        run_loss, n_run = run_loss + loss.item(), n_run + 1
        if step % cfg.eval_every == 0 or step == cfg.steps:
            score = _select_score(pipe, cond=False)
            improved = score is None or score < best
            print(f"[base] step {step}/{cfg.steps} ({time.time() - t0:.0f}s): train={run_loss / n_run:.6g} "
                  f"val={score:.6g} lr={sched.get_last_lr()[0]:.3g}{' *' if improved else ''}", flush=True)
            pipe.logger.log("base", step=step, train_loss=run_loss / n_run, val_base=score, quiet=True)
            run_loss, n_run = 0.0, 0
            if improved:
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


def train_direct(pipe):
    """Direct hypernet training: wheel frozen, H trained on the task loss with a fresh train bank every
    step; save-on-best on the selection split (base vs conditioned on the fixed eval bank)."""
    cfg = pipe.cfg.direct
    if cfg.steps <= 0:
        return
    mwd = getattr(cfg, "mode_weight_decay", -1.0)
    if mwd >= 0:
        named = list(pipe.hyper.named_parameters())
        groups = [{"params": [p for n, p in named if "mode_nets" not in n], "weight_decay": cfg.weight_decay},
                  {"params": [p for n, p in named if "mode_nets" in n], "weight_decay": mwd}]
        opt = torch.optim.AdamW(groups, lr=cfg.lr)
    else:
        opt = torch.optim.AdamW(pipe.hyper.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    wu, floor = max(0, int(cfg.warmup_steps)), cfg.end_lr / cfg.lr if cfg.lr > 0 else 0.0

    def lr_factor(i):
        if wu and i < wu:
            return (i + 1) / wu
        if cfg.schedule == "cosine":
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, (i - wu) / max(1, cfg.steps - wu))))
        return 1.0

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    it = cycle(pipe.loader(pipe.train_ds, cfg.batch_size, shuffle=True, drop_last=True))

    def score():
        ev = pipe.evaluate(split=pipe.select_split) if pipe.select_split else pipe.evaluate(indices=list(range(len(pipe.train_ds))))
        return ev

    n_tr = min(int(getattr(cfg, "train_eval_n", 0)), len(pipe.train_ds))
    tr_idx = [int(i * len(pipe.train_ds) / n_tr) for i in range(n_tr)] if n_tr > 0 else None

    def train_score():
        if tr_idx is None:
            return None
        return pipe.evaluate(indices=tr_idx, max_batches=0)

    ev = score()
    best, best_state = ev["cond"], copy.deepcopy(pipe.hyper.state_dict())
    print(f"[direct] step 0: val base={ev['base']:.6g} cond={ev['cond']:.6g} margin={ev.get('margin_rel', 0):+.2%}", flush=True)
    pipe.logger.log("direct", step=0, val_base=ev["base"], val_cond=ev["cond"], margin_rel=ev.get("margin_rel"))
    run_loss, n_run, t0 = 0.0, 0, time.time()
    with frozen(pipe.wheel):
        for step in range(1, cfg.steps + 1):
            pipe.hyper.train()
            b = to_device(next(it), pipe.device)
            th = pipe.hyper(pipe.task.condition(b))
            loss = pipe.task.loss(pipe.cw, th, b, pipe.train_bank(cfg.bank_size, 30_000_000 + step)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            D.average_grads(pipe.hyper.parameters())
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(pipe.hyper.parameters(), cfg.grad_clip)
            opt.step(); sched.step()
            run_loss, n_run = run_loss + loss.item(), n_run + 1
            if step % cfg.eval_every == 0 or step == cfg.steps:
                D.average_buffers(pipe.hyper)  # feature-normalizer stats: identical on every rank
                ev = score()
                tr = train_score()
                improved = ev["cond"] < best
                if improved:
                    best, best_state = ev["cond"], copy.deepcopy(pipe.hyper.state_dict())
                    pipe.save("best_direct")  # on disk, so a stopped run keeps its best weights
                trs = (f" | train-set base={tr['base']:.6g} cond={tr['cond']:.6g} margin={tr.get('margin_rel', 0):+.2%}"
                       if tr else "")
                print(f"[direct] step {step}/{cfg.steps} ({time.time() - t0:.0f}s): train={run_loss / n_run:.6g} "
                      f"val base={ev['base']:.6g} cond={ev['cond']:.6g} margin={ev.get('margin_rel', 0):+.2%} "
                      f"lr={sched.get_last_lr()[0]:.2g}{' *' if improved else ''}{trs}", flush=True)
                pipe.logger.log("direct", step=step, train_loss=run_loss / n_run, val_base=ev["base"],
                                val_cond=ev["cond"], margin_rel=ev.get("margin_rel"),
                                train_base=tr and tr["base"], train_cond=tr and tr["cond"],
                                train_margin=tr and tr.get("margin_rel"))
                run_loss, n_run = 0.0, 0
                pipe.save(f"direct_step{step:06d}")  # every check (scratch space is not a constraint)
    pipe.hyper.load_state_dict(best_state)


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
    w0_norm = torch.sqrt(sum(p.abs().pow(2).sum() for p in w0)).clamp_min(1e-12)  # abs: complex-safe
    opt = torch.optim.Adam([
        {"params": pipe.hyper.parameters(), "lr": cfg.hyper_lr},
        {"params": pipe.wheel.parameters(), "lr": cfg.hyper_lr / cfg.wheel_lr_ratio},
    ])
    it = cycle(pipe.loader(pipe.train_ds, cfg.batch_size, shuffle=True, drop_last=True))
    best = _select_score(pipe, cond=True)
    best = float("inf") if best is None else best
    print(f"[joint] step 0: val cond={best:.6g}", flush=True)
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
        dev2 = (th - th0).pow(2)
        red = getattr(cfg, "anchor_reduce", "sum")
        if red == "relative":   # squared drift relative to the anchor's own mean square (scale-free)
            anchor = dev2.mean(1) / th0.pow(2).mean(1).clamp_min(1e-20)
        elif red == "mean":
            anchor = dev2.mean(1)
        else:
            anchor = dev2.sum(1)
        loss = task_l + cfg.anchor_weight * anchor.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        D.average_grads(params)
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            D.average_buffers(pipe.hyper)
            disp = torch.sqrt(sum((p.detach() - q).abs().pow(2).sum() for p, q in zip(pipe.wheel.parameters(), w0))) / w0_norm
            ev = pipe.evaluate(split=pipe.select_split) if pipe.select_split else None
            pipe.logger.log("joint", step=step, task_loss=task_l.item(), wheel_displacement=disp,
                            val_cond=ev and ev["cond"], val_wheel_standalone=ev and ev["base"])
            score = ev["cond"] if ev else -step
            improved = score < best
            print(f"[joint] step {step}/{cfg.steps}: task={task_l.item():.6g} anchor={anchor.mean().item():.3g} "
                  f"val cond={ev and ev['cond']:.6g} wheel-standalone={ev and ev['base']:.6g} "
                  f"wheel_disp={float(disp):.3g}{' *' if improved else ''}", flush=True)
            pipe.save(f"joint_step{step:06d}")
            if improved:
                best = score
                best_state = (copy.deepcopy(pipe.wheel.state_dict()), copy.deepcopy(pipe.hyper.state_dict()))
                pipe.save("best_joint")
    pipe.wheel.load_state_dict(best_state[0])
    pipe.hyper.load_state_dict(best_state[1])
