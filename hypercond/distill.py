"""M-step: regress the hypernetwork onto anchored targets (landscape §7c/7d).

Plain L2 in coefficient space; early stopping on a time-blocked tail of the target set. Optional
semi-supervised task loss on untargeted train samples (dense patches get regression, the rest of
condition space gets the task loss).
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from .targets import IndexedDataset, TargetDataset, TargetStore
from .utils import cycle, frozen, to_device


def _target_mse(pipe, member, ds, bs):
    member.eval()
    se, ref, n = 0.0, 0.0, 0
    with torch.no_grad():
        for b in pipe.loader(ds, bs, shuffle=False):
            b = to_device(b, pipe.device)
            p, t = member(pipe.task.condition(b)), b["_target"]
            se += (p - t).pow(2).sum().item(); ref += t.pow(2).sum().item(); n += t.numel()
    return se / max(n, 1), se / max(ref, 1e-12)


def distill_member(pipe, member, store: TargetStore, seed: int, cfg=None) -> dict:
    cfg = cfg or pipe.cfg.distill
    torch.manual_seed(seed)
    n = len(store)
    nval = int(round(cfg.target_val_frac * n)) if n >= 10 else 0
    tr_rows, va_rows = torch.arange(0, n - nval), torch.arange(n - nval, n)
    ds_tr = TargetDataset(pipe.train_ds, store.indices[tr_rows], store.theta[tr_rows])
    ds_va = TargetDataset(pipe.train_ds, store.indices[va_rows], store.theta[va_rows]) if nval else ds_tr
    loader = pipe.loader(ds_tr, cfg.batch_size, shuffle=True, drop_last=len(ds_tr) > cfg.batch_size)

    unl_iter = None
    if cfg.task_loss_weight > 0:
        targeted = set(store.indices.tolist())
        rest = [i for i in range(len(pipe.train_ds)) if i not in targeted]
        if rest:
            unl_iter = cycle(pipe.loader(IndexedDataset(pipe.train_ds, rest), cfg.batch_size, shuffle=True))

    opt = torch.optim.AdamW(member.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best, best_state, bad, step = float("inf"), copy.deepcopy(member.state_dict()), 0, 0
    with frozen(pipe.wheel):
        for epoch in range(cfg.epochs):
            member.train()
            for b in loader:
                b = to_device(b, pipe.device)
                loss = F.mse_loss(member(pipe.task.condition(b)), b["_target"])
                if unl_iter is not None:
                    u = to_device(next(unl_iter), pipe.device)
                    th = member(pipe.task.condition(u))
                    loss = loss + cfg.task_loss_weight * pipe.task.loss(pipe.cw, th, u, pipe.train_bank(4, seed * 7919 + step)).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(member.parameters(), cfg.grad_clip)
                opt.step()
                step += 1
            val, _ = _target_mse(pipe, member, ds_va, cfg.batch_size)
            if val < best - 1e-12:
                best, best_state, bad = val, copy.deepcopy(member.state_dict()), 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
    member.load_state_dict(best_state)
    tr_mse, tr_rel = _target_mse(pipe, member, ds_tr, cfg.batch_size)
    va_mse, va_rel = _target_mse(pipe, member, ds_va, cfg.batch_size)
    return {"target_mse_train": tr_mse, "target_rel_train": tr_rel, "target_mse_val": va_mse,
            "target_rel_val": va_rel, "epochs": epoch + 1}


def distill_ensemble(pipe, store: TargetStore, fresh: bool, seed_base: int = 0) -> list[dict]:
    out = []
    for k in range(len(pipe.hyper.members)):
        if fresh:
            pipe.hyper.members[k] = pipe.build_member()
        out.append(distill_member(pipe, pipe.hyper.members[k], store, seed=seed_base + k))
    hc = pipe.cfg.hyper
    pipe.hyper.fit_trust_cap(store.theta, hc.trust_quantile, hc.trust_factor)
    return out
