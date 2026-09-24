"""Diagnostics (landscape §2/§4/§5, diagnostics doc §2–§5, headroom doc §4).

Every function here is method-agnostic and returns a plain dict suitable for the JSON report.
"""
from __future__ import annotations

import copy
import math

import torch

from .hypernet import HyperEnsemble
from .oracle import anchored_solve, generate_targets
from .targets import TargetStore
from .utils import rel_dist, to_device


# ---------------------------------------------------------------------------- neighbors
def gather_conditions(pipe, indices, max_n=None) -> torch.Tensor:
    idx = list(indices)[: max_n] if max_n else list(indices)
    out = []
    for b in pipe.loader(pipe.train_ds, 256, indices=idx, shuffle=False):
        out.append(pipe.task.condition(to_device(b, pipe.device)).flatten(1).float().cpu())
    return torch.cat(out) if out else torch.empty(0)


def _nearest(X: torch.Tensor, k: int = 1, chunk: int = 1024) -> torch.Tensor:
    """k-th nearest neighbor index for each row (excluding self)."""
    nn_idx = []
    for s in range(0, X.shape[0], chunk):
        d = torch.cdist(X[s: s + chunk], X)
        d[torch.arange(d.shape[0]), torch.arange(s, s + d.shape[0])] = math.inf
        nn_idx.append(d.topk(k, largest=False).indices[:, k - 1])
    return torch.cat(nn_idx)


def neighbor_pairs(pipe, indices) -> torch.Tensor:
    """[P, 2] pairs of original train indices that are neighbors at the data's similarity scale."""
    p = pipe.task.neighbor_pairs(pipe.train_ds, list(indices))
    if p is not None:
        return torch.as_tensor(p, dtype=torch.long)
    idx = torch.as_tensor(list(indices))[: pipe.cfg.em.knn_max]
    if idx.numel() < 2:
        return torch.empty(0, 2, dtype=torch.long)
    X = gather_conditions(pipe, idx.tolist())
    X = (X - X.mean(0)) / X.std(0).clamp_min(1e-8)
    j = _nearest(X, 1)
    return torch.stack([idx, idx[j]], dim=1)


def neighbor_distance(store: TargetStore, pairs: torch.Tensor) -> float | None:
    rows = store.rows()
    keep = [(rows[a], rows[b]) for a, b in pairs.tolist() if a in rows and b in rows]
    if not keep:
        return None
    ra, rb = map(torch.tensor, zip(*keep))
    return float(rel_dist(store.theta[ra], store.theta[rb]).median())


def target_stats(store: TargetStore, pairs) -> dict:
    return {"depth_solve": float(store.loss_solve.median()), "depth": float(store.loss_eval.median()),
            "neighbor_distance": neighbor_distance(store, pairs),
            "target_norm": float(store.theta.norm(dim=1).median())}


# ---------------------------------------------------------------------------- gaps
def three_gap(depth: float, train: float, test: float) -> dict:
    """held-out = target depth + fitting gap (train - depth) + generalization gap (test - train)."""
    return {"depth": depth, "train": train, "test": test,
            "fitting_gap": train - depth, "generalization_gap": test - train}


# ---------------------------------------------------------------------------- layer 1: selection noise
@torch.no_grad()
def _line_losses(pipe, batch, ta, tb, n=7):
    out = []
    for a in torch.linspace(0, 1, n):
        out.append(pipe.task.loss(pipe.cw, (1 - a) * ta + a * tb, batch, pipe.eval_bank))
    return torch.stack(out, 1)  # [B, n]


def twin_test(pipe, n: int = 64, jitter: float = 1e-2, lams=None, steps=None) -> dict:
    """Solve the SAME samples twice from jittered inits. Twin distance comparable to neighbor
    distance, with a loss barrier on the line between twins, means the scatter is selection noise
    (Layer 1), not signal. Run at several anchor strengths to see selection sharpen as anchors weaken."""
    lams = lams or [pipe.cfg.oracle.lam, pipe.cfg.oracle.lam * 0.01]
    idx = pipe.target_indices()[:n]
    batch = to_device(next(iter(pipe.loader(pipe.train_ds, len(idx), indices=idx, shuffle=False))), pipe.device)
    B = len(idx)
    zero = torch.zeros(B, pipe.space.dim, device=pipe.device)
    g = torch.Generator().manual_seed(0)
    res = {}
    for lam in lams:
        th = []
        for _ in range(2):
            init = zero + jitter * torch.randn(zero.shape, generator=g).to(pipe.device)
            t, _ = anchored_solve(pipe.cw, pipe.task, batch, pipe.solve_bank, zero, lam,
                                  steps or pipe.cfg.oracle.steps, pipe.cfg.oracle.lr, init=init)
            th.append(t)
        L = _line_losses(pipe, batch, th[0], th[1])
        barrier = (L[:, 1:-1].max(1).values - 0.5 * (L[:, 0] + L[:, -1])).clamp_min(0)
        res[f"lam={lam:g}"] = {"twin_distance": float(rel_dist(th[0], th[1]).median()),
                               "barrier_median": float(barrier.median()),
                               "barrier_rel": float((barrier / L[:, [0, -1]].mean(1).clamp_min(1e-12)).median())}
    return res


# ---------------------------------------------------------------------------- density law
def decorrelation_curve(pipe, store: TargetStore, bins: int = 10, k_max: int = 16, max_n: int = 2000) -> dict:
    """Target similarity vs condition similarity. The bend (where neighboring targets stop sharing
    structure) sets the spacing your data must beat. Re-measure per system and prediction lead."""
    idx = store.indices[:max_n]
    rows = store.rows()
    X = gather_conditions(pipe, idx.tolist())
    Xc = X - X.mean(1, keepdim=True)
    Xc = Xc / Xc.norm(dim=1, keepdim=True).clamp_min(1e-12)
    T = store.theta[[rows[int(i)] for i in idx]]
    Tn = T / T.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Xs = (X - X.mean(0)) / X.std(0).clamp_min(1e-8)
    a_list, b_list = [], []
    for k in sorted({1, 2, 4, 8, k_max}):
        if k < X.shape[0]:
            j = _nearest(Xs, k)
            a_list.append(torch.arange(X.shape[0])); b_list.append(j)
    g = torch.Generator().manual_seed(0)
    a_list.append(torch.randint(0, X.shape[0], (X.shape[0],), generator=g))
    b_list.append(torch.randint(0, X.shape[0], (X.shape[0],), generator=g))
    a, b = torch.cat(a_list), torch.cat(b_list)
    keep = a != b
    a, b = a[keep], b[keep]
    cc = (Xc[a] * Xc[b]).sum(1)
    ts = (Tn[a] * Tn[b]).sum(1)
    edges = torch.quantile(cc, torch.linspace(0, 1, bins + 1))
    table = []
    for i in range(bins):
        m = (cc >= edges[i]) & (cc <= edges[i + 1])
        if m.any():
            table.append({"cond_corr": float(cc[m].mean()), "target_cos": float(ts[m].mean()), "n": int(m.sum())})
    bend = None
    if table:
        peak = max(r["target_cos"] for r in table)
        above = [r for r in table if r["target_cos"] >= 0.5 * peak]
        bend = min(r["cond_corr"] for r in above) if above else None
    return {"table": table, "bend_cond_corr": bend,
            "note": "bend = lowest condition correlation whose target similarity is still >= half the peak"}


def annealing_ladder(pipe, lams: list[float], distill: bool = False, n: int | None = None) -> list[dict]:
    """Depth-smoothness law: progressively weaker anchors, warm-started per rung. Log depth and
    neighbor-target distance per rung (and optionally held-out after a fresh distillation) to locate
    YOUR neighbor-distance threshold."""
    from .distill import distill_member
    idx = pipe.target_indices()[: n] if n else pipe.target_indices()
    pairs = neighbor_pairs(pipe, idx)
    prev, out = None, []
    for r, lam in enumerate(sorted(lams, reverse=True)):
        store = generate_targets(pipe, idx, lam=lam, init_store=prev, round_idx=r)
        rec = {"lam": lam, **target_stats(store, pairs)}
        if distill:
            m = pipe.build_member()
            rec.update(distill_member(pipe, m, store, seed=r))
            ens = HyperEnsemble([m]).to(pipe.device)
            ens.fit_trust_cap(store.theta, pipe.cfg.hyper.trust_quantile, pipe.cfg.hyper.trust_factor)
            rec["train"] = pipe.evaluate(indices=store.indices.tolist(), hyper=ens)["cond"]
            if pipe.select_split:
                rec["test"] = pipe.evaluate(split=pipe.select_split, hyper=ens)["cond"]
        out.append(rec)
        pipe.logger.log("ladder", **rec)
        prev = store
    return out


# ---------------------------------------------------------------------------- headroom gate
def headroom_gate(pipe, n: int | None = None, steps_mult: float | None = None) -> dict:
    """Per-sample anchored oracle vs trained base at this wheel size, with the deployment plumbing.
    Predicts the conditioning margin before any hypernetwork is trained."""
    ec = pipe.cfg.eval
    n = n or ec.gate_samples
    steps = int(pipe.cfg.oracle.steps * (steps_mult or ec.gate_steps_mult))
    idx = pipe.target_indices()[:n]
    store = generate_targets(pipe, idx, steps=steps)
    base, oracle, shuf = 0.0, 0.0, 0.0
    rows = store.rows()
    with torch.no_grad():
        for b in pipe.loader(pipe.train_ds, pipe.cfg.oracle.batch_size, indices=idx, shuffle=False):
            th = store.theta[[rows[int(i)] for i in b["_idx"]]].to(pipe.device)
            b = to_device(b, pipe.device)
            base += pipe.task.loss(pipe.cw, None, b, pipe.eval_bank).sum().item()
            oracle += pipe.task.loss(pipe.cw, th, b, pipe.eval_bank).sum().item()
            shuf += pipe.task.loss(pipe.cw, th.roll(1, 0), b, pipe.eval_bank).sum().item()
    N = len(idx)
    base, oracle, shuf = base / N, oracle / N, shuf / N
    ratio = oracle / max(base, 1e-12)
    ciw = pipe.task.condition_in_wheel
    if ratio > 0.95:
        verdict = ("oracle ~ base: task saturated at this wheel size, or update space too restricted "
                   "(re-check with update_space.additive=dense). Spend on a bigger wheel; a null hypernet "
                   "result here is not a verdict on the method.")
    elif ciw is True:
        verdict = "oracle << base with condition in wheel: pure Channel-2 (capacity localization) headroom. Full stack applies."
    elif ciw is False:
        verdict = ("oracle << base, condition not in wheel: mixed channels. Compare the shuffle control and a "
                   "concat variant to separate information from capacity localization.")
    else:
        verdict = "oracle << base: headroom exists. Set Task.condition_in_wheel to attribute it to a channel."
    return {"base": base, "oracle": oracle, "shuffled_oracle": shuf, "oracle_over_base": ratio,
            "shuffled_over_base": shuf / max(base, 1e-12), "steps": steps, "n": N, "verdict": verdict}


# ---------------------------------------------------------------------------- guidance diagnostic
def guidance_sweep(pipe, scales=None, split=None) -> dict:
    """Five-minute scale sweep on s * theta. Fires only if hypernet training used heavy shrinkage;
    if it fires, distill the scale into the weights — never pay extra wheel evaluations."""
    scales = scales or pipe.cfg.eval.guidance_scales
    split = split or pipe.select_split or "train"
    res = {f"{s:g}": pipe.evaluate(split=split, scale=s)["cond"] for s in scales}
    best = min(res, key=res.get)
    fires = best != "1" and res[best] < 0.99 * res.get("1", math.inf)
    return {"loss_by_scale": res, "best_scale": float(best), "fires": bool(fires),
            "verdict": ("under-expressed conditioning: use guidance DISTILLATION (bake s* into the update)"
                        if fires else "amplitudes calibrated; no guidance")}


# ---------------------------------------------------------------------------- two-axis probe
def _fresh_eval(pipe, store: TargetStore, member, seed: int) -> dict:
    from .distill import distill_member
    rec = distill_member(pipe, member, store, seed=seed)
    ens = HyperEnsemble([member]).to(pipe.device)
    ens.fit_trust_cap(store.theta, pipe.cfg.hyper.trust_quantile, pipe.cfg.hyper.trust_factor)
    rec["train"] = pipe.evaluate(indices=store.indices.tolist(), hyper=ens)["cond"]
    if pipe.select_split:
        rec["test"] = pipe.evaluate(split=pipe.select_split, hyper=ens)["cond"]
    return rec


def learning_curve_probe(pipe, store: TargetStore, fracs=(0.5, 0.75, 1.0)) -> dict:
    """Data axis: retrain on contiguous prefixes (same spacing distribution, fewer targets).
    Held-out still sloping in n -> data-limited; buying capacity now is wasted."""
    out = []
    for f in fracs:
        k = max(2, int(round(f * len(store))))
        rec = {"frac": f, "n": k, **_fresh_eval(pipe, store.subset(torch.arange(k)), pipe.build_member(), seed=int(f * 1000))}
        out.append(rec)
        pipe.logger.log("learning_curve", **rec)
    slope = None
    if pipe.select_split and len(out) >= 2:
        slope = (out[0]["test"] - out[-1]["test"]) / max(abs(out[-1]["test"]), 1e-12)
    return {"points": out, "rel_improvement_first_to_last": slope,
            "verdict": None if slope is None else ("data-limited (still sloping in n)" if slope > 0.02 else "flat in n: not data-limited")}


def capacity_probe(pipe, store: TargetStore, axes=("depth", "encoder_dim", "width", "head_rank"), factors=(0.5, 2.0)) -> dict:
    """Model axis: halve/double each capacity axis separately at fixed data. The axis that moves BOTH
    train and test is where expressivity binds. Only paired train/test movement is diagnostic."""
    hc = pipe.cfg.hyper
    ref = {"setting": "reference", **_fresh_eval(pipe, store, pipe.build_member(), seed=0)}
    rows = [ref]
    for ax in axes:
        base_val = getattr(hc, ax)
        if not base_val:
            continue
        for f in factors:
            v = max(1, int(round(base_val * f)))
            rec = {"setting": f"{ax}={v}", **_fresh_eval(pipe, store, pipe.build_member(**{ax: v}), seed=1)}
            rec["d_train"] = rec["train"] - ref["train"]
            if "test" in rec:
                rec["d_test"] = rec["test"] - ref["test"]
            rows.append(rec)
            pipe.logger.log("capacity_probe", **rec)
    return {"rows": rows}


# ---------------------------------------------------------------------------- FLOPs
def flop_report(pipe) -> dict:
    """epsilon = hypernet forward / (K * wheel forward): the size of the matched-FLOP competitor
    wheel S*(1+epsilon). Requires Task.wheel_example_inputs."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        b = to_device(next(iter(pipe.loader(pipe.train_ds, 1, shuffle=False))), pipe.device)
        s, args = pipe.task.wheel_example_inputs(b)
    except NotImplementedError:
        return {"available": False, "note": "implement Task.wheel_example_inputs for FLOP accounting"}
    with torch.no_grad():
        th = pipe.hyper(pipe.task.condition(b))
        counts = {}
        for name, fn in [("wheel", lambda: pipe.cw(None, s, *args)),
                         ("conditioned_wheel", lambda: pipe.cw(th, s, *args)),
                         ("hypernet", lambda: pipe.hyper(pipe.task.condition(b)))]:
            try:
                with FlopCounterMode(display=False) as fc:
                    fn()
                counts[name] = int(fc.get_total_flops())
            except Exception as e:  # pragma: no cover
                counts[name] = None
                counts[f"{name}_error"] = repr(e)
    K = pipe.task.solver_steps
    out = {"available": True, "solver_steps": K, **counts}
    if counts.get("wheel") and counts.get("hypernet") is not None:
        out["epsilon"] = counts["hypernet"] / (K * counts["wheel"])
        if counts.get("conditioned_wheel"):
            out["per_step_update_overhead"] = counts["conditioned_wheel"] / counts["wheel"] - 1
        out["note"] = ("compare against a plain wheel of size S*(1+epsilon+overhead) trained on the stack's full "
                       "training budget (the compute-matched control)")
    return out
