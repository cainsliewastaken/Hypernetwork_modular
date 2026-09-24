"""EM self-anchoring loop (landscape §7d) — the core engine.

E-step: regenerate targets with each solve anchored at (and warm-started from) the hypernetwork's
        own current prediction.  M-step: re-regress the hypernetwork onto them.
Alternating minimization of sum_i [task(base + ΔW_i) + lam ||ΔW_i - H(x_i)||^2]; the fixed point is
the deepest smooth family of updates supported by the data density. Stop on the alarms.
"""
from __future__ import annotations

import copy

from .diagnostics import neighbor_pairs, target_stats, three_gap
from .distill import distill_ensemble
from .oracle import generate_targets


def run_em(pipe, rounds: int | None = None, start_round: int = 0, tag: str = "em") -> list[dict]:
    cfg = pipe.cfg.em
    rounds = cfg.rounds if rounds is None else rounds
    indices = pipe.target_indices()
    pairs = neighbor_pairs(pipe, indices)
    history, prev = [], pipe.state.get("em_last")
    best_score, best_state = float("inf"), None

    for r in range(start_round, start_round + rounds):
        lam = cfg.lam_schedule[min(r, len(cfg.lam_schedule) - 1)] if cfg.lam_schedule else pipe.cfg.oracle.lam
        anchor_fn = None if r == 0 else (lambda b: pipe.hyper(pipe.task.condition(b)))
        store = generate_targets(pipe, indices, anchor_fn=anchor_fn, lam=lam, seed=r, round_idx=r)
        rec = {"round": r, "lam": lam, "n_targets": len(store), **target_stats(store, pairs)}

        fresh = (r == 0) or not pipe.cfg.distill.warm_start
        dm = distill_ensemble(pipe, store, fresh=fresh, seed_base=1000 * (r + 1))
        rec["target_rel_train"] = sum(d["target_rel_train"] for d in dm) / len(dm)
        rec["target_rel_val"] = sum(d["target_rel_val"] for d in dm) / len(dm)

        train = pipe.evaluate(indices=store.indices.tolist())["cond"]
        test = pipe.evaluate(split=pipe.select_split)["cond"] if pipe.select_split else None
        rec.update(three_gap(rec["depth"], train, test if test is not None else train))
        if prev is not None and prev.get("depth"):
            rec["increment"] = (prev["depth"] - rec["depth"]) / abs(prev["depth"])
        alarms = pipe.alarms.check(rec, prev)
        rec["alarms"] = alarms
        pipe.logger.log(tag, **rec)
        for a in alarms:
            print(f"  ALARM: {a}", flush=True)

        pipe.save_targets(store)
        pipe.state["em_round"] = r
        pipe.state["em_last"] = rec
        score = test if test is not None else train
        if score < best_score:
            best_score, best_state = score, copy.deepcopy(pipe.hyper.state_dict())
        pipe.save("latest")
        history.append(rec)
        prev = rec

        converged = any("converged" in a for a in alarms)
        damaged = any(("threshold" in a) or ("gap opening" in a) for a in alarms)
        if damaged or converged:
            break

    if best_state is not None:
        pipe.hyper.load_state_dict(best_state)
    return history
