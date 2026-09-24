#!/usr/bin/env python
"""Evaluate a checkpoint and run diagnostics.

Always reported, per split (time-blocked splits from the task): base loss, conditioned loss with
deployment guards installed, shuffle control (updates from mismatched conditions), relative margin,
and any deliverable metrics from Task.evaluate (e.g. sampled rollout MSE). Plus the three-gap
decomposition against the latest targets.

Diagnostics (--diag, any subset):
  gate           headroom gate: anchored oracle vs base (+ shuffle), with channel verdict
  twin           twin test: selection noise (Layer 1) at several anchor strengths
  decorrelation  target similarity vs condition similarity; the spacing bend
  ladder         annealing ladder: depth / neighbor distance per anchor strength (--ladder-lams)
  guidance       scale sweep on the update (should be flat-optimal at 1)
  learning_curve data-axis probe on prefixes of the target set
  capacity       model-axis probe: halve/double depth, encoder_dim, width, head_rank
  flops          epsilon = hypernet / (K * wheel): size of the matched-FLOP competitor
  all            everything above

  python eval.py --ckpt runs/exp1/latest.pt --diag gate guidance flops
"""
import argparse

from hypercond import diagnostics as D
from hypercond.config import apply_overrides
from hypercond.pipeline import Pipeline

ALL = ["gate", "twin", "decorrelation", "ladder", "guidance", "learning_curve", "capacity", "flops"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--diag", nargs="*", default=[], choices=ALL + ["all"])
    ap.add_argument("--ladder-lams", nargs="*", type=float, default=[10.0, 3.0, 1.0, 0.3, 0.1, 0.03])
    ap.add_argument("--ladder-distill", action="store_true", help="also distill per rung (locates the threshold)")
    ap.add_argument("--out", default="eval_report.json")
    args = ap.parse_args()

    pipe = Pipeline.from_checkpoint(args.ckpt, overrides=apply_overrides({}, args.set))
    diags = ALL if "all" in args.diag else args.diag
    rep = {"splits": pipe.evaluate_all(shuffle_control=True, extra_metrics=True)}

    store = pipe.load_targets()
    if store is not None:
        train = pipe.evaluate(indices=store.indices.tolist())["cond"]
        test = pipe.evaluate(split=pipe.select_split)["cond"] if pipe.select_split else train
        pairs = D.neighbor_pairs(pipe, store.indices.tolist())
        rep["targets"] = D.target_stats(store, pairs)
        rep["three_gap"] = D.three_gap(rep["targets"]["depth"], train, test)

    for name in diags:
        print(f"=== diagnostic: {name} ===", flush=True)
        if name in ("decorrelation", "learning_curve", "capacity") and store is None:
            rep[name] = {"skipped": "no targets in checkpoint (run the em stage)"}
        elif name == "gate":
            rep[name] = D.headroom_gate(pipe)
        elif name == "twin":
            rep[name] = D.twin_test(pipe)
        elif name == "decorrelation":
            rep[name] = D.decorrelation_curve(pipe, store)
        elif name == "ladder":
            rep[name] = D.annealing_ladder(pipe, args.ladder_lams, distill=args.ladder_distill)
        elif name == "guidance":
            rep[name] = D.guidance_sweep(pipe)
        elif name == "learning_curve":
            rep[name] = D.learning_curve_probe(pipe, store)
        elif name == "capacity":
            rep[name] = D.capacity_probe(pipe, store)
        elif name == "flops":
            rep[name] = D.flop_report(pipe)
        pipe.logger.log(f"diag_{name}", quiet=True, result=rep[name])

    pipe.write_report(args.out, rep)
    for split, r in rep["splits"].items():
        print(f"{split:>12}: base={r['base']:.6g} cond={r['cond']:.6g} margin={r.get('margin_rel', 0):+.2%} "
              f"shuffled={r.get('shuffled', float('nan')):.6g}")
    if "three_gap" in rep:
        g = rep["three_gap"]
        print(f"three-gap: depth={g['depth']:.6g} fitting_gap={g['fitting_gap']:+.4g} generalization_gap={g['generalization_gap']:+.4g}")
    for k in diags:
        v = rep.get(k, {})
        if isinstance(v, dict) and "verdict" in v and v["verdict"]:
            print(f"{k}: {v['verdict']}")
    print(f"report -> {pipe.out / args.out}")


if __name__ == "__main__":
    main()
