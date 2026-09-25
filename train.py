#!/usr/bin/env python
"""Train the full stack: base -> meta -> headroom gate -> EM (distill + self-anchoring) -> joint finisher.

Examples
  python train.py --config configs/default.yaml --set task=my_pkg.tasks:MyTask out_dir=model_checkpoints/exp1
  python train.py --resume model_checkpoints/exp1/latest.pt                     # continue unfinished stages
  python train.py --config configs/default.yaml --set stages='[base,gate]'   # headroom gate only
  torchrun --standalone --nproc-per-node=4 train.py ...   # data-parallel base / direct stages (batch sizes per GPU)
"""
import argparse
import os
import sys

from hypercond.config import apply_overrides, load_config
from hypercond import distributed as D
from hypercond.pipeline import Pipeline


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML/JSON config file")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted overrides, e.g. em.rounds=4")
    ap.add_argument("--resume", default=None, help="checkpoint to resume (stages already completed are skipped)")
    ap.add_argument("--no-final-eval", action="store_true")
    args = ap.parse_args()

    D.init()  # torchrun: one process per GPU; no-op for single-process runs
    if not D.is_main():
        sys.stdout = open(os.devnull, "w")

    if args.resume:
        pipe = Pipeline.from_checkpoint(args.resume, overrides=apply_overrides({}, args.set))
    else:
        cfg = load_config(args.config, args.set)
        if not cfg.task:
            ap.error("config must set `task` (e.g. --set task=my_pkg.tasks:MyTask)")
        pipe = Pipeline(cfg)

    pipe.run()

    if not args.no_final_eval:
        report = {"splits": pipe.evaluate_all(), "gate": pipe.state["reports"].get("gate"),
                  "em_last": pipe.state.get("em_last")}
        pipe.write_report("train_report.json", report)
        for split, r in report["splits"].items():
            print(f"[final] {split}: base={r['base']:.6g} cond={r['cond']:.6g} "
                  f"margin={r.get('margin_rel', 0):+.2%} shuffled={r.get('shuffled', float('nan')):.6g}")
    D.shutdown()


if __name__ == "__main__":
    main()
