"""How much of a trained hypernet's margin is per-sample conditioning, and how much is one shared weight change?

For a checkpoint, on the fixed val set (the same points, tau draws and noise as training's checks), scores:
  base      the checkpoint's wheel alone (theta = 0)
  cond      the hypernet's per-field update dW(x)                         (the reported "cond")
  mean      the field-averaged update dW_bar (mean of dW(x) over `--n-mean` evenly spaced train fields), same for
            every field: a plain fine-tune of the wheel
  shuffled  dW(x') from a different val field (x' = the next field in the batch)
Conditioning benefit = mean - cond (> 0 means dW(x) beats the best shared update the hypernet itself implies).
Also reports, per updated tensor, the relative size |dW|/|W| and the mean field-to-field cosine of dW.

  srun ... shifter python -u experiments/conditioning_split.py model_checkpoints/it13_mh128_utau/best_direct.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hypercond.pipeline import Pipeline  # noqa: E402
from hypercond.utils import to_device  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--n-mean", type=int, default=512, help="train fields averaged for dW_bar")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    for path in args.ckpts:
        pipe = Pipeline.from_checkpoint(path, overrides={"out_dir": str(Path(path).parent / "cond_split")})
        pipe.wheel.eval(); pipe.hyper.eval()
        task, dev = pipe.task, pipe.device

        # dW_bar over evenly spaced train fields
        n_tr = len(pipe.train_ds)
        idx = [int(i * n_tr / args.n_mean) for i in range(args.n_mean)]
        s, n = None, 0
        for b in pipe.loader(pipe.train_ds, args.batch, indices=idx):
            th = pipe.hyper.predict(task.condition(to_device(b, dev)))
            s = th.sum(0) if s is None else s + th.sum(0)
            n += th.shape[0]
        th_bar = s / n

        sums = {"base": 0.0, "cond": 0.0, "mean": 0.0, "shuffled": 0.0}
        cnt, ths = 0, []
        for b in pipe.loader(pipe.datasets[pipe.select_split], args.batch):
            b = to_device(b, dev)
            th = pipe.hyper.predict(task.condition(b))
            B = th.shape[0]
            sums["base"] += float(task.loss(pipe.cw, None, b, pipe.eval_bank).sum())
            sums["cond"] += float(task.loss(pipe.cw, th, b, pipe.eval_bank).sum())
            sums["mean"] += float(task.loss(pipe.cw, th_bar.expand(B, -1), b, pipe.eval_bank).sum())
            sums["shuffled"] += float(task.loss(pipe.cw, th.roll(1, 0), b, pipe.eval_bank).sum())
            cnt += B
            if len(ths) < 4:
                ths.append(th.float().cpu())
        r = {k: v / cnt for k, v in sums.items()}
        b0 = r["base"]
        print(f"\n== {path}  ({cnt} val points)")
        for k in ("base", "mean", "cond", "shuffled"):
            print(f"  {k:<9} {r[k]:.6f}   margin vs base {1 - r[k] / b0:+.3%}")
        print(f"  conditioning benefit (mean - cond): {r['mean'] - r['cond']:+.6f} "
              f"= {(r['mean'] - r['cond']) / b0:+.3%} of base; shared part (base - mean): {(b0 - r['mean']) / b0:+.3%}")

        # per-tensor size and field-to-field cosine
        T = torch.cat(ths)
        params = dict(pipe.wheel.named_parameters())
        o = 0
        print(f"  {'tensor':<26}{'|dW|/|W|':>10}{'cos(fields)':>13}{'|dW - dW_bar|/|dW|':>20}")
        for e in pipe.space.entries:
            d = T[:, o:o + e["add_dim"]]
            db = th_bar[o:o + e["add_dim"]].float().cpu()
            o += e["add_dim"]
            p = params[e["name"]].detach().cpu()
            w = torch.view_as_real(p) if p.is_complex() else p
            dn = d / d.norm(dim=1, keepdim=True).clamp_min(1e-30)
            c = dn @ dn.T
            m = c.shape[0]
            cos = float((c.sum() - m) / (m * (m - 1)))
            dev_ = float(((d - db).norm(dim=1) / d.norm(dim=1).clamp_min(1e-30)).mean())
            print(f"  {e['name']:<26}{float(d.norm(dim=1).mean() / w.norm()):10.3f}{cos:13.4f}{dev_:20.4f}")


if __name__ == "__main__":
    main()
