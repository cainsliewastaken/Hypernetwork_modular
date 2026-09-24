"""End-to-end smoke test on the synthetic toy task: every stage, checkpoint/resume, all diagnostics.
Run:  python -m pytest -q tests   (or: python tests/test_smoke.py)"""
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hypercond.config import load_config  # noqa: E402
from hypercond.pipeline import Pipeline  # noqa: E402

SMALL = [
    f"task={ROOT / 'tests' / 'toy_task.py'}:ToyTask", "device=cpu",
    "hyper.width=64", "hyper.depth=2", "hyper.encoder_dim=32", "hyper.soup=2",
    "base.steps=300", "base.eval_every=100", "base.lr=3e-3",
    "meta.steps=20", "meta.eval_every=10",
    "oracle.steps=30", "oracle.lr=3e-2", "oracle.lam=0.01",
    "distill.epochs=60", "distill.patience=15",
    "em.rounds=3", "em.neighbor_threshold=10.0",
    "joint.steps=60", "joint.eval_every=20",
    "eval.gate_samples=64",
]


def test_identity_at_zero():
    cfg = load_config(None, SMALL + ["out_dir=/tmp/hc_id"])
    pipe = Pipeline(cfg)
    b = next(iter(pipe.loader(pipe.train_ds, 8)))
    z = torch.zeros(8, pipe.space.dim)
    assert torch.allclose(pipe.task.loss(pipe.cw, z, b, pipe.eval_bank),
                          pipe.task.loss(pipe.cw, None, b, pipe.eval_bank), atol=1e-6)
    # zero-initialized head => untrained hypernet reproduces the base exactly
    assert torch.count_nonzero(pipe.hyper(pipe.task.condition(b))) == 0


def test_full_pipeline(tmp_path=Path("/tmp/hc_smoke")):
    out = str(tmp_path)
    subprocess.run(["rm", "-rf", out])
    r = subprocess.run([sys.executable, str(ROOT / "train.py"), "--set", *SMALL, f"out_dir={out}"],
                       capture_output=True, text=True, cwd=ROOT)
    print(r.stdout[-4000:], r.stderr[-4000:])
    assert r.returncode == 0
    r = subprocess.run([sys.executable, str(ROOT / "eval.py"), "--ckpt", f"{out}/latest.pt", "--diag", "all",
                        "--ladder-lams", "1.0", "0.1", "0.01"],
                       capture_output=True, text=True, cwd=ROOT)
    print(r.stdout[-4000:], r.stderr[-4000:])
    assert r.returncode == 0
    assert (Path(out) / "eval_report.json").exists()


if __name__ == "__main__":
    test_identity_at_zero()
    test_full_pipeline()
    print("OK")
