"""CPU check for the turb2d history plumbing (seconds; reads frames from the packed cache via mmap).

  1. val/test items with history=2 have exactly the same (x, y) and length as history=0, and "hist" holds the
     preceding frames; the train split loses exactly its first 2 pairs.
  2. ModeNetHyper builds and runs with history (mode-net + trunk, and mode-net only), output shape = space.dim,
     zero at init; history=0 still gives the old parameter names (old checkpoints load).

    shifter --image=nersc/pytorch:26.01.01 python -u tests/check_history.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hypercond.config import load_config  # noqa: E402
from hypercond.update_space import UpdateSpace  # noqa: E402
from tasks.turb2d import Turb2DTask  # noqa: E402

L = 2
MODENET = ["task_kwargs.hyper_head=modenet", "update_space.additive=dense", "update_space.gain=false",
           "update_space.obs_basis=constant", "update_space.additive_scale=none", "hyper.soup=1"]


def main():
    torch.manual_seed(0)
    t0, tL = Turb2DTask(in_memory=False), Turb2DTask(in_memory=False, history=L)
    d0, dL = t0.datasets(), tL.datasets()
    for split in ("val", "test"):
        a, b = d0[split], dL[split]
        assert len(a) == len(b), (split, len(a), len(b))
        for i in (0, 1, len(a) // 2, len(a) - 1):
            ia, ib = a[i], b[i]
            assert torch.equal(ia["x"], ib["x"]) and torch.equal(ia["y"], ib["y"]), (split, i)
            base = b.dataset if hasattr(b, "dataset") else b
            j = b.indices[i] if hasattr(b, "indices") else i
            raw = base[j]
            for l in range(1, L + 1):  # hist[l-1] is omega_{t-l}: x of the item l back, or earlier frames
                want = (torch.from_numpy(base.frames[j + base.offset - l].copy()) - base.mean) / base.std
                assert torch.allclose(raw["hist"][l - 1], want), (split, i, l)
    tr0, trL = d0["train"], dL["train"]
    assert len(trL) == len(tr0) - L and torch.equal(trL[0]["x"], tr0[L]["x"]) and torch.equal(trL[0]["y"], tr0[L]["y"])
    assert torch.equal(trL[5]["hist"][0], trL[4]["x"][0]) and torch.equal(trL[5]["hist"][1], trL[3]["x"][0])
    print(f"[data] ok: val/test pairs identical to history=0; train {len(tr0)} -> {len(trL)}")

    cfg = load_config("configs/turb2d.yaml", MODENET)
    wheel = t0.build_wheel()
    space = UpdateSpace(wheel, cfg.update_space, t0.observed_range())
    batch = {k: torch.stack([dL["val"][i][k] for i in range(2)]) for k in ("x", "y", "hist")}
    keys0 = set(Turb2DTask(hyper_head="modenet").build_hypernet(cfg.hyper, space).state_dict())
    for trunk in (True, False):
        task = Turb2DTask(hyper_head="modenet", history=L, history_trunk=trunk)
        h = task.build_hypernet(cfg.hyper, space)
        with torch.no_grad():
            th = h(task.condition(batch))
        assert th.shape == (2, space.dim) and th.abs().max() == 0, th.shape
        # perturb the zero-init outputs and check the history inputs actually reach theta
        with torch.no_grad():
            for m in h.mode_nets.values():
                m.out.weight.normal_(0, 1e-2)
            th1 = h(task.condition(batch))
            b2 = dict(batch, hist=batch["hist"].flip(0))
            th2 = h(task.condition(b2))
        assert (th1 - th2).abs().max() > 0, "history has no effect"
        extra = set(h.state_dict()) - keys0
        print(f"[hyper] trunk={trunk}: theta {tuple(th.shape)}, params {sum(p.numel() for p in h.parameters()):,}, "
              f"new/changed keys: {len(extra)}")
    print("[hyper] ok; history=0 parameter names unchanged:",
          keys0 == set(Turb2DTask(hyper_head="modenet", history=0).build_hypernet(cfg.hyper, space).state_dict()))


if __name__ == "__main__":
    main()
