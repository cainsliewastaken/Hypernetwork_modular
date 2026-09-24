"""Run logging and the standing alarms (landscape §4, diagnostics §6)."""
from __future__ import annotations

import json
import time
from pathlib import Path

from .utils import to_float


class RunLogger:
    def __init__(self, out_dir):
        self.path = Path(out_dir) / "log.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, stage: str, quiet: bool = False, **rec):
        rec = {"stage": stage, "time": time.time(), **to_float(rec)}
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if not quiet:
            show = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in rec.items()
                    if k not in ("time", "stage") and not isinstance(v, (list, dict))}
            print(f"[{stage}] " + " ".join(f"{k}={v}" for k, v in show.items()), flush=True)


class Alarms:
    def __init__(self, em_cfg):
        self.cfg = em_cfg

    def check(self, rec: dict, prev: dict | None) -> list[str]:
        out = []
        nd = rec.get("neighbor_distance")
        if nd is not None and nd >= self.cfg.neighbor_threshold:
            out.append(f"neighbor_distance {nd:.3f} >= threshold {self.cfg.neighbor_threshold}: targets past the "
                       "smoothness threshold at this data spacing")
        if prev is not None:
            g, gp = rec.get("generalization_gap"), prev.get("generalization_gap")
            ref = abs(prev.get("test") or 1.0)
            if g is not None and gp is not None and (g - gp) > self.cfg.gap_tol * ref:
                out.append(f"generalization gap opening ({gp:.4g} -> {g:.4g})")
            inc = rec.get("increment")
            if inc is not None and inc < self.cfg.min_increment_rel:
                out.append(f"EM increment {inc:.4f} below {self.cfg.min_increment_rel} (converged)")
        return out
