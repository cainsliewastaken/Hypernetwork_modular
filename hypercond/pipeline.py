"""Pipeline: owns the task, wheel, update space, hypernetwork, banks, logging and checkpoints."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

from .conditioned import ConditionedWheel
from .config import Config, config_from_dict
from .dashboard import Alarms, RunLogger
from .hypernet import FlattenEncoder, HyperEnsemble, HyperNetwork
from .targets import IndexedDataset, TargetStore
from .update_space import UpdateSpace
from .utils import load_object, make_generator, resolve_device, seed_all, to_device, to_float

STAGES = ("base", "meta", "gate", "em", "joint")


class Pipeline:
    def __init__(self, cfg: Config, task=None):
        self.cfg = cfg
        seed_all(cfg.seed)
        self.device = resolve_device(cfg.device)
        self.out = Path(cfg.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.logger = RunLogger(self.out)

        self.task = task if task is not None else load_object(cfg.task)(**cfg.task_kwargs)
        self.datasets: dict[str, Dataset] = self.task.datasets()
        if "train" not in self.datasets:
            raise KeyError("Task.datasets() must contain 'train'")
        self.train_ds = self.datasets["train"]
        ss = cfg.eval.select_split
        others = [k for k in self.datasets if k != "train"]
        self.select_split = ss if ss in self.datasets else (others[0] if others else None)

        self.wheel = self.task.build_wheel().to(self.device)
        self.space = UpdateSpace(self.wheel, cfg.update_space, self.task.observed_range(),
                                 self.task.update_param_names(self.wheel)).to(self.device)
        self.cw = ConditionedWheel(self.wheel, self.space, chunk_size=cfg.vmap_chunk)

        sample = to_device(default_collate([self.train_ds[0]]), self.device)
        self.cond_dim = int(self.task.condition(sample).flatten(1).shape[1])
        self.hyper = HyperEnsemble([self.build_member() for _ in range(max(1, cfg.hyper.soup))]).to(self.device)

        self.solve_bank = to_device(self.task.make_bank(cfg.oracle.bank_size, make_generator(cfg.oracle.bank_seed), "solve"), self.device)
        self.eval_bank = to_device(self.task.make_bank(cfg.eval.bank_size, make_generator(cfg.eval.bank_seed), "eval"), self.device)
        self.alarms = Alarms(cfg.em)
        self.state = {"completed": [], "em_round": -1, "em_last": None, "targets": None, "reports": {}}
        self.logger.log("setup", quiet=True, update_space=self.space.describe(),
                        wheel_params=sum(p.numel() for p in self.wheel.parameters()),
                        hyper_params=sum(p.numel() for p in self.hyper.parameters()))
        print(f"[setup] wheel params={sum(p.numel() for p in self.wheel.parameters())} "
              f"update dim={self.space.dim} (D={self.space.D} x {self.space.n_basis} basis fns, {self.space.obs.spec}) "
              f"hyper params={sum(p.numel() for p in self.hyper.parameters())} soup={len(self.hyper.members)} "
              f"select_split={self.select_split}", flush=True)

    # ------------------------------------------------------------------ building blocks
    def build_member(self, **overrides) -> HyperNetwork:
        hc = dataclasses.replace(self.cfg.hyper, **overrides)
        enc = self.task.build_encoder(hc)
        if enc is None:
            enc = FlattenEncoder(self.cond_dim, hc.encoder_dim)
        return HyperNetwork(enc, enc.out_dim, self.space.dim, hc).to(self.device)

    def loader(self, ds: Dataset, batch_size: int, indices=None, shuffle=False, drop_last=False) -> DataLoader:
        if not isinstance(ds, IndexedDataset):
            ds = IndexedDataset(ds, indices)
        elif indices is not None:
            raise ValueError("indices given for an already-indexed dataset")
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last and len(ds) > batch_size)

    def train_bank(self, n: int, step: int):
        return to_device(self.task.make_bank(n, make_generator(self.cfg.seed * 1_000_003 + step), "train"), self.device)

    def target_indices(self) -> list[int]:
        idx = self.task.target_indices(self.train_ds)
        if idx is not None:
            return sorted(int(i) for i in idx)
        n = max(1, int(round(self.cfg.em.target_fraction * len(self.train_ds))))
        return list(range(n))  # contiguous prefix: a dense patch, not thin coverage

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def evaluate(self, split: str | None = None, indices=None, hyper=None, max_batches: int | None = None,
                 shuffle_control: bool = False, extra_metrics: bool = False, scale: float = 1.0) -> dict:
        """Mean base / conditioned (guards installed) losses on the fixed eval bank.

        split=None with indices -> train samples at those indices."""
        hyper = hyper or self.hyper
        ds = self.datasets[split] if split else self.train_ds
        mb = self.cfg.eval.max_batches if max_batches is None else max_batches
        wt, ht = self.wheel.training, hyper.training
        self.wheel.eval(); hyper.eval()
        sums, n = {}, 0

        def add(k, v):
            sums[k] = sums.get(k, 0.0) + float(v.sum())

        for bi, b in enumerate(self.loader(ds, self.cfg.eval.batch_size, indices=indices, shuffle=False)):
            if mb and bi >= mb:
                break
            b = to_device(b, self.device)
            th = hyper.predict(self.task.condition(b)) * scale
            base = self.task.loss(self.cw, None, b, self.eval_bank)
            add("base", base)
            add("cond", self.task.loss(self.cw, th, b, self.eval_bank))
            if shuffle_control and th.shape[0] > 1:
                add("shuffled", self.task.loss(self.cw, th.roll(1, 0), b, self.eval_bank))
            if extra_metrics:
                for k, v in self.task.evaluate(self.cw, None, b).items():
                    add(f"base_{k}", v)
                for k, v in self.task.evaluate(self.cw, th, b).items():
                    add(f"cond_{k}", v)
            n += base.shape[0]
        self.wheel.train(wt); hyper.train(ht)
        out = {k: v / max(n, 1) for k, v in sums.items()}
        out["n"] = n
        if out.get("base"):
            out["margin_rel"] = 1 - out["cond"] / out["base"]
        return out

    def evaluate_all(self, shuffle_control=True, extra_metrics=True) -> dict:
        return {name: self.evaluate(split=name if name != "train" else None,
                                    indices=None, shuffle_control=shuffle_control, extra_metrics=extra_metrics)
                for name in self.datasets}

    # ------------------------------------------------------------------ stages
    def run(self, stages=None):
        from .diagnostics import headroom_gate
        from .em import run_em
        from .stages import train_base, train_joint, train_meta

        for st in stages or self.cfg.stages:
            if st not in STAGES:
                raise ValueError(f"Unknown stage {st!r}; choose from {STAGES}")
            if st in self.state["completed"]:
                print(f"[{st}] already completed, skipping", flush=True)
                continue
            if st in ("base", "meta") and self.state["targets"] is not None:
                raise RuntimeError(f"stage {st} after targets exist would invalidate them (bases rebuild)")
            print(f"=== stage: {st} ===", flush=True)
            if st == "base":
                train_base(self)
            elif st == "meta":
                train_meta(self)
            elif st == "gate":
                rep = headroom_gate(self)
                self.state["reports"]["gate"] = rep
                self.logger.log("gate", **rep)
            elif st == "em":
                run_em(self, start_round=self.state["em_round"] + 1 if self.state["em_round"] >= 0 else 0,
                       rounds=self.cfg.em.rounds - (self.state["em_round"] + 1))
            elif st == "joint":
                train_joint(self)
                if self.cfg.joint.steps > 0 and self.cfg.joint.refresh_em and self.state["targets"] is not None:
                    run_em(self, rounds=1, start_round=self.state["em_round"] + 1, tag="em_refresh")
            self.state["completed"].append(st)
            self.save("latest")
            self.save(f"after_{st}")

    # ------------------------------------------------------------------ persistence
    def save_targets(self, store: TargetStore):
        path = self.out / f"targets_round{store.round}.pt"
        store.save(path)
        self.state["targets"] = str(path)

    def load_targets(self) -> TargetStore | None:
        p = self.state.get("targets")
        return TargetStore.load(p) if p and Path(p).exists() else None

    def save(self, tag: str):
        torch.save({"cfg": self.cfg.to_dict(), "wheel": self.wheel.state_dict(), "space": self.space.state_dict(),
                    "hyper": self.hyper.state_dict(), "state": self.state}, self.out / f"{tag}.pt")

    @classmethod
    def from_checkpoint(cls, path, overrides: dict | None = None, task=None) -> "Pipeline":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        d = ck["cfg"]
        n_members = len({k.split(".")[1] for k in ck["hyper"] if k.startswith("members.")})
        if overrides:
            from .config import _deep_update
            _deep_update(d, overrides)
        d["hyper"]["soup"] = n_members  # structure comes from the checkpoint
        pipe = cls(config_from_dict(d), task=task)
        pipe.wheel.load_state_dict(ck["wheel"])
        pipe.space.load_state_dict(ck["space"])
        pipe.hyper.load_state_dict(ck["hyper"])
        pipe.state = ck["state"]
        return pipe

    def write_report(self, name: str, report: dict):
        (self.out / name).write_text(json.dumps(to_float(report), indent=2))
