"""Typed configuration. Every knob maps to a section of the reference documents."""
from __future__ import annotations

import json
import typing
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class UpdateSpaceConfig:
    """Which wheel parameters the hypernetwork may modify, and how (landscape doc §6)."""
    include: list = field(default_factory=list)   # regexes on parameter names; empty = all eligible weights
    exclude: list = field(default_factory=list)
    gain: bool = True                  # multiplicative per-output-channel gain (1 + a) * W
    additive: str = "subspace"         # none | subspace | dense : additive operator-space term
    rank: int = 8                      # subspace rank r (coefficients r*r per weight; gauge-free)
    basis: str = "svd"                 # svd (top singular vectors of base W) | random (fixed orthonormal)
    additive_scale: str = "weight_rms" # none | weight_rms : coefficients in units relative to the weight
    obs_basis: str = "affine"          # constant | affine | poly:K | hat:K  (resolution in the observed variable)
    allow_activation_additive: bool = False  # biases / norm shifts / embeddings: the memorization channel
    spectral: str = "same"             # complex 4-D spectral weights (in, out, m1, m2): same (use `additive`) |
                                       # coarse: dense coarse c x c grid per channel pair, bilinearly upsampled to (m1, m2)
    spectral_coarse: int = 16          # c for spectral=coarse
    spectral_align_corners: bool = True


@dataclass
class HyperConfig:
    encoder_dim: int = 256     # default encoder bandwidth (0 = raw flattened condition)
    out_scale: float = 1.0     # fixed multiplier on the head output (small values = smaller effective head step)
    width: int = 512
    depth: int = 4             # residual trunk blocks (strongest conventional capacity axis)
    head_rank: int = 0         # 0 = full head; >0 = low-rank head (beware the collapse rank)
    feature_clip: float = 6.0  # clamp on normalized encoder features (deployment guard)
    std_floor: float = 1e-2    # feature std floor, relative to median feature std
    norm_momentum: float = 0.01
    trust_quantile: float = 0.99   # trust-region cap = factor * quantile of target norms
    trust_factor: float = 1.5
    soup: int = 3              # prediction-averaged M-step seeds


@dataclass
class BaseTrainConfig:
    steps: int = 0
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 64
    bank_size: int = 4
    grad_clip: float = 1.0
    eval_every: int = 500
    warmup_steps: int = 0         # linear warmup from 0
    schedule: str = "cosine"      # after warmup: cosine (to 0 at `steps`) | constant


@dataclass
class MetaConfig:
    """First-order bi-level base refinement: train the base to be the best anchor (§7b)."""
    steps: int = 0
    lr: float = 1e-4
    batch_size: int = 32
    bank_size: int = 4
    inner_steps: int = 10
    inner_lr: float = 1e-2
    lam: float = 1.0
    eval_every: int = 200


@dataclass
class OracleConfig:
    """Anchored proximal solves (§3): short, identical budgets; consistency over depth."""
    lam: float = 1.0
    steps: int = 50
    lr: float = 1e-2
    batch_size: int = 64
    bank_size: int = 12
    bank_seed: int = 1234


@dataclass
class DistillConfig:
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 64
    grad_clip: float = 1.0
    target_val_frac: float = 0.1   # time-blocked tail of the target set, for early stopping
    patience: int = 15
    task_loss_weight: float = 0.0  # semi-supervised task loss on untargeted samples
    warm_start: bool = True        # warm-start M-steps from the previous round


@dataclass
class EMConfig:
    rounds: int = 3
    lam_schedule: list = field(default_factory=list)  # per-round lambda; empty = oracle.lam every round
    target_fraction: float = 1.0      # contiguous prefix of train set gets targets (dense patch, not thin)
    min_increment_rel: float = 0.01   # stop when relative depth increment falls below this
    neighbor_threshold: float = 1.0   # neighbor-target relative distance alarm (MEASURE YOUR OWN)
    gap_tol: float = 0.05             # relative opening of the generalization gap that triggers an alarm
    knn_max: int = 5000               # cap on samples used for kNN neighbor pairs


@dataclass
class DirectConfig:
    """Direct hypernet training on the task loss (wheel frozen, fresh train bank every step).
    Amortizes over many stochastic draws instead of solving per-sample targets on a fixed bank, for
    tasks whose loss is too noisy for oracle targets (e.g. diffusion score matching)."""
    steps: int = 0
    lr: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 6
    bank_size: int = 8
    grad_clip: float = 0.25
    eval_every: int = 250
    warmup_steps: int = 0         # linear warmup from 0
    schedule: str = "constant"    # after warmup: constant | cosine (to end_lr at `steps`)
    end_lr: float = 0.0
    mode_weight_decay: float = -1.0  # >=0: separate AdamW weight decay for params named "*mode_nets*" (old best: 0.1)
    train_eval_n: int = 0         # >0: also score this many evenly spaced TRAIN pairs on the eval bank at every
                                  # check (train vs val margin = generalization gap; capability_diagnostics.md §2)


@dataclass
class JointConfig:
    """Two-timescale joint finisher (§7f)."""
    steps: int = 0
    hyper_lr: float = 1e-4
    wheel_lr_ratio: float = 5.0   # wheel lr = hyper_lr / ratio  (3-10 recommended)
    anchor_weight: float = 1.0    # L2 anchor of H to its own distilled predictions
    batch_size: int = 64
    bank_size: int = 4
    grad_clip: float = 1.0
    eval_every: int = 100
    refresh_em: bool = True       # one warm-started EM round after the joint phase
    anchor_reduce: str = "sum"    # sum | mean | relative (drift^2 / anchor^2, scale-free; for very large update spaces)


@dataclass
class EvalConfig:
    bank_size: int = 16
    bank_seed: int = 987
    batch_size: int = 128
    max_batches: int = 0          # 0 = full split
    select_split: str = "val"
    guidance_scales: list = field(default_factory=lambda: [0.0, 0.5, 1.0, 1.25, 1.5, 2.0])
    gate_samples: int = 256
    gate_steps_mult: float = 2.0


@dataclass
class Config:
    task: str = ""                # "pkg.module:TaskClass" or "path/to/file.py:TaskClass"
    task_kwargs: dict = field(default_factory=dict)
    out_dir: str = "model_checkpoints/default"
    seed: int = 0
    device: str = "auto"
    vmap_chunk: int = 0           # 0 = no chunking of the per-sample vmap
    stages: list = field(default_factory=lambda: ["base", "meta", "gate", "em", "joint"])
    update_space: UpdateSpaceConfig = field(default_factory=UpdateSpaceConfig)
    hyper: HyperConfig = field(default_factory=HyperConfig)
    base: BaseTrainConfig = field(default_factory=BaseTrainConfig)
    meta: MetaConfig = field(default_factory=MetaConfig)
    oracle: OracleConfig = field(default_factory=OracleConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    em: EMConfig = field(default_factory=EMConfig)
    direct: DirectConfig = field(default_factory=DirectConfig)
    joint: JointConfig = field(default_factory=JointConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def _build(cls, d: dict):
    hints = typing.get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise KeyError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for f in fields(cls):
        if f.name in d:
            v, t = d[f.name], hints[f.name]
            if is_dataclass(t) and isinstance(v, dict):
                v = _build(t, v)
            elif t is float and isinstance(v, (str, int)) and not isinstance(v, bool):
                v = float(v)  # YAML 1.1 reads "3e-3" as a string
            elif t is int and isinstance(v, str):
                v = int(float(v))
            elif t is list and isinstance(v, list):
                v = [float(x) if isinstance(x, str) and _is_number(x) else x for x in v]
            kwargs[f.name] = v
    return cls(**kwargs)


def _is_number(x: str) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        return False


def _parse_value(s: str):
    if yaml is not None:
        return yaml.safe_load(s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


def apply_overrides(d: dict, overrides: list[str]) -> dict:
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        node = d
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(val)
    return d


def config_from_dict(d: dict) -> Config:
    return _build(Config, d)


def load_config(path: str | None = None, overrides: list[str] | None = None) -> Config:
    d = Config().to_dict()
    if path:
        text = Path(path).read_text()
        loaded = yaml.safe_load(text) if (yaml is not None and not path.endswith(".json")) else json.loads(text)
        _deep_update(d, loaded or {})
    apply_overrides(d, overrides or [])
    return config_from_dict(d)


def _deep_update(dst: dict, src: dict):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict) and k != "task_kwargs":
            _deep_update(dst[k], v)
        else:
            dst[k] = v
