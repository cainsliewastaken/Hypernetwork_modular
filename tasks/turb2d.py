"""2D turbulence (data_lowres) with a pretrained concat-FNO diffusion wheel.

Problem (same as HyperNetwork-Research/diffusion/train_three_stage.py, concat_condition=1)
    condition   x = omega_t, normalized with the global mean/std the wheels were trained with
    target      y = omega_{t+1} - omega_t (normalized): a one-step residual, dt = 1 frame
    wheel       FNO2DScoreField score model, called as wheel(tau, cat([noisy_y, x], C))
    loss        VP-SDE denoising score matching, ((std * score + eps)^2).mean() per sample
    s           the diffusion time tau in [tau_min, 1]; the update is resolved in tau

Pretrained wheels (2-layer FNO, modes 64x64, bias_modes 64, FiLM time conditioning, readout x2,
trained on frames 10000..20000 until early stopping):
    w16  experiment_concat_d2_narrow/CONCAT_D2/model_best.pt      best val 0.2513 @ epoch 216.5
    w32  experiment_concat_d2_w32/CONCAT_D2_W32/model_best.pt     best val 0.2146 @ epoch 234.5
(the w16 val used the tail split with 10 batches, w32 the stride split with the full val set,
so those two numbers are not directly comparable; here both are scored on the same splits.)

Data: training_data/data_lowres/{10000..30700}.mat, key 'Omega' (256x256), one trajectory in time
order. The .mat files are packed once into a float32 .npy cache (``python -m tasks.turb2d --pack``);
``datasets()`` packs automatically if the cache is missing. Default splits are time-blocked:
    train 10000..20000  the frames the wheels were trained on
    val   20001..22000  unseen by both wheels (EM selection split)
    test  22001..30700  unseen, far extrapolation in time

History (``history=L`` > 0): each item also carries the L previous frames, "hist" = [omega_{t-1}, ..., omega_{t-L}],
and the hypernet condition becomes cat([x, hist], C) = (1+L) channels. The wheel still sees only omega_t.
Val/test items keep exactly the same (x, y) pairs and indices as history=0 (their history reaches back into the
preceding frames, e.g. val item 0 uses frame 20000). The train split starts at the first cached frame, so its first
L pairs are dropped (it has L fewer items).
"""
from __future__ import annotations

import argparse
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset, Subset

from hypercond.task import Task
from tasks.fno2d import FNO2DScoreField

REPO = Path(__file__).resolve().parents[1]
SCRATCH = os.environ.get("SCRATCH", "/pscratch/sd/c/cainslie")
OLD_CKPTS = Path(SCRATCH) / "model_chkpts_scratch"
CACHE_DIR = Path(SCRATCH) / "hypernet_diffusion_cache"

WHEELS = {
    "w16": dict(width=16, ckpt=str(OLD_CKPTS / "experiment_concat_d2_narrow/CONCAT_D2/model_best.pt")),
    "w32": dict(width=32, ckpt=str(OLD_CKPTS / "experiment_concat_d2_w32/CONCAT_D2_W32/model_best.pt")),
}
FNO_KW = dict(in_channels=2, out_channels=1, modes1=64, modes2=64, num_layers=2, use_layernorm=False,
              use_time_embedding=True, time_conditioning="film", time_embed_dim=128, time_scale=1.0,
              readout_hidden_mult=2, uses_condition_concat=True, bias_modes=64)
# normalization the wheels were trained with (frames 10000..20000); mean ~ -6.6e-9, std ~ 10.42
DEFAULT_STATS = str(CACHE_DIR / "omega_data_lowres_10000_20000_normstats.npz")
DEFAULT_SPLITS = {"train": [10000, 20000], "val": [20001, 22000], "test": [22001, 30700]}
N_GRID = 256


# ---------------------------------------------------------------------- data
def cache_path(start: int, stop: int, cache_dir=CACHE_DIR) -> Path:
    return Path(cache_dir) / f"omega_data_lowres_{start}_{stop}_f32.npy"


def pack_frames(data_dir, start: int, stop: int, out: Path, workers: int = 16):
    """Read {start..stop}.mat (inclusive) into one raw (unnormalized) float32 [N, 256, 256] .npy."""
    from scipy.io import loadmat

    data_dir, out = Path(data_dir), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    idx = list(range(start, stop + 1))
    tmp = out.with_suffix(".tmp.npy")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(len(idx), N_GRID, N_GRID))

    def one(j):
        arr[j] = np.asarray(loadmat(data_dir / f"{idx[j]}.mat")["Omega"], dtype=np.float32)

    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        for n, _ in enumerate(ex.map(one, range(len(idx))), 1):
            if n % 2000 == 0 or n == len(idx):
                print(f"[pack] {n}/{len(idx)} frames ({time.time() - t0:.0f}s)", flush=True)
    arr.flush()
    del arr
    os.replace(tmp, out)
    print(f"[pack] wrote {out}", flush=True)


class PairDataset(Dataset):
    """Items {"x": omega_t, "y": omega_{t+1} - omega_t}, both [1, H, W], normalized. Time-ordered.

    frames[0] is `offset` frames before item 0's x (first_frame is item 0's x). With history=L (needs offset >= L)
    items also carry "hist": [L, H, W] = omega_{t-1}, ..., omega_{t-L}, plus N(0, hist_noise^2) in normalized units
    if hist_noise > 0."""

    def __init__(self, frames: np.ndarray, first_frame: int, mean: float, std: float, history: int = 0,
                 offset: int = 0, hist_noise: float = 0.0):
        if offset < history:
            raise ValueError(f"need offset >= history, got {offset} < {history}")
        self.frames, self.first_frame = frames, int(first_frame)
        self.mean, self.std = float(mean), float(std)
        self.history, self.offset, self.hist_noise = int(history), int(offset), float(hist_noise)

    def __len__(self):
        return self.frames.shape[0] - 1 - self.offset

    def _frame(self, j):
        return (torch.from_numpy(np.asarray(self.frames[j], dtype=np.float32)) - self.mean) / self.std

    def __getitem__(self, i):
        j = i + self.offset
        a = torch.from_numpy(np.asarray(self.frames[j], dtype=np.float32))
        b = torch.from_numpy(np.asarray(self.frames[j + 1], dtype=np.float32))
        item = {"x": ((a - self.mean) / self.std)[None], "y": ((b - a) / self.std)[None]}
        if self.history:
            h = torch.stack([self._frame(j - l) for l in range(1, self.history + 1)])
            if self.hist_noise > 0:
                h = h + self.hist_noise * torch.randn_like(h)
            item["hist"] = h
        return item


# ---------------------------------------------------------------------- diffusion
class VPSDE:
    """Power schedule beta(t) = beta_min + (beta_max - beta_min) t^power (old CLI defaults)."""

    def __init__(self, beta_min=0.01, beta_max=55.0, power=5.0):
        self.beta_min, self.beta_max, self.power = beta_min, beta_max, power

    def beta(self, t):
        return self.beta_min + (self.beta_max - self.beta_min) * t ** self.power

    def marginal(self, t):
        """(alpha(t), std(t)) with x_t = alpha x_0 + std eps."""
        integral = self.beta_min * t + (self.beta_max - self.beta_min) * t ** (self.power + 1) / (self.power + 1)
        alpha = torch.exp(-0.5 * integral)
        return alpha, torch.sqrt(1.0 - alpha ** 2)


class LowModeEncoder(nn.Module):
    """Phase-preserving hypernet features: real/imag of the Fourier modes of x.

    modes = 0 -> all modes (the full rfft2, H x (W//2 + 1)); modes = m -> only |kx|, ky < m.
    history = L > 0 (x has 1+L channels: omega_t, omega_{t-1}, ...): also encodes the L successive increments
    omega_{t-l+1} - omega_{t-l}, each with its own projection (early fusion: the small frame-to-frame change is
    encoded directly, not recovered from separately compressed frames). history = 0 ignores extra channels."""

    def __init__(self, modes: int, out_dim: int, grid: int = N_GRID, history: int = 0):
        super().__init__()
        self.modes, self.history = int(modes), int(history)
        n_modes = grid * (grid // 2 + 1) if self.modes <= 0 else 2 * self.modes * self.modes
        in_dim = 2 * n_modes
        self.proj = nn.Linear(in_dim, out_dim) if out_dim > 0 else nn.Identity()
        self.dproj = nn.ModuleList(nn.Linear(in_dim, out_dim) if out_dim > 0 else nn.Identity()
                                   for _ in range(self.history))
        self.out_dim = (out_dim if out_dim > 0 else in_dim) * (1 + self.history)

    def _feats(self, x2d):
        f = torch.fft.rfft2(x2d, norm="ortho")
        if self.modes > 0:
            m = self.modes
            f = torch.cat([f[:, :m, :m], f[:, -m:, :m]], dim=1)
        return torch.view_as_real(f).flatten(1)

    def forward(self, x):
        x = x.float()
        out = [self.proj(self._feats(x[:, 0]))]
        for l, p in enumerate(self.dproj):
            out.append(p(self._feats(x[:, l] - x[:, l + 1])))
        return torch.cat(out, dim=1)


# ---------------------------------------------------------------------- task
class Turb2DTask(Task):
    condition_in_wheel = True  # concat wheel: x is already an input channel

    def __init__(self, wheel: str = "w16", wheel_ckpt: str | None = None, splits: dict | None = None,
                 data_dir: str | None = None, cache_dir: str | None = None, stats_file: str = DEFAULT_STATS,
                 norm_mean: float | None = None, norm_std: float | None = None, in_memory: bool = True,
                 tau_min: float = 1e-3, beta_min: float = 0.01, beta_max: float = 55.0, power: float = 5.0,
                 encoder_modes: int = 0, solver_steps: int = 1000, checkpoint_bank: bool = True,
                 pretrained: bool = True, train_tau_power: float = 1.0, hyper_head: str = "linear",
                 delta_scale: float = 0.02, val_protocol: str = "random", val_seed: int = 20240901,
                 val_points: int = 512, mode_hidden: int = 64, mode_layers: int = 2, mode_z_dim: int = 64,
                 mode_attn_layers: int = 0, mode_attn_heads: int = 8, history: int = 0, history_trunk: bool = True,
                 history_noise: float = 0.0):
        if wheel not in WHEELS:
            raise ValueError(f"wheel must be one of {list(WHEELS)}, got {wheel!r}")
        self.wheel_name, self.width = wheel, WHEELS[wheel]["width"]
        self.wheel_ckpt = wheel_ckpt or WHEELS[wheel]["ckpt"]
        self.splits = {k: [int(v[0]), int(v[1])] for k, v in (splits or DEFAULT_SPLITS).items()}
        if "train" not in self.splits:
            raise KeyError("splits must contain 'train'")
        self.data_dir = Path(data_dir) if data_dir else REPO / "training_data" / "data_lowres"
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.in_memory = bool(in_memory)
        self.tau_min = float(tau_min)
        self.sde = VPSDE(beta_min, beta_max, power)
        self.encoder_modes = int(encoder_modes)
        self.checkpoint_bank = bool(checkpoint_bank)
        self.pretrained = bool(pretrained)  # False: fresh init (same class/init as the old repo), for base training
        # train banks: tau = tau_min + (1 - tau_min) u^p; p = 2 concentrates shared-parameter training at low noise,
        # where the conditioning headroom lives (diffusion_specifics.md §2). Solve/eval banks stay stratified-uniform.
        self.train_tau_power = float(train_tau_power)
        self.hyper_head = str(hyper_head)    # linear (framework HyperNetwork) | modenet (old `full` head port)
        self.delta_scale = float(delta_scale)
        self.mode_hidden, self.mode_layers, self.mode_z_dim = int(mode_hidden), int(mode_layers), int(mode_z_dim)
        self.mode_attn_layers, self.mode_attn_heads = int(mode_attn_layers), int(mode_attn_heads)
        # "random" (default): the val split is `val_points` pairs sampled uniformly (without replacement) from
        # the whole val range, and every (point, draw) gets its own tau ~ U[tau_min, 1] and noise. All of it is
        # seeded (by val_seed and the point's index), so every check of every run scores the identical draw.
        # "bank": the first val pairs, stratified taus, noise shared across samples (experiment log A-E).
        self.val_protocol, self.val_seed, self.val_points = str(val_protocol), int(val_seed), int(val_points)
        # hypernet-only history: L previous frames. history_trunk=False gives them to the mode-nets only (ablation);
        # history_noise perturbs train-split history frames (stand-in for generated frames during rollout)
        self.history, self.history_trunk, self.history_noise = int(history), bool(history_trunk), float(history_noise)
        self.solver_steps = int(solver_steps)  # reverse-SDE steps per sample (old sampler: 1000 Euler-Maruyama)
        if norm_mean is None or norm_std is None:
            st = np.load(stats_file)
            norm_mean, norm_std = float(st["mean"]), float(st["std"])
        self.mean, self.std = float(norm_mean), float(norm_std)

    # --- wheel
    def build_wheel(self):
        model = FNO2DScoreField(width=self.width, **FNO_KW)
        if not self.pretrained:
            return model
        raw = Path(self.wheel_ckpt).read_bytes()
        if raw.startswith(b"{"):  # old saveModel format: one JSON hyperparameter line, then torch.save payload
            raw = raw[raw.index(b"\n") + 1:]
        try:
            sd = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        except Exception:  # a Pipeline checkpoint (holds cfg/state objects): trusted local file
            sd = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
        if "model" in sd:  # accept wrapped checkpoints too
            sd = sd["model"]
        elif "wheel" in sd:  # Pipeline checkpoint (e.g. a finished base stage: after_base.pt)
            sd = sd["wheel"]
        model.load_state_dict(sd, strict=True)
        return model

    # --- data
    def datasets(self):
        lo = min(a for a, _ in self.splits.values())
        hi = max(b for _, b in self.splits.values())
        path = cache_path(lo, hi, self.cache_dir)
        if not path.exists():
            print(f"[turb2d] packing frames {lo}..{hi} -> {path}", flush=True)
            pack_frames(self.data_dir, lo, hi, path)
        allf = np.load(path, mmap_mode="r")
        out = {}
        L = self.history
        for name, (a, b) in self.splits.items():
            first = max(a, lo + L)  # item 0's x; only a split at the cache start loses pairs (its first L)
            fr = allf[first - L - lo: b - lo + 1]
            if self.in_memory:
                fr = np.array(fr)  # real copy into RAM (ascontiguousarray would return the read-only memmap view)
            out[name] = PairDataset(fr, first, self.mean, self.std, history=L, offset=L,
                                    hist_noise=self.history_noise if name == "train" else 0.0)
        if self.val_protocol == "random" and "val" in out and self.val_points < len(out["val"]):
            g = torch.Generator().manual_seed(self.val_seed)
            pick = torch.randperm(len(out["val"]), generator=g)[: self.val_points].sort().values.tolist()
            out["val"] = Subset(out["val"], pick)  # fixed random points from the whole val range
        return out

    def condition(self, batch):
        if self.history:
            return torch.cat([batch["x"], batch["hist"]], dim=1)  # [B, 1+L, H, W]: omega_t, omega_{t-1}, ...
        return batch["x"]

    def neighbor_pairs(self, train_ds, indices):
        idx = sorted(int(i) for i in indices)
        return [[i, j] for i, j in zip(idx, idx[1:]) if j == i + 1]  # adjacent saves on the trajectory

    def build_encoder(self, hyper_cfg):
        return LowModeEncoder(self.encoder_modes, hyper_cfg.encoder_dim,
                              history=self.history if self.history_trunk else 0)

    def build_hypernet(self, hyper_cfg, space):
        if self.hyper_head == "linear":
            return None
        if self.hyper_head == "modenet":
            from tasks.modenet_hyper import ModeNetHyper
            return ModeNetHyper(self.build_encoder(hyper_cfg), space, hyper_cfg, delta_scale=self.delta_scale,
                                mode_z_dim=self.mode_z_dim, mode_hidden=self.mode_hidden, mode_layers=self.mode_layers,
                                mode_attn_layers=self.mode_attn_layers, mode_attn_heads=self.mode_attn_heads,
                                history=self.history)
        raise ValueError(f"unknown hyper_head {self.hyper_head!r}")

    def observed_range(self):
        return (0.0, 1.0)

    # --- banks and loss
    def make_bank(self, n, generator, purpose):
        if purpose == "eval" and self.val_protocol == "random":
            return {"n_draws": torch.tensor(n), "draw_seed": torch.tensor(self.val_seed)}
        u = torch.rand(n, generator=generator) ** self.train_tau_power if purpose == "train" \
            else (torch.arange(n) + torch.rand(n, generator=generator)) / n  # tau-stratified
        tau = self.tau_min + (1.0 - self.tau_min) * u
        # fixed (CRN) noise for solve/eval banks; fresh per-sample noise for training
        eps = None if purpose == "train" else torch.randn(n, 1, N_GRID, N_GRID, generator=generator)
        return {"tau": tau, "eps": eps}

    def _index_draws(self, idx, n, seed, device):
        """Per-point random draws that depend only on the point's dataset index: tau [B, n] ~ U[tau_min, 1]
        and noise [B, n, 1, H, W]."""
        idx = idx.reshape(-1).tolist()
        taus = torch.empty(len(idx), n, device=device)
        eps = torch.empty(len(idx), n, 1, N_GRID, N_GRID, device=device)
        g = torch.Generator(device=device)
        for j, i in enumerate(idx):
            g.manual_seed(int(seed) * 1_000_003 + int(i))
            taus[j] = self.tau_min + (1.0 - self.tau_min) * torch.rand(n, generator=g, device=device)
            eps[j] = torch.randn(n, 1, N_GRID, N_GRID, generator=g, device=device)
        return taus, eps

    def loss(self, model, theta, batch, bank):
        x, y = batch["x"], batch["y"]
        B = x.shape[0]
        draws = None
        if bank.get("draw_seed") is not None:  # random-point validation: per-point tau and noise
            if "_idx" not in batch:
                raise KeyError("validation draws need batch['_idx'] (use Pipeline.loader)")
            draws = self._index_draws(batch["_idx"], int(bank["n_draws"]), int(bank["draw_seed"]), y.device)
            n = draws[0].shape[1]
        else:
            n = bank["tau"].shape[0]
        total = 0.0
        for k in range(n):
            if draws is not None:
                tv = draws[0][:, k]
                alpha, sd = self.sde.marginal(tv.view(B, 1, 1, 1))
                eps = draws[1][:, k]
            else:
                tau = bank["tau"][k]
                alpha, sd = self.sde.marginal(tau)
                eps = torch.randn_like(y) if bank["eps"] is None else bank["eps"][k].expand_as(y)
                tv = tau.expand(B)
            inp = torch.cat([alpha * y + sd * eps, x], dim=1)
            if self.checkpoint_bank and torch.is_grad_enabled():
                # recompute each draw's graph (incl. the per-sample weight deltas) in backward instead of
                # holding all of them: memory ~ one draw instead of the whole bank
                score = checkpoint(lambda t_, i_: model(theta, t_, t_, i_), tv, inp, use_reentrant=False)
            else:
                score = model(theta, tv, tv, inp)
            total = total + (sd * score + eps).pow(2).mean(dim=(1, 2, 3))
        return total / n

    TAU_BANDS = ((0.0, 0.5), (0.5, 0.8), (0.8, 1.0))  # same bins as the old repo's val_bins

    def _band_bank(self, device, n_per_band=8, seed=4242):
        """Fixed (CRN) per-band banks: n_per_band stratified taus per band, with fixed noise."""
        if getattr(self, "_bands", None) is None or self._bands[0]["tau"].device != device:
            g = torch.Generator().manual_seed(seed)
            banks = []
            for lo, hi in self.TAU_BANDS:
                lo = max(lo, self.tau_min)
                u = (torch.arange(n_per_band) + torch.rand(n_per_band, generator=g)) / n_per_band
                banks.append({"tau": (lo + (hi - lo) * u).to(device),
                              "eps": torch.randn(n_per_band, 1, N_GRID, N_GRID, generator=g).to(device)})
            self._bands = banks
        return self._bands

    def evaluate(self, model, theta, batch):
        """Score loss per tau band ([B] each), for the banded breakdown (diffusion_specifics.md §3)."""
        out = {}
        for (lo, hi), bank in zip(self.TAU_BANDS, self._band_bank(batch["x"].device)):
            out[f"tau_{lo:g}_{hi:g}"] = self.loss(model, theta, batch, bank)
        return out

    def wheel_example_inputs(self, batch):
        x = batch["x"]
        tv = torch.full((x.shape[0],), 0.5, device=x.device)
        return tv, (tv, torch.cat([batch["y"], x], dim=1))


def main():
    ap = argparse.ArgumentParser(description="Pack data_lowres .mat frames into the float32 .npy cache.")
    ap.add_argument("--pack", action="store_true")
    ap.add_argument("--start", type=int, default=min(a for a, _ in DEFAULT_SPLITS.values()))
    ap.add_argument("--stop", type=int, default=max(b for _, b in DEFAULT_SPLITS.values()))
    ap.add_argument("--data-dir", default=str(REPO / "training_data" / "data_lowres"))
    ap.add_argument("--cache-dir", default=str(CACHE_DIR))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    if not args.pack:
        ap.error("nothing to do (use --pack)")
    pack_frames(args.data_dir, args.start, args.stop, cache_path(args.start, args.stop, args.cache_dir), args.workers)


if __name__ == "__main__":
    main()
