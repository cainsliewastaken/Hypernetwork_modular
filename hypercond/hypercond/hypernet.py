"""Hypernetwork: encoder -> guarded feature normalization -> residual trunk -> zero-init head."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class FlattenEncoder(nn.Module):
    """Default encoder: flatten (phase-preserving; no pooling) then an optional linear projection.
    ``encoder_dim`` is the bandwidth axis probed by the capacity probe."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim) if out_dim > 0 else nn.Identity()
        self.out_dim = out_dim if out_dim > 0 else in_dim

    def forward(self, x):
        return self.proj(x.flatten(1).float())


class FeatureNormalizer(nn.Module):
    """Standardize with train-set statistics (EMA in train mode, frozen in eval) with a variance
    floor, then clamp. Deployment guard: near-zero-variance feature directions otherwise turn small
    generation artifacts into enormous inputs (NaNs within steps in autoregressive rollout)."""

    def __init__(self, dim: int, clip: float, std_floor: float, momentum: float):
        super().__init__()
        self.clip, self.std_floor, self.momentum = clip, std_floor, momentum
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))

    def forward(self, f):
        if self.training and f.shape[0] > 1:
            with torch.no_grad():
                m, v = f.mean(0), f.var(0, unbiased=False)
                if not bool(self.initialized):
                    self.mean.copy_(m); self.var.copy_(v); self.initialized.fill_(True)
                else:
                    self.mean.lerp_(m, self.momentum); self.var.lerp_(v, self.momentum)
        std = self.var.clamp_min(0).sqrt()
        floor = self.std_floor * std.median().clamp_min(1e-8)
        z = (f - self.mean) / torch.maximum(std, floor)
        return z.clamp(-self.clip, self.clip) if self.clip > 0 else z


class ResBlock(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, w), nn.GELU(), nn.Linear(w, w))

    def forward(self, h):
        return h + self.net(h)


class HyperNetwork(nn.Module):
    def __init__(self, encoder: nn.Module, feat_dim: int, out_dim: int, cfg):
        super().__init__()
        self.encoder = encoder
        self.norm = FeatureNormalizer(feat_dim, cfg.feature_clip, cfg.std_floor, cfg.norm_momentum)
        self.inp = nn.Linear(feat_dim, cfg.width)
        self.trunk = nn.Sequential(*[ResBlock(cfg.width) for _ in range(cfg.depth)], nn.LayerNorm(cfg.width))
        if cfg.head_rank and cfg.head_rank > 0:
            self.head = nn.Sequential(nn.Linear(cfg.width, cfg.head_rank), nn.Linear(cfg.head_rank, out_dim))
            last = self.head[-1]
        else:
            self.head = nn.Linear(cfg.width, out_dim)
            last = self.head
        nn.init.zeros_(last.weight)   # identity at zero: the system starts exactly at the base
        nn.init.zeros_(last.bias)

    def forward(self, x):
        return self.head(self.trunk(self.inp(self.norm(self.encoder(x)))))


class HyperEnsemble(nn.Module):
    """Prediction-averaged "soup" of M-step seeds, plus the deployment trust-region cap.

    forward(x)  -> raw mean prediction (training, anchors).
    predict(x)  -> guarded prediction (evaluation, deployment): norm capped, optionally gated.
    """

    def __init__(self, members: list[nn.Module]):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.register_buffer("trust_cap", torch.tensor(math.inf))

    def forward(self, x):
        if len(self.members) == 1:
            return self.members[0](x)
        return torch.stack([m(x) for m in self.members]).mean(0)

    def predict(self, x, gate=1.0):
        th = self(x)
        if torch.isfinite(self.trust_cap):
            n = th.norm(dim=1, keepdim=True).clamp_min(1e-12)
            th = th * (self.trust_cap / n).clamp(max=1.0)
        if torch.is_tensor(gate):
            gate = gate.reshape(-1, 1)
        return th * gate

    @torch.no_grad()
    def fit_trust_cap(self, targets: torch.Tensor, quantile: float, factor: float):
        n = targets.float().norm(dim=1)
        if n.numel():
            self.trust_cap.fill_(float(factor * torch.quantile(n, quantile)))
