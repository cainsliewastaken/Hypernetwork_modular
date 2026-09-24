"""Update-space design (landscape doc §6).

The update is a LINEAR function of a flat coefficient vector theta, which makes it gauge-free
(no rank-factorization gauge: subspace updates use FIXED bases with a free r x r core), identity at
zero (theta = 0 reproduces the base exactly), and cheap to combine.

theta has shape [B, n_basis * D]: n_basis blocks of D coefficients, one per basis function of the
inference-observed variable s. The coefficients actually applied are
    c(s) = sum_k phi_k(u(s)) * theta_k,      u = (s - lo) / (hi - lo)
so with obs_basis="affine" this is exactly ΔW(s) = g0 + u * g1 (the "A·τ + B" form). Oracle solves,
distillation targets and the hypernet head all live in this concatenated space.

Per selected weight W (reshaped to out x in):
    ΔW = a ⊙ W                      (gain, per output channel; multiplicative)
       + scale * U_r C V_rᵀ          (additive "subspace": fixed bases, free r x r core C)
       | scale * A                   (additive "dense": free out x in)
Biases, norm shifts and embeddings (activation-additive channels) are excluded by default: they
are the memorization channel.
"""
from __future__ import annotations

import re
import warnings

import torch
import torch.nn as nn

_ACTIVATION_ADDITIVE_MODULES = (
    nn.Embedding, nn.EmbeddingBag, nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d,
    nn.BatchNorm3d, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
)


class ObsBasis:
    """Basis functions phi(u) of the observed variable."""

    def __init__(self, spec: str, lo: float = 0.0, hi: float = 1.0):
        self.spec, self.lo, self.hi = spec, float(lo), float(hi)
        kind, _, arg = spec.partition(":")
        if kind == "constant":
            n = 1
        elif kind == "affine":
            n = 2
        elif kind == "poly":
            n = int(arg) + 1
        elif kind == "hat":
            n = int(arg)
            if n < 2:
                raise ValueError("hat:K needs K >= 2")
        else:
            raise ValueError(f"Unknown obs_basis {spec!r}")
        self.kind, self.n = kind, n

    def __call__(self, s: torch.Tensor) -> torch.Tensor:
        u = (s - self.lo) / max(self.hi - self.lo, 1e-12)
        if self.kind == "constant":
            return torch.ones_like(u)[:, None]
        if self.kind == "affine":
            return torch.stack([torch.ones_like(u), u], dim=1)
        if self.kind == "poly":
            return torch.stack([u ** i for i in range(self.n)], dim=1)
        knots = torch.linspace(0, 1, self.n, device=u.device, dtype=u.dtype)
        u = u.clamp(0, 1)
        return torch.relu(1 - (u[:, None] - knots[None]).abs() * (self.n - 1))


class UpdateSpace(nn.Module):
    def __init__(self, wheel: nn.Module, cfg, obs_range=(0.0, 1.0), param_names: list[str] | None = None):
        super().__init__()
        self.cfg = cfg
        self.obs = ObsBasis(cfg.obs_basis, *obs_range)
        selected = self._select(wheel, cfg, param_names)
        if not selected:
            raise ValueError("Update space is empty: no eligible wheel parameters selected.")
        self.entries: list[dict] = []
        offset = 0
        for name, p in selected:
            if p.dim() >= 2:
                out, inn, gain_dim = p.shape[0], p.numel() // p.shape[0], (p.shape[0] if cfg.gain else 0)
                mode = cfg.additive
            else:  # activation-additive (only when explicitly allowed)
                out, inn, gain_dim, mode = p.numel(), 1, 0, "dense"
            if mode == "subspace":
                r = max(1, min(cfg.rank, out, inn))
                add_dim = r * r
            elif mode == "dense":
                r, add_dim = 0, out * inn
            elif mode == "none":
                r, add_dim = 0, 0
            else:
                raise ValueError(f"Unknown additive mode {mode!r}")
            if gain_dim + add_dim == 0:
                continue
            self.entries.append(dict(name=name, shape=tuple(p.shape), out=out, inn=inn, gain_dim=gain_dim,
                                     mode=mode, r=r, add_dim=add_dim, offset=offset))
            offset += gain_dim + add_dim
        if cfg.gain and cfg.additive == "none":
            warnings.warn("Purely multiplicative update spaces are expressivity-capped (measured three ways); "
                          "consider additive='subspace' or 'dense'.")
        self.D = offset
        self.n_basis = self.obs.n
        self.dim = self.D * self.n_basis
        self.rebuild(wheel)

    # ------------------------------------------------------------------ selection
    @staticmethod
    def _select(wheel, cfg, param_names):
        owner = {}
        for mname, mod in wheel.named_modules():
            for pname, _ in mod.named_parameters(recurse=False):
                owner[f"{mname}.{pname}" if mname else pname] = mod
        params = dict(wheel.named_parameters())

        def is_act_additive(name, p):
            return p.dim() < 2 or isinstance(owner.get(name), _ACTIVATION_ADDITIVE_MODULES)

        if param_names is not None:
            out = []
            for n in param_names:
                p = params[n]
                if is_act_additive(n, p) and not cfg.allow_activation_additive:
                    raise ValueError(f"{n} is an activation-additive channel (bias/norm/embedding): the memorization "
                                     "channel. Set update_space.allow_activation_additive=true to override.")
                out.append((n, p))
            return out

        out = []
        for n, p in params.items():
            if cfg.include and not any(re.search(pat, n) for pat in cfg.include):
                continue
            if any(re.search(pat, n) for pat in cfg.exclude):
                continue
            if is_act_additive(n, p) and not cfg.allow_activation_additive:
                continue
            out.append((n, p))
        if cfg.allow_activation_additive:
            warnings.warn("Activation-additive channels are enabled: expect memorization of rough target components.")
        return out

    # ------------------------------------------------------------------ bases
    @torch.no_grad()
    def rebuild(self, wheel: nn.Module):
        """(Re)compute fixed bases/scales from the current base weights.

        Call only BEFORE any targets exist (after base / meta training): targets and hypernet
        outputs are coordinates in these bases, so rebuilding afterwards invalidates them.
        """
        params = dict(wheel.named_parameters())
        for i, e in enumerate(self.entries):
            W = params[e["name"]].detach().float().reshape(e["out"], e["inn"])
            if self.cfg.additive_scale == "weight_rms":
                scale = W.pow(2).mean().sqrt().clamp_min(1e-6)
            else:
                scale = torch.ones((), device=W.device)
            self.register_buffer(f"scale_{i}", scale.reshape(()))
            if e["mode"] == "subspace":
                r = e["r"]
                if self.cfg.basis == "svd":
                    U, _, Vh = torch.linalg.svd(W, full_matrices=False)
                    Ur, Vr = U[:, :r], Vh[:r].T
                else:
                    g = torch.Generator().manual_seed(1000 + i)
                    Ur = torch.linalg.qr(torch.randn(e["out"], r, generator=g))[0].to(W.device)
                    Vr = torch.linalg.qr(torch.randn(e["inn"], r, generator=g))[0].to(W.device)
                self.register_buffer(f"U_{i}", Ur.contiguous())
                self.register_buffer(f"V_{i}", Vr.contiguous())

    # ------------------------------------------------------------------ application
    def coefficients(self, theta: torch.Tensor, s) -> torch.Tensor:
        """Resolve theta [B, n*D] at observed variable s -> c [B, D]."""
        B = theta.shape[0]
        if not torch.is_tensor(s):
            s = torch.tensor(float(s), device=theta.device)
        s = s.to(theta.device, theta.dtype).reshape(-1).expand(B) if s.numel() == 1 else s.to(theta.device, theta.dtype).reshape(B)
        phi = self.obs(s)  # [B, n]
        return torch.einsum("bn,bnd->bd", phi, theta.view(B, self.n_basis, self.D))

    def deltas(self, c: torch.Tensor, params: dict) -> dict:
        """Per-sample weight deltas {name: [B, *shape]} from resolved coefficients c [B, D]."""
        B = c.shape[0]
        out = {}
        for i, e in enumerate(self.entries):
            W = params[e["name"]]
            W2 = W.reshape(e["out"], e["inn"])
            seg = c[:, e["offset"]: e["offset"] + e["gain_dim"] + e["add_dim"]]
            d = torch.zeros(B, e["out"], e["inn"], device=c.device, dtype=W.dtype)
            pos = 0
            if e["gain_dim"]:
                d = d + seg[:, : e["out"], None] * W2[None]
                pos = e["out"]
            scale = getattr(self, f"scale_{i}")
            if e["mode"] == "dense":
                d = d + scale * seg[:, pos: pos + e["add_dim"]].view(B, e["out"], e["inn"])
            elif e["mode"] == "subspace":
                r = e["r"]
                C = seg[:, pos: pos + r * r].view(B, r, r)
                d = d + scale * torch.einsum("or,brq,iq->boi", getattr(self, f"U_{i}"), C, getattr(self, f"V_{i}"))
            out[e["name"]] = d.view(B, *e["shape"])
        return out

    def describe(self) -> dict:
        return {
            "dim": self.dim, "D": self.D, "n_basis": self.n_basis, "obs_basis": self.obs.spec,
            "params": [{k: e[k] for k in ("name", "shape", "mode", "r", "gain_dim", "add_dim")} for e in self.entries],
        }
