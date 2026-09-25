"""Mode-net hypernetwork for the turb2d FNO wheel: a port of the `spectral_head=full` hypernet from
HyperNetwork-Research/diffusion/hypernet.py (the best score-trained head there).

Global path:   x -> LowModeEncoder (Fourier features) -> FeatureNormalizer -> residual MLP trunk -> z
Spectral path: for every complex spectral weight (c_in, c_out, m1, m2), one shared 1x1-conv MLP runs over
               the mode grid. Per-mode inputs: phase (re, im) and log1p|x_hat(k)| of the condition at that
               mode, the mode coordinates (kx, ky, log(1+|k|)), and a projection of z. Per-mode output: a
               full complex (c_in, c_out) matrix, times `delta_scale`.
               With history L > 0 (condition channels omega_t, omega_{t-1}, ..., omega_{t-L}), each mode also gets,
               per lag l, the log-magnitude ratio log1p|x_hat_t| - log1p|x_hat_{t-l}| and the phase difference
               (cos, sin) of x_hat_t vs x_hat_{t-l}: a direct estimate of that mode's growth and rotation rate.
Real weights:  a dense head z -> full tensor, times `delta_scale`.

The output is the flat coefficient vector theta of a `dense`, gain-free, `constant`-basis UpdateSpace
with additive_scale=none, so dW = theta laid out in the space's own entry order. obs_basis=affine is supported
(the heads emit one block per basis function: dW(tau) = g0 + u(tau) g1). The last layers are
zero-initialized, so theta = 0 at the start (identity at zero).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hypercond.hypernet import FeatureNormalizer, ResBlock

_EPS = 1e-8


def mode_coords(m1: int, m2: int, neg: bool, device) -> torch.Tensor:
    """(3, m1, m2): kx_n, ky_n, log(1+|k|); kx negative for the weights2 (negative-frequency) rows."""
    kx = torch.arange(-m1, 0, device=device, dtype=torch.float32) if neg \
        else torch.arange(0, m1, device=device, dtype=torch.float32)
    ky = torch.arange(0, m2, device=device, dtype=torch.float32)
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    d = float(max(m1, m2))
    return torch.stack([KX / d, KY / d, torch.log1p(torch.sqrt(KX ** 2 + KY ** 2))], dim=0)


class AxialModeAttention(nn.Module):
    """Row + column attention on the (kx, ky) mode grid (port of the old repo's AxialModeAttention): each mode
    sees its full kx row and ky column, i.e. the whole grid in two hops, without all-to-all over 64^2 modes."""

    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        while dim % heads:
            heads -= 1
        self.heads, self.head_dim = heads, dim // heads
        self.norm = nn.GroupNorm(1, dim)
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.xavier_normal_(self.qkv.weight)
        nn.init.xavier_normal_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = (t.view(B, self.heads, self.head_dim, H, W) for t in self.qkv(self.norm(x)).chunk(3, dim=1))

        def rows(t):  # sequences along W, one per (B, H)
            return t.permute(0, 3, 1, 4, 2).reshape(B * H, self.heads, W, self.head_dim)

        def cols(t):  # sequences along H, one per (B, W)
            return t.permute(0, 4, 1, 3, 2).reshape(B * W, self.heads, H, self.head_dim)

        row = F.scaled_dot_product_attention(rows(q), rows(k), rows(v))
        row = row.reshape(B, H, self.heads, W, self.head_dim).permute(0, 2, 4, 1, 3)
        col = F.scaled_dot_product_attention(cols(q), cols(k), cols(v))
        col = col.reshape(B, W, self.heads, H, self.head_dim).permute(0, 2, 4, 3, 1)
        return x + self.proj((row + col).reshape(B, C, H, W))


class SpectralModeNet(nn.Module):
    """Shared per-mode MLP (1x1 convs over the mode grid) emitting a full complex (c_in, c_out) per mode."""

    def __init__(self, c_in: int, c_out: int, z_in: int, z_dim: int = 64, hidden: int = 64, layers: int = 2,
                 c_state: int = 1, n_basis: int = 1, attn_layers: int = 0, attn_heads: int = 8, history: int = 0):
        super().__init__()
        self.c_in, self.c_out, self.n_basis = c_in, c_out, n_basis
        self.c_state, self.history = c_state, history
        self.z_proj = nn.Linear(z_in, z_dim)
        f_in = 3 * c_state + 3 * history + 3 + z_dim
        mods = [nn.Conv2d(f_in, hidden, 1), nn.GELU()]
        # optional axial attention across modes after the stem (old repo: mode_mix=axial, 2 layers, 8 heads)
        mods += [AxialModeAttention(hidden, attn_heads) for _ in range(attn_layers)]
        for _ in range(layers - 1):
            mods += [nn.Conv2d(hidden, hidden, 1), nn.GELU()]
        self.net = nn.Sequential(*mods)
        self.out = nn.Conv2d(hidden, n_basis * 2 * c_in * c_out, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, xhat_block, z, coords):
        B, _, m1, m2 = xhat_block.shape
        mag = (xhat_block.real.pow(2) + xhat_block.imag.pow(2)).clamp_min(_EPS * _EPS).sqrt()
        cur, cm = xhat_block[:, :self.c_state], mag[:, :self.c_state]
        hist = []
        if self.history:  # channels c_state.. are omega_{t-1}, ..., omega_{t-L} (single-channel state)
            past, pm = xhat_block[:, self.c_state:], mag[:, self.c_state:]
            rot = cur[:, :1] * past.conj() / (cm[:, :1] * pm)             # unit phasor: phase(x_t) - phase(x_{t-l})
            hist = [torch.log1p(cm[:, :1]) - torch.log1p(pm), rot.real, rot.imag]
        feats = torch.cat([
            cur.real / cm, cur.imag / cm, torch.log1p(cm), *hist,
            coords.unsqueeze(0).expand(B, -1, -1, -1),
            self.z_proj(z)[:, :, None, None].expand(B, -1, m1, m2),
        ], dim=1)
        o = self.out(self.net(feats))                                      # (B, nb*2*ci*co, m1, m2)
        o = o.view(B, self.n_basis, self.c_in, self.c_out, 2, m1, m2).permute(0, 1, 2, 3, 5, 6, 4)
        return o.reshape(B, self.n_basis, -1)                              # per basis: real view (ci, co, m1, m2, 2)


class ModeNetHyper(nn.Module):
    def __init__(self, encoder: nn.Module, space, hyper_cfg, delta_scale: float = 0.02, mode_z_dim: int = 64,
                 mode_hidden: int = 64, mode_layers: int = 2, mode_attn_layers: int = 0, mode_attn_heads: int = 8,
                 history: int = 0):
        super().__init__()
        if any(e["gain_dim"] for e in space.entries) \
                or any(e["mode"] != "dense" for e in space.entries) or space.cfg.additive_scale != "none":
            raise ValueError("ModeNetHyper needs update_space: additive=dense, gain=false, additive_scale=none, "
                             "spectral=same (obs_basis: constant, or affine for tau-resolved updates)")
        nb = self.n_basis = space.n_basis
        self.encoder = encoder
        feat = encoder.out_dim
        w = hyper_cfg.width
        self.norm = FeatureNormalizer(feat, hyper_cfg.feature_clip, hyper_cfg.std_floor, hyper_cfg.norm_momentum)
        self.inp = nn.Linear(feat, w)
        self.trunk = nn.Sequential(*[ResBlock(w) for _ in range(hyper_cfg.depth)], nn.LayerNorm(w))
        self.delta_scale = float(delta_scale)
        self.jobs = []  # (kind, key, add_dim, meta) in space entry order
        self.mode_nets = nn.ModuleDict()
        self.dense = nn.ModuleDict()
        for e in space.entries:
            key = e["name"].replace(".", "__")
            if e["complex"] and len(e["shape"]) == 4:
                ci, co, m1, m2 = e["shape"]
                self.mode_nets[key] = SpectralModeNet(ci, co, w, mode_z_dim, mode_hidden, mode_layers, n_basis=nb,
                                                      attn_layers=mode_attn_layers, attn_heads=mode_attn_heads,
                                                      history=history)
                self.register_buffer(f"coords_{key}", mode_coords(m1, m2, e["name"].endswith("weights2"), "cpu"))
                self.jobs.append(("spectral", key, e["add_dim"], (m1, m2, e["name"].endswith("weights2"))))
            else:
                lin = nn.Linear(w, nb * e["add_dim"])
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
                self.dense[key] = lin
                self.jobs.append(("dense", key, e["add_dim"], None))
        self.jobs_dim = {key: d for _, key, d, _ in self.jobs}
        self.out_dim = space.dim

    def forward(self, x):
        z = self.trunk(self.inp(self.norm(self.encoder(x))))
        xhat = torch.fft.rfft2(x.float())                                   # (B, 1+L, H, W//2+1)
        parts = []
        for kind, key, _, meta in self.jobs:
            if kind == "spectral":
                m1, m2, neg = meta
                blk = xhat[:, :, -m1:, :m2] if neg else xhat[:, :, :m1, :m2]
                parts.append(self.mode_nets[key](blk, z, getattr(self, f"coords_{key}")))
            else:
                parts.append(self.dense[key](z).view(-1, self.n_basis, self.jobs_dim[key]))
        # theta layout = [basis 0: all entries | basis 1: all entries ...], as UpdateSpace.coefficients expects
        return self.delta_scale * torch.cat(parts, dim=2).reshape(x.shape[0], -1)
