"""FNO2D score field, vendored from HyperNetwork-Research/diffusion (fno_spectral_bias.py + models.py).

Parameter names, shapes and the forward pass match ``FNO2DScoreField`` there, so its checkpoints load
with ``strict=True``. The optional attention blocks (``spatial_attn`` / ``k_attn``) are left out: they
are off in the pretrained concat-D2 wheels and have no parameters when off.

Spectral weights (``weights1``/``weights2``, ``spectral_bias``) are native ``cfloat``. The update space
handles them through their real view (see ``hypercond.update_space``).

vmap notes: no BatchNorm, no data-dependent control flow (only shape checks), and the spectral-bias
padding is out-of-place.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = int(dim)

    def forward(self, t):
        if t.dim() == 0:
            t = t[None]
        t = t.float().reshape(-1)
        half = self.dim // 2
        if half <= 0:
            return t[:, None]
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        W_rfft = W // 2 + 1
        low = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        high = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        mid_h = H - 2 * self.modes1
        zeros_mid_h = torch.zeros(batchsize, self.out_channels, mid_h, self.modes2,
                                  dtype=torch.cfloat, device=x.device)
        left_block = torch.cat([low, zeros_mid_h, high], dim=2)
        zeros_right = torch.zeros(batchsize, self.out_channels, H, W_rfft - self.modes2,
                                  dtype=torch.cfloat, device=x.device)
        out_ft = torch.cat([left_block, zeros_right], dim=3)
        return torch.fft.irfft2(out_ft, s=(H, W))


class PointwiseMLP2d(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, 1),
        )

    def forward(self, x):
        return self.net(x)


class FNO2DScoreField(nn.Module):
    """Called as ``model(t, x)``: t scalar / [B] noise level, x [B, C, H, W] (or [C, H, W])."""

    def __init__(self, in_channels=1, out_channels=1, modes1=16, modes2=16, width=64, num_layers=4,
                 use_layernorm=False, use_time_embedding=True, time_conditioning="film", time_embed_dim=128,
                 time_scale=1.0, readout_hidden_mult=2, uses_condition_concat=False, bias_modes=16):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.width = int(width)
        self.num_layers = int(num_layers)
        self.use_layernorm = bool(use_layernorm)
        self.use_time_embedding = bool(use_time_embedding) and str(time_conditioning) != "none"
        self.time_conditioning = str(time_conditioning)
        self.time_scale = float(time_scale)
        self.uses_condition_concat = bool(uses_condition_concat)
        self.bias_modes = int(bias_modes)

        self.p = nn.Linear(self.in_channels + 2, self.width)
        self.spec_convs = nn.ModuleList([SpectralConv2d(self.width, self.width, int(modes1), int(modes2))
                                         for _ in range(self.num_layers)])
        self.mlps = nn.ModuleList([PointwiseMLP2d(self.width, self.width, self.width)
                                   for _ in range(self.num_layers)])
        self.ws = nn.ModuleList([nn.Conv2d(self.width, self.width, 1) for _ in range(self.num_layers)])

        if self.bias_modes > 0:
            self.spectral_bias = nn.ParameterList([
                nn.Parameter(torch.zeros(self.width, 2 * self.bias_modes, self.bias_modes, dtype=torch.cfloat))
                for _ in range(self.num_layers)
            ])
        else:
            self.spectral_bias = None

        self.norms = nn.ModuleList([nn.LayerNorm(self.width) for _ in range(self.num_layers)]) \
            if self.use_layernorm else None

        if self.use_time_embedding:
            n_sites = self.num_layers + 1
            self.time_mlp = nn.Sequential(
                SinusoidalTimeEmbedding(int(time_embed_dim)),
                nn.Linear(int(time_embed_dim), self.width),
                nn.GELU(),
                nn.Linear(self.width, 2 * n_sites * self.width),
            )
            nn.init.zeros_(self.time_mlp[-1].weight)
            nn.init.zeros_(self.time_mlp[-1].bias)
        else:
            self.time_mlp = None

        readout_hidden = max(self.width, int(readout_hidden_mult) * self.width)
        self.readout = nn.Sequential(
            nn.Conv2d(self.width, readout_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(readout_hidden, self.out_channels, kernel_size=1),
        )

    def get_grid(self, batchsize, size_x, size_y, device, dtype):
        gridx = torch.linspace(0, 1, size_x, device=device, dtype=dtype).view(1, size_x, 1, 1)
        gridx = gridx.repeat(batchsize, 1, size_y, 1)
        gridy = torch.linspace(0, 1, size_y, device=device, dtype=dtype).view(1, 1, size_y, 1)
        gridy = gridy.repeat(batchsize, size_x, 1, 1)
        return torch.cat([gridx, gridy], dim=-1)

    def _time_params(self, t, batch_size, device, dtype):
        if self.time_mlp is None:
            return None
        t_tensor = torch.as_tensor(t, device=device, dtype=dtype) * self.time_scale
        if t_tensor.dim() == 0:
            t_tensor = t_tensor.expand(batch_size)
        else:
            t_tensor = t_tensor.reshape(-1)
            if t_tensor.numel() == 1:
                t_tensor = t_tensor.expand(batch_size)
            elif t_tensor.numel() != batch_size:
                raise ValueError(f"Time tensor has {t_tensor.numel()} values but batch size is {batch_size}.")
        film = self.time_mlp(t_tensor).to(dtype=dtype)
        return film.view(batch_size, self.num_layers + 1, 2, self.width, 1, 1)

    def _apply_time(self, h, time_params, site_idx):
        if time_params is None:
            return h
        scale = time_params[:, site_idx, 0]
        shift = time_params[:, site_idx, 1]
        if self.time_conditioning == "film":
            return h * (1.0 + scale) + shift
        if self.time_conditioning == "add":
            return h + shift
        return h

    def _bias_field_spatial(self, k, H, W):
        bf = self.spectral_bias[k]
        bm = self.bias_modes
        W_rfft = W // 2 + 1
        # out-of-place pad (an in-place scatter into zeros is illegal under vmap)
        pad_ky = W_rfft - bm
        pad_kx_far = H - bm
        pos = F.pad(bf[..., :bm, :], (0, pad_ky, 0, pad_kx_far))
        neg = F.pad(bf[..., bm:, :], (0, pad_ky, pad_kx_far, 0))
        return torch.fft.irfft2(pos + neg, s=(H, W))

    def forward(self, t, x):
        added_batch_dim = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            added_batch_dim = True
        if x.dim() != 4:
            raise ValueError(f"Expected x with shape (C,H,W) or (B,C,H,W), got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"FNO2DScoreField expected {self.in_channels} input channels, got {x.shape[1]}.")

        b, _, h, w = x.shape
        x_hw = x.permute(0, 2, 3, 1).contiguous()
        x_hw = torch.cat([x_hw, self.get_grid(b, h, w, x.device, x.dtype)], dim=-1)

        h_state = self.p(x_hw).permute(0, 3, 1, 2).contiguous()
        time_params = self._time_params(t, b, x.device, x.dtype)
        h_state = self._apply_time(h_state, time_params, site_idx=0)

        for k in range(self.num_layers):
            h_state = self.mlps[k](self.spec_convs[k](h_state)) + self.ws[k](h_state)
            if self.spectral_bias is not None:
                bias = self._bias_field_spatial(k, h, w)
                h_state = h_state + (bias.unsqueeze(0) if bias.dim() == 3 else bias)
            h_state = self._apply_time(h_state, time_params, site_idx=k + 1)
            if self.use_layernorm:
                h_state = self.norms[k](h_state.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
            if k < self.num_layers - 1:
                h_state = F.gelu(h_state)

        out = self.readout(h_state)
        return out.squeeze(0) if added_batch_dim else out

    # ------------------------------------------------------------------ per-sample weights, no vmap
    def forward_batched_params(self, params: dict, t, x):
        """Same as ``forward(t, x)`` but any parameter in ``params`` may carry a leading batch dim B
        (per-sample weights); missing names use the module's own parameters. Every layer here is linear
        in its weight (linear / 1x1 conv / spectral einsum), so per-sample weights are applied as batched
        tensor ops, which keeps the GPU busy where torch.func.vmap would be dispatcher-bound."""
        P = dict(self.named_parameters())
        P.update(params)
        b, _, h, w = x.shape

        def lin(z, name):  # z [B, ..., in]
            W, bias = P[f"{name}.weight"], P[f"{name}.bias"]
            if W.dim() == 2:
                y = z @ W.transpose(0, 1)
            else:
                y = torch.einsum("b...i,boi->b...o", z, W)
            return y + (bias if bias.dim() == 1 else bias.view(b, *([1] * (y.dim() - 2)), -1))

        def conv1(z, name):  # 1x1 conv, z [B, C, H, W]
            W, bias = P[f"{name}.weight"][..., 0, 0], P[f"{name}.bias"]
            eq = "bihw,oi->bohw" if W.dim() == 2 else "bihw,boi->bohw"
            y = torch.einsum(eq, z, W)
            return y + (bias.view(1, -1, 1, 1) if bias.dim() == 1 else bias.view(b, -1, 1, 1))

        def spec(z, k):
            conv = self.spec_convs[k]
            W1, W2 = P[f"spec_convs.{k}.weights1"], P[f"spec_convs.{k}.weights2"]
            m1, m2 = conv.modes1, conv.modes2
            z_ft = torch.fft.rfft2(z)
            eq1 = "bixy,ioxy->boxy" if W1.dim() == 4 else "bixy,bioxy->boxy"
            eq2 = "bixy,ioxy->boxy" if W2.dim() == 4 else "bixy,bioxy->boxy"
            low = torch.einsum(eq1, z_ft[:, :, :m1, :m2], W1)
            high = torch.einsum(eq2, z_ft[:, :, -m1:, :m2], W2)
            zeros_mid = torch.zeros(b, conv.out_channels, h - 2 * m1, m2, dtype=torch.cfloat, device=z.device)
            zeros_right = torch.zeros(b, conv.out_channels, h, w // 2 + 1 - m2, dtype=torch.cfloat, device=z.device)
            out_ft = torch.cat([torch.cat([low, zeros_mid, high], dim=2), zeros_right], dim=3)
            return torch.fft.irfft2(out_ft, s=(h, w))

        x_hw = torch.cat([x.permute(0, 2, 3, 1), self.get_grid(b, h, w, x.device, x.dtype)], dim=-1)
        h_state = lin(x_hw, "p").permute(0, 3, 1, 2)

        time_params = None
        if self.time_mlp is not None:
            tt = torch.as_tensor(t, device=x.device, dtype=x.dtype) * self.time_scale
            tt = tt.reshape(-1).expand(b) if tt.numel() == 1 else tt.reshape(b)
            e = F.gelu(lin(self.time_mlp[0](tt), "time_mlp.1"))
            time_params = lin(e, "time_mlp.3").to(x.dtype).view(b, self.num_layers + 1, 2, self.width, 1, 1)
        h_state = self._apply_time(h_state, time_params, site_idx=0)

        for k in range(self.num_layers):
            x1 = conv1(F.gelu(conv1(spec(h_state, k), f"mlps.{k}.net.0")), f"mlps.{k}.net.2")
            h_state = x1 + conv1(h_state, f"ws.{k}")
            if self.spectral_bias is not None:
                bf = P[f"spectral_bias.{k}"]
                bm, pad_ky, pad_kx_far = self.bias_modes, w // 2 + 1 - self.bias_modes, h - self.bias_modes
                pos = F.pad(bf[..., :bm, :], (0, pad_ky, 0, pad_kx_far))
                neg = F.pad(bf[..., bm:, :], (0, pad_ky, pad_kx_far, 0))
                bias = torch.fft.irfft2(pos + neg, s=(h, w))
                h_state = h_state + (bias.unsqueeze(0) if bias.dim() == 3 else bias)
            h_state = self._apply_time(h_state, time_params, site_idx=k + 1)
            if self.use_layernorm:
                h_state = self.norms[k](h_state.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            if k < self.num_layers - 1:
                h_state = F.gelu(h_state)

        return conv1(F.gelu(conv1(h_state, "readout.0")), "readout.2")
