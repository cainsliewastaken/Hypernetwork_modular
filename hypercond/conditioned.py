"""ConditionedWheel: run the wheel with per-sample weights W + ΔW_i(s)."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.func import functional_call, vmap
from torch.utils._pytree import tree_map


class ConditionedWheel(nn.Module):
    """Call as ``model(theta, s, *args, **kwargs)``.

    * theta None  -> plain base wheel, ``wheel(*args, **kwargs)`` (no vmap).
    * theta [B, dim] -> per-sample effective weights, applied with torch.func.vmap over the batch.
      Positional tensor args whose leading dim equals B are split per sample (each sample sees a
      batch of 1); everything else (and all kwargs) is broadcast unchanged.

    Gradients flow to theta (oracle solves, distillation with task loss) and to the wheel's own
    parameters (joint finisher, meta refinement), including through the multiplicative gain.
    """

    def __init__(self, wheel: nn.Module, space, chunk_size: int | None = None):
        super().__init__()
        self.wheel = wheel
        self.space = space
        self.chunk_size = chunk_size or None

    def forward(self, theta, s, *args, **kwargs):
        if theta is None:
            return self.wheel(*args, **kwargs)
        B = theta.shape[0]
        params = dict(self.wheel.named_parameters())
        c = self.space.coefficients(theta, s)
        deltas = self.space.deltas(c, params)
        batched = {n: params[n].unsqueeze(0) + d for n, d in deltas.items()}

        split = tuple(torch.is_tensor(a) and a.dim() > 0 and a.shape[0] == B for a in args)
        in_dims = (0,) + tuple(0 if sp else None for sp in split)

        def one(pb, *a):
            a = tuple(x.unsqueeze(0) if sp else x for x, sp in zip(a, split))
            out = functional_call(self.wheel, pb, a, kwargs, strict=False)
            return tree_map(lambda t: t.squeeze(0) if torch.is_tensor(t) else t, out)

        return vmap(one, in_dims=in_dims, randomness="different", chunk_size=self.chunk_size)(batched, *args)
