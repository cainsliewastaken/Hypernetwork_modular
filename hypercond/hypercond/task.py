"""The Task interface: everything problem-specific lives behind this class.

A task supplies the wheel, the data, the condition x, the (common-random-number) banks, and a
per-sample loss. The package supplies everything else: update-space construction, the hypernetwork,
anchored oracle solves, the EM loop, the joint finisher, diagnostics, and evaluation.

Conventions
-----------
* Dataset items are dicts of tensors; batches are the default-collated dicts.
* Datasets should be ORDERED (by trajectory / time). Contiguous prefixes are used for dense target
  patches and learning-curve probes, and the tail of the target set is used for time-blocked
  validation. Random ordering silently turns "dense patch" into "thin coverage".
* ``datasets()`` must contain ``"train"``. Any other keys are evaluation splits (e.g. ``"val"``,
  ``"val_interp"``, ``"val_extrap"``, ``"test"``). Make them time-blocked; random splits hide
  non-stationarity completely.
* The wheel is always called through a :class:`~hypercond.conditioned.ConditionedWheel` ``model``::

      out = model(theta, s, *args, **kwargs)

  ``theta`` is ``None`` (plain base) or a ``[B, dim]`` coefficient tensor; ``s`` is the
  inference-observed variable (noise level, iteration index, stage, scale...) as a float or ``[B]``
  tensor, used to resolve the update ΔW(s). Positional tensor args with leading dim B are split per
  sample; kwargs are broadcast. If your wheel also consumes s, pass it in ``args`` too.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class Task(ABC):
    #: Wheel iterations per solve (K). Used for FLOP accounting (hypernet cost amortizes K-fold).
    solver_steps: int = 1

    #: Does the wheel already receive the condition x (concatenated / cross-attended)?
    #: True -> the information channel is exactly zero for deterministic tasks; any headroom is
    #: capacity localization (Channel 2). None = unknown. Used only for the headroom-gate verdict.
    condition_in_wheel: bool | None = None

    # ------------------------------------------------------------------ required
    @abstractmethod
    def build_wheel(self) -> nn.Module:
        """Return the (untrained or pretrained) wheel. Must be torch.func.vmap-compatible
        (no BatchNorm running-stat updates, no in-place ops on inputs, no data-dependent Python control flow on tensor values)."""

    @abstractmethod
    def datasets(self) -> dict[str, Dataset]:
        """Return {"train": ..., <eval splits>...}. Items are dicts of tensors."""

    @abstractmethod
    def condition(self, batch: dict) -> torch.Tensor:
        """Extract the hypernetwork input x from a batch: tensor [B, ...]."""

    @abstractmethod
    def make_bank(self, n: int, generator: torch.Generator, purpose: str) -> Any:
        """Return a bank of n stochastic draws (a pytree of tensors) shared across all samples.

        purpose is one of:
          "solve" — fixed bank for oracle solves (keep it stratified in s; CRN, non-optional),
          "eval"  — fixed bank for all reported scores (separate from the solve bank),
          "train" — fresh draws for shared-parameter training (base, meta, joint). This is where
                    regime allocation lives (e.g. concentrating diffusion noise levels at low τ).
        Return None if the task is deterministic.
        """

    @abstractmethod
    def loss(self, model, theta: torch.Tensor | None, batch: dict, bank: Any) -> torch.Tensor:
        """Per-sample task loss, shape [B], averaged over the bank. Must be differentiable in theta
        (oracle solves) and in wheel parameters (base / meta / joint training)."""

    # ------------------------------------------------------------------ optional
    def observed_range(self) -> tuple[float, float]:
        """Range of the observed variable s; used to normalize the update basis to u in [0, 1]."""
        return (0.0, 1.0)

    def build_encoder(self, hyper_cfg) -> nn.Module | None:
        """Optional problem-aware encoder x -> [B, F]; must expose ``out_dim``. Use phase-preserving
        features, never pooled summaries. None = default flatten+linear encoder."""
        return None

    def update_param_names(self, wheel: nn.Module) -> list[str] | None:
        """Optional explicit list of wheel parameter names the update may touch. None = use the
        update_space include/exclude regexes over all eligible (>=2-D operator) weights."""
        return None

    def target_indices(self, train_ds: Dataset) -> list[int] | None:
        """Optional explicit train indices that receive oracle targets (dense patches at valid
        spacing). None = contiguous prefix of size em.target_fraction."""
        return None

    def neighbor_pairs(self, train_ds: Dataset, indices: list[int]) -> torch.Tensor | None:
        """Optional [P, 2] tensor of index pairs that are neighbors at your data's similarity scale
        (e.g. adjacent saves on a trajectory). None = 1-NN in standardized condition space."""
        return None

    def evaluate(self, model, theta: torch.Tensor | None, batch: dict) -> dict[str, torch.Tensor]:
        """Optional deliverable metrics (e.g. sampled/rollout MSE), each [B]. Score loss and the
        deliverable can diverge — report both at decision points."""
        return {}

    def wheel_example_inputs(self, batch: dict) -> tuple[Any, tuple]:
        """Optional (s, args) for ONE wheel evaluation on this batch, for FLOP accounting."""
        raise NotImplementedError
