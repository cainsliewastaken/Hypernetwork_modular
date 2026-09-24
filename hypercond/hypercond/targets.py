"""Target storage and datasets."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset


@dataclass
class TargetStore:
    indices: torch.Tensor          # [N] train-set indices (sorted)
    theta: torch.Tensor            # [N, dim]
    loss_solve: torch.Tensor       # [N] task loss on the solve bank
    loss_eval: torch.Tensor        # [N] task loss on the eval bank (honest depth)
    round: int = 0
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return int(self.indices.numel())

    def rows(self) -> dict:
        return {int(i): k for k, i in enumerate(self.indices.tolist())}

    def subset(self, rows: torch.Tensor) -> "TargetStore":
        return TargetStore(self.indices[rows], self.theta[rows], self.loss_solve[rows], self.loss_eval[rows],
                           self.round, dict(self.meta))

    def save(self, path):
        torch.save(self.__dict__, path)

    @classmethod
    def load(cls, path):
        return cls(**torch.load(path, map_location="cpu", weights_only=False))


class IndexedDataset(Dataset):
    """View of a base dataset at given indices; each item gains "_idx" (the original index)."""

    def __init__(self, base: Dataset, indices=None):
        self.base = base
        self.indices = list(range(len(base))) if indices is None else [int(i) for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i = self.indices[k]
        item = dict(self.base[i])
        item["_idx"] = torch.tensor(i)
        return item


class TargetDataset(IndexedDataset):
    def __init__(self, base: Dataset, indices, theta: torch.Tensor):
        super().__init__(base, indices)
        self.theta = theta

    def __getitem__(self, k):
        item = super().__getitem__(k)
        item["_target"] = self.theta[k]
        return item
