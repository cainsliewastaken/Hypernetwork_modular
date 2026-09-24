"""Synthetic smoke-test task (NOT a problem file): an undersized wheel regressing a stage-dependent
nonlinear map, with the condition concatenated into the wheel (pure Channel-2 regime).

Conditions lie on smooth ordered trajectories so the dense-patch / neighbor machinery is exercised.
The observed variable s is a "stage" in [0, 1] drawn from the bank.
"""
import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from hypercond.task import Task


class _DS(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return {"x": self.x[i]}


class Wheel(nn.Module):
    def __init__(self, d=8, h=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d + 1, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, d))

    def forward(self, x, s):
        return self.net(torch.cat([x, s.reshape(-1, 1).to(x.dtype)], dim=1))


class ToyTask(Task):
    solver_steps = 4
    condition_in_wheel = True

    def __init__(self, d=8, n_traj=8, traj_len=64, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.d = d
        self.A = torch.randn(d, d, generator=g) * 1.5
        self.Bm = torch.randn(d, d, generator=g) * 2.0
        t = torch.linspace(0, 2 * math.pi, traj_len)
        trajs = []
        for _ in range(n_traj):
            f = torch.rand(d, generator=g) * 2 + 0.5
            ph = torch.rand(d, generator=g) * 2 * math.pi
            trajs.append(torch.sin(t[:, None] * f + ph))  # ordered, smooth
        X = torch.cat(trajs)
        n_tr = int(0.75 * X.shape[0])
        self.X_tr, self.X_va = X[:n_tr], X[n_tr:]  # time/trajectory-blocked split

    def target(self, x, s):
        s = s.reshape(-1, 1)
        return torch.tanh(x @ self.A.T) * (1 + s) + 0.5 * torch.sin(x @ self.Bm.T) * s

    def build_wheel(self):
        return Wheel(self.d)

    def datasets(self):
        return {"train": _DS(self.X_tr), "val": _DS(self.X_va)}

    def condition(self, batch):
        return batch["x"]

    def make_bank(self, n, generator, purpose):
        if purpose == "train":
            return torch.rand(n, generator=generator)
        return (torch.arange(n) + torch.rand(n, generator=generator)) / n  # stratified

    def loss(self, model, theta, batch, bank):
        x = batch["x"]
        B = x.shape[0]
        out = 0.0
        for s in bank:
            sv = s.expand(B)
            pred = model(theta, sv, x, sv)
            out = out + (pred - self.target(x, sv)).pow(2).mean(1)
        return out / len(bank)

    def evaluate(self, model, theta, batch):
        x = batch["x"]
        sv = torch.ones(x.shape[0], device=x.device)
        return {"mse_at_s1": (model(theta, sv, x, sv) - self.target(x, sv)).pow(2).mean(1)}

    def wheel_example_inputs(self, batch):
        s = torch.ones(batch["x"].shape[0], device=batch["x"].device)
        return s, (batch["x"], s)
