"""GPU check for the turb2d wheel: forward_batched_params == vmap path (outputs and theta grads), plus
step timings. Needs the pretrained w16 checkpoint and norm stats on $SCRATCH.

    python -u tests/check_fno_batched.py
"""
import time

import torch

from hypercond.conditioned import ConditionedWheel
from hypercond.config import load_config
from hypercond.update_space import UpdateSpace
from tasks.turb2d import Turb2DTask


def step(cw, task, theta, batch, bank):
    th = theta.detach().clone().requires_grad_(True)
    loss = task.loss(cw, th, batch, bank).mean()
    loss.backward()
    return loss.detach(), th.grad


def main():
    torch.manual_seed(0)
    dev = torch.device("cuda")
    cfg = load_config("configs/turb2d.yaml")
    task = Turb2DTask(wheel="w16")
    wheel = task.build_wheel().to(dev)
    space = UpdateSpace(wheel, cfg.update_space, task.observed_range()).to(dev)
    fast, slow = ConditionedWheel(wheel, space), ConditionedWheel(wheel, space, batched=False)
    for p in wheel.parameters():
        p.requires_grad_(False)

    B = 6
    batch = {"x": torch.randn(B, 1, 256, 256, device=dev), "y": 0.1 * torch.randn(B, 1, 256, 256, device=dev)}
    bank = {k: (v.to(dev) if v is not None else None) for k, v in task.make_bank(4, torch.Generator().manual_seed(1), "eval").items()}
    theta = 0.05 * torch.randn(B, space.dim, device=dev)

    tv = torch.full((B,), 0.3, device=dev)
    inp = torch.cat([batch["y"], batch["x"]], 1)
    with torch.no_grad():
        a, b_ = fast(theta, tv, tv, inp), slow(theta, tv, tv, inp)
        base, zero = wheel(tv, inp), fast(torch.zeros_like(theta), tv, tv, inp)
    rel = lambda u, v: float((u - v).norm() / v.norm().clamp_min(1e-12))
    print(f"forward  batched vs vmap rel diff = {rel(a, b_):.2e}   theta=0 vs base = {rel(zero, base):.2e}   "
          f"update effect = {rel(a, base):.2e}")
    la, ga = step(fast, task, theta, batch, bank)
    lb, gb = step(slow, task, theta, batch, bank)
    print(f"loss     batched={la.item():.6f} vmap={lb.item():.6f}   grad rel diff = {rel(ga, gb):.2e}")

    def timeit(cw, Bt, bank_n=8, reps=3):
        bt = {"x": torch.randn(Bt, 1, 256, 256, device=dev), "y": 0.1 * torch.randn(Bt, 1, 256, 256, device=dev)}
        bk = {"tau": torch.rand(bank_n, device=dev) * 0.999 + 1e-3, "eps": None}
        th = 0.05 * torch.randn(Bt, space.dim, device=dev)
        step(cw, task, th, bt, bk); torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(reps):
            step(cw, task, th, bt, bk)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / reps
        return dt, torch.cuda.max_memory_allocated() / 2**30

    for name, cw, Bs in [("vmap", slow, [6]), ("batched", fast, [6, 24, 48])]:
        for Bt in Bs:
            dt, mem = timeit(cw, Bt)
            print(f"timing   {name:8s} B={Bt:3d} bank=8: {dt:6.3f} s/step  {Bt / dt:7.1f} samples/s  peak {mem:5.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
