"""How much does frame history tell us about the next increment y_t = omega_{t+1} - omega_t, beyond omega_t?

Cheap, model-free check before training anything with history (experiments/turb2d_log.md, section H).
Per Fourier mode k, fit a complex least-squares map (shared over time, fit on train, scored on val) from a
feature set to y_hat_t(k), and report the val relative MSE  sum|y - pred|^2 / sum|y|^2  (1 = predicts nothing),
overall and per |k| shell.

Feature sets (all per mode, complex coefficients, plus an intercept):
  markov-lin    omega_t
  markov-phys   omega_t, N_t, where N_t = FFT(u . grad omega) computed from omega_t (the Navier-Stokes advection term)
  markov-phys2  + N2_t, the second-order-in-time advection term (d/dt of N along dω/dt ~ N), also from omega_t
  +lag1 / +lag2 the same plus omega_{t-1} (and omega_{t-2})
  persistence   y_t ~ y_{t-1}  (no fit)

Reading it: if markov-phys+lag1 is well below markov-phys, history carries information about y that a
first-order function of omega_t doesn't (memory from unresolved scales or the sub-frame integration). The
per-mode linear fits are much weaker than the wheel, so the absolute numbers are not wheel errors, and a
nonlinear function of omega_t could recover part of any gap. What matters is the gap, and where in k it sits.

Usage (CPU, reads the packed cache; ~1 min):
  shifter --image=nersc/pytorch:26.01.01 python -u experiments/history_info.py
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tasks.turb2d import CACHE_DIR, DEFAULT_SPLITS, DEFAULT_STATS, N_GRID, cache_path  # noqa: E402

SHELLS = ((0, 8), (8, 32), (32, 64), (64, 1e9))  # |k| bands; the wheel's spectral weights cover |kx|, ky < 64


def wavenumbers(n=N_GRID):
    kx = torch.fft.fftfreq(n, 1.0 / n)
    ky = torch.fft.rfftfreq(n, 1.0 / n)
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    return KX, KY


def advection(what, KX, KY, vhat=None):
    """FFT of u(vhat) . grad omega, with u from the streamfunction psi = vhat / |k|^2, u = (dpsi/dy, -dpsi/dx);
    vhat defaults to omega_hat (the Navier-Stokes advection term). Constants and sign conventions don't matter:
    each mode gets its own fitted complex coefficient."""
    k2 = (KX ** 2 + KY ** 2).clamp_min(1.0)
    psi = (what if vhat is None else vhat) / k2
    n = N_GRID
    u = torch.fft.irfft2(1j * KY * psi, s=(n, n), norm="ortho")
    v = torch.fft.irfft2(-1j * KX * psi, s=(n, n), norm="ortho")
    wx = torch.fft.irfft2(1j * KX * what, s=(n, n), norm="ortho")
    wy = torch.fft.irfft2(1j * KY * what, s=(n, n), norm="ortho")
    return torch.fft.rfft2(u * wx + v * wy, norm="ortho")


def features(frames, lags, KX, KY):
    """frames [n, 1+lags+1, H, W] = omega_{t-lags}, ..., omega_t, omega_{t+1} (normalized).
    Returns y_hat [n, K] and a dict of per-mode features [n, K] (complex)."""
    F = torch.fft.rfft2(frames, norm="ortho")          # [n, T, H, W//2+1]
    cur = F[:, lags]
    N = advection(cur, KX, KY)
    # second-order Markov term: d/dt of the advection term along omega_t' ~ N, i.e. u(N).grad(omega) + u(omega).grad(N)
    f = {"w_t": cur, "N_t": N, "N2_t": advection(cur, KX, KY, vhat=N) + advection(N, KX, KY, vhat=cur)}
    for l in range(1, lags + 1):
        f[f"w_t-{l}"] = F[:, lags - l]
    y = F[:, lags + 1] - cur
    y_prev = cur - F[:, lags - 1]
    flat = lambda t: t.reshape(t.shape[0], -1)
    return flat(y), {k: flat(v) for k, v in f.items()}, flat(y_prev)


def load(allf, lo, ts, lags, mean, std):
    out = np.stack([allf[t - lags - lo: t + 2 - lo] for t in ts])   # [n, lags+2, H, W], contiguous reads
    return (torch.from_numpy(out.astype(np.float32)) - mean) / std


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lags", type=int, default=2)
    ap.add_argument("--n-train", type=int, default=2000, help="evenly spaced anchor times in the train split")
    ap.add_argument("--n-val", type=int, default=1000, help="evenly spaced anchor times in the val split")
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--ridge", type=float, default=1e-6, help="relative ridge (x mean diagonal of the Gram matrix)")
    args = ap.parse_args()
    torch.set_num_threads(min(32, torch.get_num_threads()))
    t0 = time.time()

    lo = min(a for a, _ in DEFAULT_SPLITS.values())
    hi = max(b for _, b in DEFAULT_SPLITS.values())
    allf = np.load(cache_path(lo, hi, CACHE_DIR), mmap_mode="r")
    st = np.load(DEFAULT_STATS)
    mean, std = float(st["mean"]), float(st["std"])
    L = args.lags
    (tr_a, tr_b), (va_a, va_b) = DEFAULT_SPLITS["train"], DEFAULT_SPLITS["val"]
    # anchor t needs omega_{t-L} .. omega_{t+1} inside the split (train) / t+1 inside the split (val; its history
    # may reach back into the train frames, exactly as the history dataset does)
    ts_tr = np.linspace(tr_a + L, tr_b - 1, args.n_train).round().astype(int)
    ts_va = np.linspace(va_a, va_b - 1, args.n_val).round().astype(int)
    KX, KY = wavenumbers()
    kmag = torch.sqrt(KX ** 2 + KY ** 2).reshape(-1)

    lagcols = [f"w_t-{l}" for l in range(1, L + 1)]
    models = {"markov-lin": ["w_t"], "markov-phys": ["w_t", "N_t"], "markov-phys2": ["w_t", "N_t", "N2_t"]}
    for l in range(1, L + 1):
        models[f"markov-lin+lag{l}"] = ["w_t"] + lagcols[:l]
        models[f"markov-phys+lag{l}"] = ["w_t", "N_t"] + lagcols[:l]
        models[f"markov-phys2+lag{l}"] = ["w_t", "N_t", "N2_t"] + lagcols[:l]

    def design(f, cols):
        X = torch.stack([f[c] for c in cols] + [torch.ones_like(f["w_t"])], dim=-1)   # [n, K, p]
        return X.to(torch.complex128)

    # --- fit: accumulate per-mode normal equations over train chunks
    G = {m: 0 for m in models}
    bvec = {m: 0 for m in models}
    ystats = [0.0, 0.0]
    for i in range(0, len(ts_tr), args.chunk):
        y, f, _ = features(load(allf, lo, ts_tr[i:i + args.chunk], L, mean, std), L, KX, KY)
        y = y.to(torch.complex128)
        for m, cols in models.items():
            X = design(f, cols)
            G[m] = G[m] + torch.einsum("nkp,nkq->kpq", X.conj(), X)
            bvec[m] = bvec[m] + torch.einsum("nkp,nk->kp", X.conj(), y)
    print(f"[fit] {len(ts_tr)} train anchors, {len(models)} models ({time.time() - t0:.0f}s)", flush=True)
    coef = {}
    for m in models:
        g = G[m]
        p = g.shape[-1]
        lam = args.ridge * torch.diagonal(g, dim1=-2, dim2=-1).real.mean(-1, keepdim=True).clamp_min(1e-30)
        g = g + lam[..., None] * torch.eye(p, dtype=g.dtype)
        coef[m] = torch.linalg.solve(g, bvec[m].unsqueeze(-1)).squeeze(-1)   # [K, p]

    # --- score on val
    names = list(models) + ["persistence"]
    sse = {m: torch.zeros(kmag.numel(), dtype=torch.float64) for m in names}
    energy = torch.zeros(kmag.numel(), dtype=torch.float64)
    w_energy = torch.zeros(kmag.numel(), dtype=torch.float64)
    for i in range(0, len(ts_va), args.chunk):
        fr = load(allf, lo, ts_va[i:i + args.chunk], L, mean, std)
        y, f, y_prev = features(fr, L, KX, KY)
        y = y.to(torch.complex128)
        energy += y.abs().pow(2).sum(0)
        w_energy += f["w_t"].to(torch.complex128).abs().pow(2).sum(0)
        for m, cols in models.items():
            pred = torch.einsum("nkp,kp->nk", design(f, cols), coef[m])
            sse[m] += (y - pred).abs().pow(2).sum(0)
        sse["persistence"] += (y - y_prev.to(torch.complex128)).abs().pow(2).sum(0)
    print(f"[val] {len(ts_va)} val anchors ({time.time() - t0:.0f}s)", flush=True)

    # --- report
    bands = [("all", torch.ones_like(kmag, dtype=torch.bool))] + \
            [(f"|k| {a:g}-{b:g}" if b < 1e8 else f"|k| >= {a:g}", (kmag >= a) & (kmag < b)) for a, b in SHELLS]
    tot = float(energy.sum())
    print(f"\n|y|^2 / |omega_t|^2 (val) = {tot / float(w_energy.sum()):.4g}   "
          f"(y std {math.sqrt(tot / len(ts_va) / N_GRID ** 2):.4g} in normalized units)")
    print("share of y energy by band: " + ", ".join(f"{n}: {float(energy[mk].sum()) / tot:.3f}" for n, mk in bands[1:]))
    w = max(len(n) for n in names)
    print(f"\nval relative MSE (1 = no skill); lags={L}, train anchors={len(ts_tr)}, val anchors={len(ts_va)}")
    print(f"{'model':<{w}}  " + "  ".join(f"{n:>12}" for n, _ in bands))
    for m in names:
        row = [float(sse[m][mk].sum()) / max(float(energy[mk].sum()), 1e-30) for _, mk in bands]
        print(f"{m:<{w}}  " + "  ".join(f"{r:12.4f}" for r in row))
    print(f"\n({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
