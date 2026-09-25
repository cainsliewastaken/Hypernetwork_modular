# turb2d (data_lowres, w16 concat-FNO wheel): experiment log

Running log of every experiment on the 2D turbulence task in this repo. Newest results at the bottom of each table. Keep it updated after every run, including failures.

## Setup (shared by all runs unless stated)

- **Data:** `training_data/data_lowres`, frames 10000–30700, packed to `$SCRATCH/hypernet_diffusion_cache/omega_data_lowres_10000_30700_f32.npy`. Normalized with the old stats file (mean ≈ 0, std ≈ 10.42).
- **Splits (time-blocked):** train 10000–20000, val 20001–22000, test 22001–30700. Val numbers below use the first 512 val pairs (`eval.max_batches=32`, batch 16) and a fixed 32-draw τ-stratified eval bank (seed 987).
- **Wheel:** pretrained w16 (`experiment_concat_d2_narrow/CONCAT_D2/model_best.pt`), frozen. Val base score loss on this protocol is **0.245518**.
- **Val numbers in sections A–E** use the earlier eval bank (see Standing rules).
- **Loss:** VP-SDE denoising score matching on the residual x_{t+1} − x_t, with the condition concatenated (as in the old repo).
- **Margin** = 1 − cond/base. Positive means the hypernet is better than the frozen wheel.
- **Compute:** interactive job 58844824 (4× A100 40 GB, m5326), except the first session (job 58842806).

## Standing rules

- **Validation (from section F on) is fixed and never changes between runs:** 512 val points sampled uniformly without replacement from the whole val split (frames 20001–22000), each scored on 16 draws of τ ~ U[0.001, 1] with fresh Gaussian noise. The whole draw (points, τ, noise) is seeded (seed 20240901 and each point's index), so every check of every run scores the identical random draw. The metric is the plain mean score loss (std·score + ε)² over all points and draws. The user asked for this as the most honest validation, with no τ grid, no contiguous frame block and no noise shared across samples. Training-side choices (τ sampling, lr, batch, heads) are free to change.
- **Learning rate (2026-09-24):** the old hypernet lr (peak 3e-4, cosine → 3e-5) is **too high for this loss**, i.e. score training with low-τ sampling (τ ∝ u²), which puts the gradient where the loss and its variance are largest. it6 (mode-net, depth 4) peaked at +2.08% at epoch 6, then oscillated and degraded at lr ≈ 2.5e-4 (epoch 10.5: −5.0%, epoch 11: −4.6%, train loss rising 0.47 → 0.50). Next runs use peak 1e-4.
- Runs before the switch (sections A–E) were scored on an earlier eval bank: the **first** 512 val pairs, 32 stratified τ, noise shared across samples (seed 987). Their numbers are comparable with each other but **not** with runs from section F on.

## Reference: old repo (HyperNetwork-Research)

| run | what | result |
|---|---|---|
| CONCAT_D2 (w16 wheel training) | val by epoch: 0: 1.014, 0.5: 0.867, 1: 0.549, 2: 0.474, 4: 0.364, 16: 0.303, 216: 0.252 | the wheel used here |
| D2_hyper_full (history run) | w16 frozen, `spectral_head=full` mode-net head, lr 3e-4 → 3e-5, batch 6/GPU × 4. Val by epoch: 1: −0.5%, 5: +1.2%, 14: +2.7%, 29: +7%, 96: +14%, best 0.2046 (+19%) vs base 0.2517 | **the bar to beat (w16)** |
| D2_w32_hyper_full_attn_wd | w32, full head + trunk attention + weight decay | 0.168 vs frozen 0.21 (best score-trained run) |
| D2_w32_hyper_upsample | w32, coarse-grid upsample head | 0.194 (weaker than full) |
| oracle / gate / distillation work | see `diagnostics/ORACLE_FORCING_SUMMARY.md` | oracle targets don't transfer; **don't use the gate** |

## FLOP accounting (to be measured with `eval.py --diag flops`)

The thesis compares against a plain wheel given ε more FLOPs, with ε = hypernet FLOPs / (K · wheel FLOPs), K = 1000 sampler steps. Rough estimates (**unmeasured**):
- w16 wheel forward at 256²: ~0.7 GFLOP, so ~700 GFLOP per field over 1000 steps.
- Linear head from width 512 to D outputs: ~2·512·D FLOP once per field. D = 0.53M costs ~0.5 GFLOP; τ-affine doubles it.
- Per-step cost of applying the update: recombining W + ΔW (plus τ·g₁ for τ-affine) is ~1–2 FLOP per updated weight, ~4–9 MFLOP per step for w16 (~1% of a wheel forward). τ-affine roughly doubles this per-step cost, so it has to earn it.
- Every result below should eventually carry its measured ε.

Analytic estimate for the mode-net head, w16, 256², K = 1000 (unmeasured; FFT/memory constants approximate):

| | per field, once | per sampler step | total over K=1000 | ε |
|---|---|---|---|---|
| plain wheel | – | ≈ 0.58 GFLOP (FFTs 2×84 M, spectral einsums 2×17 M, pointwise 2×101 M, bias irfft 2×42 M, lift 8 M, readout 71 M) | ≈ 577 GFLOP | – |
| mode-net, τ-constant | 1.4 GFLOP (mode-nets 1.35 G) | 0 (W+ΔW fixed per field) | +1.4 GFLOP | **≈ 0.24%** |
| mode-net, τ-affine | 2.5 GFLOP (mode-nets 2.43 G) | +16.8 MFLOP (recombine W+g0+u·g1 over 8.4M entries, 2.9% of a step; memory-bound) | +19.3 GFLOP | **≈ 3.3%** |

At K = 50: τ-constant ε ≈ 4.9%, τ-affine ≈ 11.5%. (A "τ-affine only on the real weights" variant was considered and dropped: the mode-net already updates the time-MLP, so a τ-constant Δθ already reshapes each sample's FiLM(τ); and per the old oracle ablations the spectral and dense parts of Δθ cancel each other, so τ-resolving only the dense part isn't a cheap stand-in for τ-affine.)

## Runs

### A. Infrastructure checks

| id | what | result | verdict |
|---|---|---|---|
| base_w16_scratch | train the w16 wheel **from scratch** in this repo with the old recipe (lr 3e-4, warmup 500, AdamW wd 0.01, clip 0.25, batch 24, 8 τ) | val 0: 1.013, 0.5: 0.844, 1: 0.545, 1.5: 0.491, 2: 0.438 (old: 1.014 / 0.867 / 0.549 / 0.525 / 0.474) | **no data/loss/training bug**: matches or beats the old curve |
| DDP check | best rank-8 config, 1 GPU vs DDP 4 GPUs (batch 6 × 4) | epoch 1: +0.20% both; DDP 128 s/epoch vs ~250 s on 1 GPU | DDP correct, ~2× faster |

### B. Oracle / gate (stopped: don't repeat)

| id | update space | result |
|---|---|---|
| gate w16, bank 12, λ 1, lr 1e-2 / 3e-3 | rank-8 SVD subspace + gain, τ-affine | oracle −0.2% loss, shuffled same (no sample-specific headroom) |
| gate w16, bank 32 | same | −0.2% |
| gate λ 1e-2 / 1e-3 | same | +0.4% / +26% (**worse**): overfits the fixed noise bank |

### C. Direct score training, rank-8 SVD subspace (expressivity-capped per the theory docs)

| id | settings | result |
|---|---|---|
| direct lr 1e-4 / 3e-4 (16 modes) | batch 24, uniform τ | −30% … −101% at 100 steps (diverged) |
| train_w16 16 modes, lr {1e-5, 3e-6} × gain {on, off} | batch 24, uniform τ, τ-affine | epoch 1: −2.56 / −0.60 / −0.15 / **+0.20%** (lr 3e-6, no gain best) |
| all-modes encoder, same 4 | | epoch 1–4: lr 3e-6 no gain +0.20 → +0.41%; lr 1e-5 stuck −0.2…−0.4% |
| DDP best (all modes, lr 3e-6, no gain, rank 8) | 4 GPUs | epoch 1: +0.20, 2: +0.20, 3: +0.33, 4: +0.46, 5: +0.51, 6: +0.55, 7: **+0.64%** (old at 5: +1.2%) |

Takeaway: stable but capped far below the old run. The subspace adds no new operator directions (theory doc §6).

### D. Coarse-16 upsampled spectral updates (+ dense others)

| id | settings | result (half epoch unless noted) |
|---|---|---|
| it1 | constant in τ, lr 1e-4, output scale 1 | −19.7% (diverging) |
| it2 | same, lr 3e-5 / 1e-5 | −7.9% / −2.0%, then lr 1e-5 −8.2% at epoch 1 |
| it3 | constant, output × 0.02, lr 3e-4 | −7.6% |
| it4 | coarse-4 variant (diagnostic) | stopped before any result |
| it5_e2e_lr3e-5 | **τ-affine + low-τ training (τ ∝ u²)**, output × 0.02, warmup 500 | −1.26% (stopped for it6) |
| it5_e2e_lr1e-4 | same, lr 1e-4 | half epoch −0.68%, epoch 1 **−6.86%** (worsening as lr warmed up; stopped) |

Takeaway: τ-affine plus low-τ delays the blow-up but doesn't stop it (epoch 1: −6.9%). Coarse per-output linear heads (≈0.5–1M independent outputs) fail at every lr tried. Dropped in favour of the mode-net head.

### E. Mode-net head (port of the old `full` head)

| id | settings | result |
|---|---|---|
| it6_modenet (GPUs 2,3) | per-mode shared 1×1 MLP on (x̂(k) phase/log-mag, k, z) → full (c_in × c_out) per mode, dense heads for the real weights, delta 0.02, τ-constant, low-τ training, lr 3e-4 warmup 500 cosine → 3e-5 (12.5k steps), batch 24 | epoch 0.5: −4.04, 1: −1.86, 1.5: −0.22, 2: +0.49, 2.5: +0.52, 3: +0.97, 3.5: +0.94, 4: +1.67, 4.5: +1.54, 5: +1.81, 5.5: +1.81, 6: **+2.08**, 6.5: +1.97, 7: +1.40, 7.5: +1.89, 8: +1.93, 8.5: −0.74, 9: +1.06, 9.5: −1.12, 10: −0.20, 10.5: −5.02, 11: −4.56% (degraded at peak lr ≈ 2.4e-4; lr too high for this loss, see Standing rules) |
| it6_modenet_utau (GPUs 0,1) | same, **uniform τ** (exact old recipe): ablation of low-τ allocation | epoch 0.5: −5.08, 1: −2.12, 1.5: +0.26, 2: +0.56, 2.5: +0.47, 3: +1.18, 3.5: +1.13, 4: **+1.65%** (stopped at epoch 4 for it7) |
| it7_modenet_affine (GPUs 0,1) | it6 (low τ) + **τ-affine** update (mode-net emits g0 and g1): does τ-affine earn its FLOPs? Hypernet 25.5M params (vs 22.4M), ~10% slower per step | epoch 0.5: −11.03, 1: −5.95, 1.5: −1.91, 2: +0.32, 2.5: +1.10, 3: +0.87, 3.5: +1.41, 4: +2.14, 4.5: **+2.18%** (stopped at epoch 4.5). Verdict for now: +0.1–0.5 points over τ-constant at the same epoch for ~14× the overhead (ε 3.3% vs 0.24%). Not clearly worth it; revisit after capacity scaling, and decide with a per-τ-band margin breakdown (diffusion_specifics.md §3) |
| it8_modenet_depth8 (GPUs 0,1) | it6 config with **trunk depth 4 → 8** (capacity probe: depth is the strongest measured axis, "monotone, gap-shrinking"; +~2 MFLOP/field) + train-vs-val gap tracking (`train_eval_n=512`) | running |

Low-τ vs uniform τ with the mode-net head: epoch 4 +1.67% vs +1.65%, **no measurable difference** (and none at inference either; it only changes training). Both beat the old run at epochs 3–4 (old: −1.7% at 3, +1.2% at 5).

### Comparison with the old run (margin vs own base)

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 7.5 | 14 |
|---|---|---|---|---|---|---|---|---|---|
| old D2_hyper_full | −0.48 | +0.54 | −1.67 | +0.41 | +1.23 | −0.27 | +0.18 | +0.03 | +2.75 |
| it6 mode-net port (low τ) | −1.86 | +0.49 | +0.97 | +1.67 | +1.81 | +2.08 | +1.40 | +1.89 | |

The port is ahead of the old run at every check from epoch 2.5, and much steadier (old swung to −8.5% at 5.5). Caveats: different val frames (old: stride split inside 10000–20000; here 20001–20513, unseen by the wheel), and the global trunk here is an MLP on all Fourier modes rather than the old dilated CNN.

### F. Honest validation (random points, random τ), lr 1e-4

| id | settings | result |
|---|---|---|
| it9_d4 (job 58844824) | mode-net port, trunk depth 4, τ-constant, low-τ training, lr 1e-4 peak (warmup 500) cosine → 1e-5 over 8000 steps (~19 ep), batch 24, train-vs-val gap tracked | val base **0.252624**. Epoch 1: val −1.53%, train set −1.51% (stopped: moved to a new 4 h job) |
| it9_d8 (job 58844824) | same, **trunk depth 8** (capacity probe, paired with it9_d4) | epoch 1: val **−0.40%**, train set −0.39% (stopped: moved to a new job) |
| it10_d4 (job 58849507, ends 23:38) | it9_d4 rerun, 12000 steps (~29 ep); checkpoint every check | epoch 1: val cond 0.25648 (−1.53%), train 0.25570 (−1.51%). Epoch 4: val **0.25026 (+0.94%)**, train-set 0.24968 vs base 0.25189 (+0.88%). Epoch 8: val 0.25066 (+0.78%), train 0.24977 (+0.84%); plateaued, gap ≈ 0. Stopped at epoch 8 |
| it10_d8 | it9_d8 rerun (depth 8) | epoch 1: val 0.25364 (−0.40%), train 0.25287 (−0.39%). Epoch 4: val 0.25038 (+0.89%), train 0.24965 (+0.89%). **Depth 8 = depth 4 on train and val at epoch 4; no train-val gap in either.** Not data-limited, and trunk depth isn't the binding axis. Stopped at epoch 4 |
| it11_mh128 (GPUs 0,1) | it10_d4 with **mode-net hidden 64 → 128** (spectral-head capacity probe; the mode-net emits all 8.4M spectral updates). Hypernet 22.58M params; mode-net ≈ 3.0 GFLOP/field (ε ≈ 0.5% at K=1000, vs 0.24%) | epoch 4: val **0.24918 (+1.36%)**, train-set 0.24862 (+1.30%) vs hidden 64: 0.25026 / 0.24968. **Moves train and val together, no gap → the spectral head is the binding axis**. Epoch 8: val **0.24800 (+1.83%)**, train 0.24726 (+1.84%) (best). Then drift: 9: 0.24962, 10: 0.25368, 11: 0.25708, 12: 0.25930 (−2.64%), while the training loss stayed flat at ≈0.47. Stopped at epoch 12 |
| it12_mh256 (GPUs 2,3) | hidden 128 → **256** (next doubling on the binding axis; ≈ 7.0 GFLOP/field, ε ≈ 1.2%) | epoch 4: val **0.24877 (+1.52%)**, train 0.24820 (+1.46%): diminishing returns (64→128: +0.42 pt, 128→256: +0.16 pt). Epoch 8: val 0.26124 (**−3.41%**), collapsed like it11. Stopped |

**Finding: low-τ training allocation (τ ∝ u²) is misaligned with the uniform-τ validation and causes the late collapse.** Banded eval of it11 (128 val points, 8 draws per band, base → cond):

| τ band | epoch 8 (best) | epoch 12 (collapsed) |
|---|---|---|
| < 0.5 | 0.4577 → 0.4503 (−1.6%) | 0.4577 → 0.4696 (+2.6%) |
| 0.5–0.8 | 0.01901 → 0.01869 (−1.7%) | 0.01901 → 0.02209 (+16%) |
| 0.8–1 | 0.00181 → 0.00162 (−10%) | 0.00181 → 0.00616 (+3.4×) |

High τ degrades most (it is barely sampled by τ ∝ u²), but the < 0.5 band also degrades (probably its under-sampled 0.25–0.5 half), and that band dominates the mean. The training loss (fresh noise, low-τ weighted) is flat and blind to it. The same allocation likely caused the lr 3e-4 collapse (it6). **Rule: train with the τ distribution the validation uses (uniform).** diffusion_specifics.md §2's low-τ advice doesn't transfer to this validation.

| it13_mh128_utau (GPUs 0,1) | it11 with **uniform-τ training** | val by epoch: +0.12, +0.53, +0.70, +1.24, +1.46, **+1.68 (0.24838)**, +0.25, +1.30% (0.24934) (running) |
| it13_mh256_utau (GPUs 2,3) | it12 with **uniform-τ training** | +(−3.04), +0.68, +1.14, +1.31, **+1.64 (0.24848)**, +1.42, +1.57, **−3.61%** (0.26173; train loss spiked to 0.276). Stopped |

**Correction:** uniform τ doesn't remove the instability. With uniform τ the training loss tracks validation, and **both spike together** (it13_mh256 epoch 8: train 0.276, val −3.6%). Low-τ allocation made it worse and hid it from the training loss, but the root problem is optimization spikes at lr ≈ 1e-4. Train and val always move together (no generalization gap), so it isn't memorization.

Next, from the old repo's best run (`D2_w32_hyper_full_attn_wd`: hyper wd 0.01, mode-net wd 0.1, `freeze_base=0` with wheel lr 3e-5 → 3e-6, global batch 4): weight decay, then the joint finisher.

| it13_mh128_utau (cont.) | | epoch 12: val 0.25453 (**−0.76%**), train 0.25364: the same late decline as the low-τ run. Stopped |
| it14_mh128_wd (GPUs 2,3) | it13_mh128_utau + **AdamW weight decay 0.01 (trunk) / 0.1 (mode-nets)** | by epoch: −0.27, +0.46, +0.93, **+1.28 (0.24938)**, +1.20, +1.07, +0.39, +1.25, …, epoch 12: 0.25607 (**−1.36%**). **Weight decay does not prevent the swings or the late decline** (negative). Stopped |

### G. Joint two-timescale finisher (theory §7f)

| id | settings | result (val cond; base 0.252624) |
|---|---|---|
| it15_joint (GPUs 0,1) | from it13_mh128_utau best (epoch 6, 0.24839). Wheel unfrozen: hyper lr 3e-5, **wheel lr 3e-6** (ratio 10), relative anchor weight 1, uniform τ, batch 24, 6000 steps | epoch 0: 0.24839 (+1.68%); 1: 0.24662 (+2.38%); 2: 0.24682; 3: 0.24624 (+2.53%); 4: 0.24614 (+2.57%); 5: **0.24529 (+2.90%)**; 6: 0.24542; 7: 0.24604 (+2.60%). Wheel standalone 0.2518 → 0.2526 (barely moves); wheel displacement 0.15% → 0.49%; anchor drift ~1e-4 relative. **No spikes.** Continued: 8: 0.24512 (+2.97%); 9: 0.24592; 10: 0.24495 (+3.04%); 11: 0.24421; 12: **0.24418 (+3.34%, best)**; 13: 0.24445; 14.4: 0.24446. Wheel standalone drifted 0.2518 → 0.2541 (worse alone, while the system improves); displacement 0.82%. Zero inference-FLOP cost. Finished |
| it16_joint_r3 (GPUs 2,3) | same, **wheel lr 1e-5** (ratio 3): is the wheel engaged enough? | 1: 0.24647 (+2.44%); 2: 0.24599; 3: 0.24533 (+2.88%); 4: 0.24552; 5: 0.24543; 6: 0.24534; 7: **0.24426 (+3.31%)**. Ahead of ratio 10 at matched epochs (ratio 10 at 7: 0.24604); displacement ~2× (1.06% at 7). 11: 0.24423; 12: 0.24300; 13: 0.24344; 14: **0.24277 (+3.90%, best overall)**; final (14.4): 0.24314 (+3.75%). Wheel standalone 0.2526 → **0.2594** (2.7% worse alone; ratio 10: 0.2541); displacement 1.87%. The faster wheel specializes the wheel to the hypernet. Finished |
| it17_joint_cont (GPUs 0,1) | **continue it15 from its best (joint epoch 12, 0.24418)**, same settings (ratio 10), 5500 more steps (~13 epochs); anchor reset to the epoch-12 predictions. User's choice | +1: 0.24426; +2: 0.24392; +3: **0.24379 (+3.50%, best)**; +4: 0.24434; +5: 0.24443; +6: 0.24381; +7: 0.24535. Wheel standalone kept drifting (0.2540 → 0.2563). Plateaued at about +3.4–3.5% at ratio 10. **Ended at step 2912: interactive job 58849507 was cancelled at 22:48:54 from outside this session** |

(it6 stopped at epoch 11 after degrading; it8 depth-8 probe stopped at epoch 1 (val −0.31%, train set −0.10%) to rerun on the honest validation.)

## Answer so far (2026-09-24): did the proposed changes improve on plain score-loss training?

**No, not meaningfully.** Per change:
- Oracle targets / gate / EM distillation (the framework's core method): **worse**. Never beat the frozen wheel (solves fit the fixed noise bank).
- Update-space rules (rank-8 SVD subspace, gains, excluding spectral-bias fields): worse or neutral. The subspace capped at about +0.6%; the old full-tensor mode-net head was needed.
- τ-affine updates: inconclusive (+0.1–0.5 pt, within noise) at ~14× the FLOP overhead (ε 3.3% vs 0.24%).
- Low-τ training allocation (τ ∝ u²): **harmful** under uniform-τ validation (late collapses, hidden by the training loss).
- Trunk depth 4 → 8, weight decay: no effect.
- **Joint two-timescale finisher: the one positive.** Stable, took the best from +1.68% to +3.34% (val 0.24418) at zero inference-FLOP cost, but not shown to beat the old run at matched training length.

Caveats: (1) **capacity confound.** The old head's mode-net hidden width was **512** (old defaults: `hyper_head_ratio 1` × z 512, ≈ 18 GFLOP/field, ε ≈ 3%); ours is 128 (≈ 3 GFLOP, ε ≈ 0.5%), and width was measured to be the binding axis. (2) Training length: ours ≤ ~17 epochs of stable training, against ~200 for the old best. Settling it needs a long, capacity-matched run of both recipes on the same validation and seed.

Old `D2_hyper_full` vs our best, other differences: dilated-CNN trunk (z 512) vs Fourier-MLP trunk (66,048 → 256 → 512 × 4 blocks); Xavier-initialized vs zero-initialized output; spectral-bias fields updated vs excluded; lr 3e-4 → 3e-5 vs 1e-4 → 1e-5; weight decay 0.01 vs 0; training frames 95% of 10000–20000 (stride val removed) vs all of them.

## Open questions / next tests
- **Per-τ-band margin breakdown** for τ-constant (does the hypernet hurt any band? if high-τ bands are hurt, τ resolution has something to buy; if not, τ-affine's 3.3% overhead is wasted).
- Measure ε (flops diagnostic) for each surviving configuration.
- **Three-gap test** (capability_diagnostics.md §2): `direct.train_eval_n=512` now scores 512 evenly spaced train pairs on the eval bank at every check. Train ≈ val margin → expressivity-capped → capacity probe (trunk depth first, then encoder bandwidth, then width). Train ≫ val → data-limited/memorizing → weight decay / data. Decide scaling from this, not from sweeps.

### H. Frame history for the hypernet (hypernet-only; the wheel still sees only omega_t)

Plumbing (2026-09-24): `task_kwargs.history=L` feeds [omega_t, omega_{t-1}, ..., omega_{t-L}] to the hypernet. Mode-nets get per-mode, per-lag log-magnitude ratio and phase difference (cos, sin) of x_hat_t vs x_hat_{t-l}; the trunk encoder adds each successive increment with its own projection (`history_trunk=false` = mode-nets only; the trunk projection is +17M params per lag, ~34 MFLOP/field per lag). Val/test points and (x, y) pairs are identical to history=0 (checked by `tests/check_history.py`); train loses its first L pairs. `history_noise` (train only) is there for rollout exposure bias; default 0.

**Information check** (`experiments/history_info.py`, CPU, 47 s): per-mode complex least squares for y_t = omega_{t+1} - omega_t, fit on 2000 train anchors, scored on 1000 val anchors. "phys" adds the advection term u.grad(omega) computed from omega_t. Val relative MSE (1 = no skill):

| model | all | \|k\| 0-8 | 8-32 | 32-64 | >= 64 |
|---|---|---|---|---|---|
| share of y energy | 1 | 0.183 | 0.353 | 0.282 | 0.183 |
| markov-lin (omega_t) | 0.515 | 0.294 | 0.624 | 0.534 | 0.494 |
| markov-phys (omega_t, advection) | 0.207 | 0.0031 | 0.097 | 0.309 | 0.463 |
| markov-phys + lag1 | 0.179 | 0.0002 | 0.047 | 0.276 | 0.462 |
| markov-phys + lag2 | 0.172 | 0.0001 | 0.034 | 0.269 | 0.462 |
| persistence (y_t ~ y_{t-1}) | 1.555 | 0.062 | 0.870 | 2.455 | 2.980 |

Reading: beyond the physics-informed snapshot model, one lag cuts the error 13% overall, halves it at |k| 8-32, -11% at 32-64, and does **nothing at |k| >= 64** (increments there decorrelate within one frame: persistence 3x worse than predicting zero). So history carries its signal only in |k| < 64, exactly the modes the wheel's spectral weights and the mode-nets cover. Not yet separated: whether the lag gain is non-Markov memory, or just a second-order-in-time term that a nonlinear function of omega_t could compute (multistep effect). Next: add the second-order Markov term (tendency of the advection term) as a feature; if it closes the gap, the value is function simplicity, not information.

| id | settings | result (val cond; base 0.252624) |
|---|---|---|
| it17_hist1 (job 58854859, 4 GPUs, m5326, ends 00:29) | it13_mh128_utau recipe + **history=1** (mode-net history features + trunk increment projection; hypernet 39.6M params vs 22.6M, ε ≈ unchanged ~0.5%). Fresh init; batch 6/GPU x 4 = 24 global (same as it13's 12 x 2); lr 1e-4 warmup 500 cosine → 1e-5, 12000 steps; uniform τ | val by epoch: −3.11, −0.88, +0.88, +0.19, **+1.08**, +0.75% (it13: +0.12, +0.53, +0.70, +1.24, +1.46, +1.68%). Train-set tracks val (no gap). Epoch 7: +0.82%. Behind it13 from epoch 4 on and noisier: per-mode (diagonal in k) history does not help. Stopped at epoch 7 (user) for it18 |
| it18_attn_hist1 (job 58854859, 4 GPUs) | it17 + **axial mode attention** in every mode-net (`mode_attn_layers=2`, 8 heads, after the stem): row + column attention over the (kx, ky) grid, so history features can couple across modes (it17's per-mode history is diagonal in k). Same recipe, batch 6/GPU x 4 | val by epoch: −8.90, +0.37, **+1.40** (best; it13 +0.70, it17 +0.88), then **−95.3**, −10.2, −12.2% (train-set identical: optimization, not overfitting). Then −2.60, −2.66, −1.19, −0.37, −1.95, −0.57, −1.05, −3.82, −10.52% (epochs 10–18); never crossed zero again. Stopped at epoch 18 (user) |

**it18 blow-up diagnosis** (CPU, checkpoints at steps 832–2496, 6 val fields):
- The attention *weights* barely moved (2–7% per epoch; per-head q/k gain ≤ 0.5). Not a weight explosion.
- But the attention branch is a big amplifier from init: pre-GroupNorm + xavier q/k/v/proj on small stem activations (input rms 0.16–0.4) gives branch output / residual input = 4–17× at epoch 2 and up to **34×** by epoch 5 in the `spec_convs.1.weights2` mode-net (logits up to 83 from outlier modes, entropy 0.80). The "residual" block is effectively a replacement.
- The blow-up tracks **one tensor's relative update**, ‖ΔW‖/‖W‖ for `spec_convs.1.weights2` (layer 2, negative-kx half): it18 0.55 → 1.28 (epoch 3, best) → **2.20 (epoch 4, blow-up)** → 2.42 → 2.38. it13 on the same tensor: 0.23, 0.43, 0.62, 0.85, 1.13 (epochs 2–6), i.e. it18 grows it ~2.5× faster. Every other tensor stays ≤ 0.5.
- ‖ΔW‖ is nearly the same for every field (max ≈ 1.04 × mean in both runs): the dominant part of the update may be a shared correction rather than per-sample conditioning (norms only; direction similarity not yet measured).
- Likely mechanism (hypothesis): the high-gain attention branch speeds up growth of that mode-net's output layer, and the update overshoots once it exceeds ~2× the base weight. The same tensor dominating it13 (1.1× by epoch 6) may also explain the section F late declines at lr 1e-4.
- **Later (epochs 9–15): the update keeps growing while val recovers to ≈ −0.5%.** ‖ΔW‖/‖W‖ for `spec_convs.1.weights2`: 4.29 (epoch 9), 6.66 (12), 7.56 (14), **9.26 (15)**; the other spectral tensors 1.2–1.6×. Still near-identical across fields (max ≈ 1.03 × mean). The recovery is not the update shrinking back: the hypernet now emits a mostly shared ΔW several times larger than the pretrained weights, i.e. it is re-fitting the wheel's layer-2 spectral weights, not perturbing them.
- Fix candidates: zero-init (or layer-scale ~1e-2) the attention output projection so each block starts as an identity; a per-tensor trust region on ‖ΔW‖/‖W‖; measure field-to-field cosine of ΔW to split the shared part from the conditioning part.


**The update grows without limit in every direct run, and it is the same for every field** (CPU, 6 val fields; rel = ‖ΔW‖/‖W‖ with W the checkpoint's own wheel; cos = mean pairwise cosine of ΔW across fields, for `spec_convs.1.weights2`):

| run | epoch 1 | 3 | 5 | 6 | 8 | 10 | 12 | cos |
|---|---|---|---|---|---|---|---|---|
| it13_mh128_utau rel | 0.08 | 0.43 | 0.85 | 1.13 (best +1.68%) | 1.51 | 2.08 | 2.49 (−0.76%) | 0.998–0.999 |
| it11_mh128 (low τ) rel | 0.05 | 0.36 | 0.64 | 0.90 | 1.25 (best +1.83%) | 1.70 | 2.08 (−2.64%) | 0.998–0.999 |
| it15_joint rel (from it13 epoch 6) | 1.13 | 1.13 | 1.13 | | | | 1.13 (+3.2%) | 0.999 |
| it18_attn_hist1 rel | 0.15 | 1.28 (best +1.40%) | | 2.38 | | | 6.66 → 9.26 at epoch 15 | 0.996–0.998 |

- In it11 and it13 the margin peaks when this tensor's rel ≈ 1.1–1.3, then declines as rel keeps growing (2.1–2.5 by epoch 12): the late declines in section F coincide with the update running past ~1.5× the base weight.
- **cos ≈ 0.999 everywhere: ΔW is essentially identical across fields.** The field-dependent part is ≈ √(1 − 0.999) ≈ 4% of ‖ΔW‖. Most of every "margin" so far is therefore a *shared* weight correction, i.e. fine-tuning the wheel through the hypernet's output, not per-sample conditioning.
- it15_joint keeps ΔW frozen at 1.13 (anchor) while the wheel moves, and is the only run that improved steadily.
- **Missing control:** fine-tune the wheel alone (no hypernet) with the same budget on the same loss. If it reaches ~+1.7%, the direct-stage margins are not evidence for conditioning; the per-sample part must then be measured separately (e.g. margin of ΔW(x) vs the field-mean ΔW̄).

### I. Is any of it conditioning? (2026-09-24, job 58854859)

**Conditioning split** (`experiments/conditioning_split.py`, GPU; fixed val set; ΔW̄ = hypernet update averaged over 512 evenly spaced train fields, applied identically to every field; shuffled = another field's ΔW):

| checkpoint | wheel alone | wheel + ΔW̄ (shared) | wheel + ΔW(x) (reported) | wheel + shuffled ΔW | conditioning benefit (ΔW̄ − ΔW(x)) |
|---|---|---|---|---|---|
| it13_mh128_utau best (epoch 6) | 0.252624 | **0.248212 (+1.75%)** | 0.248387 (+1.68%) | 0.248428 (+1.66%) | **−0.07%** |
| it15_joint best | 0.253676 (co-trained wheel; −0.42% vs original) | **0.243732 (+3.52% vs original)** | 0.244180 (+3.34%) | 0.244245 (+3.32%) | **−0.18%** |

Per tensor (both checkpoints nearly identical): ‖ΔW‖/‖W‖ = 0.10 / 0.18 / 0.46 / 1.12 for spec_convs.{0.w1, 0.w2, 1.w1, 1.w2}; cosine across fields 0.995–0.9995; per-field part ‖ΔW − ΔW̄‖/‖ΔW‖ = 10% / 7% / 4% / 4%.

**Verdict: none of the margins so far is per-sample conditioning.** Every gain is a shared weight change; averaging the hypernet's output over fields beats its per-field output, and another field's update does almost as well. The core assumption that failed: that the pretrained wheel sits at its shared optimum for this loss, so a hypernet could only add per-sample structure. It does not (the shared correction also improves train frames equally), and the zero-init output heads (biases, coordinate-driven mode-net outputs) give the hypernet a direct shared channel, which absorbs the whole gradient.

**Control: fine-tune the wheel alone** (it19_base_ft, no hypernet; `base` stage from the pretrained w16, uniform τ, lr 3e-5 constant after 200 warmup, AdamW wd 0.01, clip 0.25, batch 8/GPU x 3 = 24, 8 τ per sample, 6240 steps, 3 GPUs, ~63 s/epoch; checkpoint `model_checkpoints/it19_base_ft/after_base.pt`):

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| val margin | +1.24 | +1.50 | +1.74 | +1.93 | +2.07 | +2.22 | +2.50 | +2.75 | +2.95 | +3.14 | **+3.24% (0.244447)** |

Still improving ~0.1 pt/epoch at the end (not converged). Plain fine-tuning passes it13's best (+1.68%) at epoch 3 and reaches it15's reported +3.34% territory in 16 minutes with zero inference overhead. **The pretrained wheel was far from converged on this loss; every hypernet margin measured so far is explained by that.**

Next: (1) converge the base (continue it19 until val plateaus) and use it as the wheel for all hypernet work; (2) make the hypernet output zero-mean across the batch (shared part must go into the wheel) plus a per-tensor trust region on ‖ΔW‖/‖W‖; (3) report the conditioning split (ΔW(x) vs ΔW̄ vs shuffled) for every run.
