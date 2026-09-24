# Diffusion-Specific Techniques for Hypernetwork-Conditioned Denoisers

*Companion to the general landscape document. Everything here is specific to a diffusion/score-based wheel (noise level τ, denoising objective, iterative sampler). Techniques that transfer to arbitrary iterated solvers — including the generalized "resolve updates in inference-observed variables" principle — live in the general document; this file contains only their diffusion instantiations and diffusion-only findings.*

---

## 1. τ-dependent weight updates (the diffusion instantiation of the biggest lever)

Parameterize the update as an explicit function of noise level: **ΔW(τ) = g₀ + τ·g₁** (the "A·τ + B" form), fused per sampler step (elementwise combine, then apply — a few percent of one wheel evaluation). Rationale specific to diffusion: τ is fully observed at every sampler step, so τ-resolution costs zero generalization budget, while a τ-static update forces every oracle solve into a single compromise across noise levels — and the conditioning headroom is heavily concentrated at low τ (measured: base loss ~10× higher in the low-τ band than the high-τ band), so the compromise is expensive. Measured effects: deeper oracle solves at identical protocol, a *smoother* target map at greater depth, and held-out below the τ-static floor that four independent method stacks had converged to. Oracle solves, distillation targets, and the hypernetwork head all operate on the concatenated (g₀‖g₁) vector; the EM loop runs unchanged in this doubled space. Literature family for extensions: TimeLoRA (multi-basis time-interpolated composition, if linear-in-τ saturates), T-LoRA (allocate update capacity by timestep — more at low noise), TC-LoRA (hypernetwork generating per-step adapters), eDiff-I (see §3).

## 2. τ-allocation of training compute — with a critical asymmetry

**Training-side (validated, large):** concentrate the τ-sampling distribution of any *shared-parameter* score training at low noise levels (e.g., sample u uniform, use τ ∝ u²). This rehabilitated end-to-end hypernetwork training from near-useless (−1% vs base) to competitive with distillation (−8.7%) at identical budget, and applies to base training and to the joint two-timescale finisher. Mechanism: uniform-τ training wastes most gradients on high-τ where the base is already near-perfect, and the per-τ gradient conflict on shared weights is a known multi-task pathology. Note where it does **not** apply: the EM loop's per-sample oracle solves have no cross-sample gradient conflict — keep their noise banks τ-uniform/stratified; low-τ allocation there is unnecessary and untested.

**Min-SNR warning:** Min-SNR-γ loss weighting, standard in image diffusion, *inverts* for this task class — for ε-prediction it downweights exactly the low-τ region where the conditioning headroom lives. Measured: worse than uniform. The multi-task-conflict theory behind it is correct; its weighting direction is tuned for image perceptual quality. Reallocate sampling instead of reweighting loss (allocation > reweighting also per the noise-schedule literature), in the low-τ direction.

**Sampler-side (the asymmetry):** concentrating the *sampler's* τ-grid at low noise **hurts** a from-pure-noise sampler at small step counts (measured: +40–70% sampled MSE) — coarse structure still needs high-τ coverage. The correct way to spend low-τ capacity at inference is the PDE-Refiner-style architecture: a deterministic one-shot prediction followed by a few low-noise refinement steps, not a reallocated from-noise DDIM grid. The refinement literature also supplies two free extras worth keeping: the denoising objective acts as spectral data augmentation (forces attention onto low-amplitude/high-frequency components that plain regression neglects — the underlying reason a diffusion wheel is a strong PDE surrogate), and the diffusion connection yields calibrated uncertainty usable for the rollout confidence gate. Stability note at very low τ: Lipschitz singularities of the score near τ→0 are a named pathology (E-TSDM) — if low-τ fine-tuning is unstable, timestep-sharing/smoothing near zero is the standard mitigation.

## 3. τ-band expert bases (eDiff-I pattern)

Split the base denoiser into specialists per noise interval (train shared, then branch and fine-tune per band). Zero added inference FLOPs (weight selection by τ; extra memory only). Measured: +5.5% combined at two bands, with the gain concentrated at high τ — which is itself diagnostic: large high-τ gains mean the shared base was capacity-starved there, while low-τ barely moving means low τ is *conditioning*-limited, i.e., the low-τ headroom belongs to the hypernetwork, not to base capacity. Run the banded loss breakdown on any new system before deciding where to spend capacity. Composes with the full stack in principle (run the target/EM pipeline per band, or one hypernetwork trunk with per-band heads); the composition is unmeasured — first thing to test at scale.

## 4. Noise handling in target generation and evaluation

- **Common-random-number noise banks** for oracle solves: a fixed, stratified set of (τ, ε) pairs shared across all solves, so two solves differ only through their conditions. This is the diffusion instantiation of the CRN requirement in the general protocol and is non-optional — resampled noise reintroduces selection scatter.
- **Fixed evaluation banks** (separate from solve banks) for all reported scores, so numbers are comparable across the entire program.
- Keep solve banks τ-stratified (see §2's note); size ~12 pairs sufficed for consistent selection in measurements.

## 5. Guidance: measured negatives and the one diagnostic worth keeping

Both guidance forms are monotonically harmful on a properly trained stack: weight-space scaling s·ΔW (free) and prediction-space extrapolation ε_base + s(ε_conditioned − ε_base) (costs a second wheel evaluation per step). The distilled/EM amplitudes are already calibrated — there is no under-expressed conditioning to amplify. Keep the five-minute s-sweep as a diagnostic on new systems: it fires only if hypernetwork training used heavy shrinkage or averaging. If it ever fires, use guidance *distillation* (one training pass baking s\* into the update) — never pay the 2× wheel evaluations at inference; a doubled wheel budget is better spent on the wheel itself, and any guidance-based configuration must beat that reallocation to justify itself.

## 6. Sampler-facing evaluation notes

- Score loss (denoising MSE on the evaluation bank) tracks conditioning quality well and is cheap, but sampled-rollout MSE is the deliverable; the two can diverge (a τ-band-imbalanced model can score well and sample badly). Report both at decision points.
- With few sampler steps, clip the predicted x₀ inside the DDIM update (numerical guard for a weak wheel; harmless for a strong one).
- The hypernetwork fires once per *forecast step* and its g(τ) fusion runs per *sampler step*; at K sampler steps the hypernetwork's cost amortizes K-fold, which is what keeps the whole stack inside the inference-FLOP budget.
