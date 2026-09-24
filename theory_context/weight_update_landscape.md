# Conditioning Iterated Solvers via Per-Sample Weight Updates: The Landscape Problem and the Method Stack

*Method-agnostic reference. Applies to any "wheel" network (diffusion denoiser, implicit/fixed-point solver, supersampler) conditioned by a hypernetwork that emits per-sample weight updates, fired once per solve. The goal throughout: maximum accuracy per inference FLOP — every technique here is training-time; inference cost never grows beyond one hypernetwork forward per solve plus negligible per-step update application.*

---

## 1. The setting

Two networks. The **wheel** is small and runs iteratively (K steps per solve). The **hypernetwork** H runs once per solve: it reads the conditioning input x and emits an update ΔW to the wheel's weights, so each problem instance runs a cheap, input-specific perturbation of a shared base model. Per-sample **oracle solves** — directly optimizing ΔW for one sample — reach far lower loss than the base, proving large headroom exists in weight space. The project is to make H capture that headroom.

## 2. The core difficulty: the oracle map looks fractal

Naively generated oracle targets form isolated "wells": two nearly identical conditions produce distant solutions with large loss barriers on the line between them, and the map x → ΔW* appears discontinuous. A hypernetwork asked to regress this map memorizes training points and fails to interpolate. The apparent fractality decomposes into **three distinct layers**, each with its own fix; diagnosing which layer dominates is the first task on any new system.

**Layer 1 — selection noise (largest, entirely removable).** Each sample's near-optimal set is a huge degenerate region. An optimizer run to convergence acts as a *discontinuous selector* on that set: tiny changes in condition, initialization, or stochasticity select far-apart solutions. Decisive test: solve the **same** sample twice from jittered initializations ("twin test") — if the two solutions are distant with a barrier between them, the scatter is selection, not signal. Selection noise *sharpens with convergence*: the harder you optimize, the deeper and more separated the wells. The smooth branch of the solution manifold exists; unconstrained solvers simply don't select it.

**Layer 2 — multiplicity / gauge freedom (removable by design).** Parameter symmetries (permutations, rescalings, rank-factorization gauge) and redundant pathways (anything the update can express that another input path also expresses) mean many parameter vectors implement the same function. Redundancy grows when the condition also enters the wheel through activations, because the update then only needs to carry residual information. Fixes: gauge-light update parameterizations (Section 6) and function-space rather than parameter-space comparisons where multiplicity persists.

**Layer 3 — genuine roughness (irreducible, but quantifiable).** After Layers 1–2 are controlled, real target variation with condition distance remains, and it grows steeply when the underlying dynamics are chaotic at the prediction lead. This layer is not noise; it is information, and it sets the data-density requirement (Section 5) and the ultimate ceiling (see the companion diagnostics document).

## 3. Target-generation protocol (consistency over depth)

The protocol that produces *learnable* targets:

- **Anchored proximal solves**: minimize task loss + λ‖ΔW − anchor‖², anchor = 0 (the base) initially. The anchor is the selection mechanism: it makes the solve a nearly deterministic, continuous function of the condition.
- **Fixed evaluation noise/data banks** (common random numbers) for any stochastic inner objective, so two solves differ only through their conditions.
- **Bounded budget**: short solves, identical step counts. Depth is deliberately sacrificed for consistency; depth is recovered later by the EM loop, not by longer solves.
- **Warning — chained warm-starting drifts**: warm-starting each solve from its neighbor's solution looks like a cheap consistency device but accumulates null-space drift (good inner-objective loss, broken actual behavior). If chaining is used, anchor every solve to the base and validate each link on held-out noise/data. Fresh anchored solves are safer.

Targets generated this way are *shallow but mutually consistent*; the entire method stack exists to deepen them without losing that consistency.

## 4. The depth–smoothness law and the two-alarm dashboard

Anneal target depth in rungs (progressively weaker anchors, warm-started per rung) and at each rung measure: target depth (median task loss), **neighbor-target distance** (relative distance between updates of adjacent/similar conditions), and hypernetwork train/held-out after distillation. The empirical law:

- Target smoothness decays monotonically with depth.
- Held-out performance is flat until neighbor-target distance crosses a threshold (≈1.0 in relative units on the systems measured — **measure your own**), then degrades sharply; the train–test gap opens at the same point.
- The threshold is invariant across bases and update spaces; the *depth at which it is crossed* is what the levers below move.

**Stopping rule**: push target precision exactly until neighboring targets stop resembling each other at your data's similarity scale. Past that point, additional per-sample precision converts into held-out damage. The two alarms — neighbor-distance ≈ threshold, and gap opening — should be logged in every training run.

## 5. The density law (the most operationally important result)

Learnability is governed by the **spacing of targets in condition space**, not their count. The instrument is the **decorrelation curve**: solve anchored targets along a trajectory (or ordered condition set) and plot target similarity vs condition similarity. It has a bend: above a condition-correlation threshold (~0.75–0.85 measured; re-measure per system and per prediction lead), neighboring targets share large common structure; below it they are near-orthogonal and no smooth map exists to learn.

Consequences, all verified by controlled A/B (same target count, spacing halved: held-out flipped from a net loss to a large gain):

- **Spacing, not count, is the budget constraint.** N targets spread too thin fail outright; the same N packed at valid spacing succeed.
- Under a fixed solve budget, cover **dense patches** at valid spacing and leave the rest of condition space to auxiliary losses (semi-supervised: regression on the targeted subset + task loss on untargeted samples), never uniform thin coverage.
- The spacing requirement steepens as target depth grows and as the prediction lead grows (chaotic sensitivity). Re-measure the bend at your actual lead before budgeting data generation.
- Prediction lead and condition spacing are independent knobs when trajectories are saved densely: pairs (x_t → x_{t+L}) can share a large lead L while their conditions x_t sit one dense save apart.

## 6. Update-space design

Which parts of the wheel the hypernetwork may modify is a first-class design decision. Rules with measured support:

- **Forbid activation-additive channels** (bias fields, constant injections into activations). These are the memorization channel: a per-sample solve can write the answer directly into them, and that component is exactly what varies roughly across conditions. Removing them cost zero depth in expressivity gates.
- **Allow operator modifications**: multiplicative gains on existing transformations *plus* additive terms in operator space (new directions in the transformations themselves). Purely multiplicative spaces (gains only, or shifts of singular values within frozen singular vectors) are expressivity-capped — measured three independent ways; the payload needs new operator directions. Mixing within a small top singular subspace recovers part but not all of the depth; sweep the subspace rank if hypernetwork-head FLOPs matter.
- **Identity at zero**: parameterize so update = 0 reproduces the base exactly, and zero-initialize the hypernetwork's output head. The system then starts at exactly the trained base and departs only through learned conditioning.
- **Resolve the update in every variable observed at inference.** Any variable the wheel's iteration exposes and the deployment observes — iteration index, solver stage, noise level, resolution/scale factor — costs **zero generalization budget**, because it never has to be interpolated from sparse data. Making the update an explicit (e.g., affine) function of such a variable was the single largest lever measured: it broke a performance floor that four independent method stacks had converged to, because a static update forces the oracle into a compromise across stages, and releasing that compromise is free. The generalization budget is spent only on smoothness in x.
- **Intrinsic rank**: targets have an effective dimensionality; output heads below it collapse (cannot fit train at all), and it grows with target depth. Locate the collapse rank on your targets before committing to low-rank heads.

## 7. The training pipeline

**(a) Base training.** Train the wheel normally, allocating training effort toward the regimes where conditioning headroom lives (see the method-specific documents for what "allocation" means per wheel type). If capacity per regime helps, specialist copies of the base swapped by an observed variable cost memory but zero inference FLOPs.

**(b) Bi-level (meta) base refinement.** Train the base to be the best *anchor for adapted models*, not the best standalone model: repeatedly run a short anchored solve on a training sample and move the base toward the solution (Reptile-style; first-order suffices). Anchor-optimal ≠ standalone-optimal — the meta-trained base can degrade standalone while the deployed system improves, and per-sample solves from it reach equal depth at smaller distance. This moves the depth–smoothness frontier itself.

**(c) Distillation before end-to-end.** Regressing the hypernetwork onto anchored targets is drastically more compute-efficient than pushing task-loss gradients through the composite: per-sample solves are well-conditioned local problems; end-to-end gradients through shared parameters conflict across samples (a known multi-task pathology). At matched budget, distillation delivered ~10× the improvement of naive end-to-end. End-to-end has a role, but later and gently (see (f)).

**(d) The EM self-anchoring loop** — the core engine. Alternate: **E-step**: regenerate targets with each solve *anchored at the hypernetwork's own current prediction* for that sample (short budget, warm-started there); **M-step**: re-regress the hypernetwork onto the new targets (plain L2; function-space distillation only for beyond-threshold targets). Mechanism: the anchor field is smooth by construction (it is a finite-capacity network's output), so each round adds a small depth increment and then filters it through the "expressible as a smooth function of x" sieve. It is alternating minimization of Σᵢ [task(base+ΔWᵢ) + λ‖ΔWᵢ − H(xᵢ)‖²] over both, hence stable; the fixed point is the **deepest smooth family of updates supported by the data density** — it constructs the floor directly instead of hunting for it. Convergence signature: increments decay geometrically while the neighbor-distance statistic rides just below threshold. Run 2–4 rounds; stop on the alarms.

**(e) Cheap variance reduction.** Average predictions over ~3 M-step seeds ("soup"; valid because the anchored protocol aligns solutions). Small, free, stacks with everything.

**(f) Joint two-timescale finisher** — the strongest finishing technique measured. After the EM stack: unfreeze **both** networks and train jointly on the task loss with the wheel's learning rate 3–10× below the hypernetwork's, the hypernetwork L2-anchored to its own distilled predictions, and save-on-best checkpointing. The timescale gap creates an implicit bi-level structure: conflicting per-sample gradient components average out at the wheel's low lr; only the consensus (common-mode, conditioning-contingent) direction moves it. Measured behavior: the system improves substantially while the wheel's *standalone* score barely moves — the wheel becomes a better substrate, not a different standalone model. Cautions: this phase is knife-edge without the anchor and lr discipline (10× too high a hypernetwork lr destroys the checkpoint in ~100 steps); the wheel lr must be high enough to actually engage (monitor the wheel's relative displacement — at too-low lr it contributes nothing); the wheel must hold exactly still *within* any phase that consumes cached targets; and if this phase runs long, follow it with a cheap warm-started EM round to re-consistency-ize targets against the moved wheel (the full interleave: joint drift → target refresh → repeat).

## 8. Falsified approaches (measured negatives, with mechanisms — do not retry without new reasons)

- **Target mixup / interpolated supervision**: midpoints between deep targets are high-loss (the well geometry); mixup supervises toward exactly those toxic midpoints.
- **Low-rank/PCA denoising of the target matrix**: the useful fine structure is genuinely high-rank across samples; projection destroys target depth. "Fine part = noise around a low-rank core" is false.
- **Update-magnitude or prediction-space guidance** (scale the update, or extrapolate conditioned-vs-base outputs): monotonically harmful when the pipeline above is used — distilled amplitudes are already calibrated. Keep a 5-minute scale-sweep as a diagnostic (fires only if training used heavy shrinkage); if it ever fires, distill the scale into the weights rather than paying extra wheel evaluations.
- **Recurrent/delta-rule blending of updates across autoregressive steps**: memory helps only when the blend window ≪ the state decorrelation time; otherwise stale updates anchored to departed states hurt, and hurt more with more memory. Fresh per-step firing is correct whenever conditions decorrelate quickly.
- **Function-space (output) distillation inside the EM loop**: unnecessary at-threshold (EM keeps targets there) and its noisier gradients degrade the fit. Its real property — refusing to memorize rough components (gap collapse) — makes it the tool for *beyond-threshold* targets only.
- **Unanchored deep solves as regression targets**: the original sin; produces the wells.
- **Naive fine-tuning of distilled checkpoints at ordinary learning rates**: destroys them; only the anchored, gentle, two-timescale form (7f) survives.

## 9. Deployment rules

- **Fresh per-solve hypernetwork firing** (justified by the decorrelation-time rule, not habit).
- **Mandatory guards** when the hypernetwork sees model-generated states: clip normalized encoder features (train-set feature statistics have near-zero variance directions that turn small generation artifacts into enormous inputs — unguarded, this produces NaNs within steps), and a trust-region cap on the update norm.
- **Conditioning horizon**: in autoregressive rollout the conditioning advantage decays as generated conditions degrade and can cross below the plain base; a confidence gate that attenuates the update with rollout uncertainty is the natural extension of the trust-region cap.
- **Non-stationarity**: if the underlying system drifts, a base trained on one window goes stale outside it, and the conditioning *map itself* drifts. Time-blocked train/test splits are mandatory (random splits hide this completely); plan periodic base/hypernetwork refresh on recent data.

## 10. Evaluation protocol musts

- Time-blocked and *interpolative vs extrapolative* held-out sets, reported separately.
- The **compute-matched control**: a plain wheel given the stack's entire training budget, compared at matched inference FLOPs. The thesis is not proven against anything less.
- Single-step metrics do not automatically convert to rollout accuracy; measure rollout horizon explicitly, with the guards installed.
- Encoder principle: phase-preserving features of the condition, never pooled summaries; encoder bandwidth is the capacity axis whose required size grows with target depth (deep targets encode fine condition structure that coarse features cannot separate), and it is the predicted first failure point at scale. Trunk depth was the strongest conventional axis measured (monotone, gap-shrinking); width has a floor but no observed ceiling; pooled summary statistics added nothing.
