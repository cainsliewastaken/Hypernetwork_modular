# When Does Per-Sample Weight Conditioning Have Headroom? The Two Channels, Task Saturation, and Scaling

*Method-agnostic companion to the landscape and diagnostics documents. The question this answers: under what conditions can a hypernetwork emitting per-sample weight updates improve on a shared base wheel at all — and how that answer scales with wheel size, task size, and inference budget. Derived jointly from a 1D KS implicit-solver study (where conditioning provably had nothing to add at adequate wheel capacity) and a 2D turbulence diffusion study (where it bought ~18% at matched inference FLOPs). The reconciliation of those two outcomes is the content of this document.*

---

## 1. The two channels

A per-sample update ΔW(x) can improve the conditioned system through exactly two mechanisms, and they have different math, different diagnostics, and different scaling behavior.

**Channel 1 — information.** The update carries information about the required output that the wheel's other inputs do not determine. Formally, with t the required per-step output and z the wheel's per-step input, the information headroom is bounded by E[Var(t | z)] − E[Var(t | z, x)]: the conditional variance that x resolves beyond what the wheel already sees.

**Corollary (architectural, wheel-type-independent):** if the hypernetwork's input is a subset of what the wheel already receives (e.g., the condition is concatenated into the wheel's input at every step) and the task is deterministic (the target is a function of the condition), the information channel is **exactly zero**. Whatever conditioning buys in such architectures, it is not information. This corollary is the general form of the KS zero-headroom result and applies verbatim to diffusion, implicit, and supersampling wheels.

The information channel opens only when: the wheel does *not* see the condition (or sees a degraded version — then the update is a communication channel); the task is drawn from an operator *family* whose identity is not inferable from the wheel's inputs (regime/parameter families — the "non-homogeneous operator" route, which is sufficient but not necessary for headroom); or the hypernetwork receives side information the wheel lacks (history under a truncating encoder, exogenous parameters).

**Channel 2 — capacity localization.** Even with zero information headroom, a per-sample update lets an undersized wheel approximate the required operator *locally* — in the neighborhood of the current condition — instead of globally over the whole attractor. The unconditioned wheel must spend its capacity representing the operator everywhere at once; the conditioned wheel spends it on one neighborhood, with the hypernetwork indexing neighborhoods. For systems whose attractor is large but locally simple (chaotic dynamics generally), the local problem is systematically cheaper than the global one, at every scale. This is gain scheduling's economics, and it is the channel the entire per-FLOP thesis runs on. Measured signature: large oracle-vs-base headroom *even with the condition concatenated* — such headroom is pure Channel 2.

## 2. Task saturation, and why UAT is irrelevant

Universal approximation guarantees representability in the infinite limit and says nothing about the rate; the rate is the whole question. Every task has a saturation scale: the wheel size at which additional capacity stops mattering at the target accuracy. Above saturation, Channel 2 closes (nothing left to localize) and, in concat architectures, Channel 1 was closed from the start — so conditioning has provably nothing to add, no matter how well it is trained. Below saturation, Channel 2 is open and grows as the capacity deficit grows.

Calibration examples from the two studies: 1D KS at moderate resolution has a low-dimensional inertial manifold — its one-step solve map is a compactly representable function, and modest wheels saturate it; every adequately-sized configuration showed zero conditioning margin, while the one undersized-wheel configuration showed a 20% margin. A 2D turbulence step is a function of a different magnitude; no affordable wheel approaches saturation, and the margin at matched FLOPs was ~18% with a 4× oracle-vs-base headroom still unharvested. **"Nonlinear PDE" is not the relevant axis; approximation demand at target accuracy is.**

One nuance by wheel type (the honest scope limit of the original KS theorem): for a convergent fixed-point wheel, the training constraint pins the network only at the fixed point — a single-valued constraint set, which is what made the KS zero-headroom argument exact. Diffusion wheels are pinned along an entire noise-level tube per sample, and supersamplers across scales: richer constraint sets, higher capacity demand at the same nominal size, hence *more* room for Channel 2 — the qualitative two-channel conclusion is unchanged, and the quantitative balance tilts further toward conditioning.

## 3. Scaling economics: the right competitor is an ε-larger wheel

The wheel's cost is multiplied by K (iterations per solve); the hypernetwork fires once and amortizes K-fold. At matched inference FLOPs, the alternative to "wheel of size S + hypernetwork" is a wheel of size S·(1+ε) with ε = (hypernet forward)/(K · wheel forward) — typically a few percent. The thesis is therefore not "small wheels are good"; it is: **at any given inference budget, the conditioned wheel at that budget beats the plain wheel at that budget, provided the task is above saturation for that budget.** Both networks grow as the budget grows; the claim scales as long as the task's global complexity outruns affordable wheels — which becomes more true, not less, for larger physical systems.

The claim's honest boundary: as the affordable wheel approaches task saturation, the margin closes (the KS regime). And from the other side, at long chaotic leads the Bayes floor rises (see the diagnostics document) — the map's predictable part shrinks, capping what any conditioning can deliver. The interesting operating region is between the two limits, and both limits are measurable.

## 4. The headroom gate (run this before investing in any new system or scale)

The universal instrument: per-sample anchored oracle solves versus the trained base, at the wheel size in question, with the deployment's input plumbing (concat or not) in place.

- **oracle ≈ base** → the task is saturated at this wheel size (or the update space is too restricted — check with an unrestricted-space solve). Conditioning has nothing to harvest; "make the wheel bigger" is the correct spend. Do not interpret a null hypernetwork result at this operating point as a verdict on the method.
- **oracle ≪ base, condition concatenated** → pure Channel-2 headroom exists; the full method stack applies, and the harvestable fraction is governed by the density/depth machinery in the landscape document, capped by the Bayes floor.
- **oracle ≪ base, condition *not* concatenated** → mixed channels; the shuffle control (updates from mismatched conditions) separates input-specific effect from generic capacity, and comparing against a concat variant separates the channels.

**Project note (2D turbulence, HyperNetwork-Research, 2026-09): do not run the gate by default.** It was tested exhaustively on the w16/w32 concat-FNO diffusion wheels and nothing done with it made it worth running. Per-sample oracle solves did show headroom, but that headroom never turned into a trained hypernetwork. Weight-space copies of free oracles are brittle: a structured near-miss scores worse than the frozen wheel. Short-solve teachers do not transfer across fields. Oracle fits on training frames did not carry over to held-out times. The best results came from training the hypernetwork directly on the score loss. See `HyperNetwork-Research/diffusion/diagnostics/ORACLE_FORCING_SUMMARY.md`. In this repo the gate is available but is not a required step, and a null or positive gate says nothing about whether score training will work.

The corresponding protocol correction, learned the hard way: the old rule "verify the wheel is not the capacity bottleneck before interpreting anything" is exactly backwards for this program — a capacity-bottlenecked wheel is the *operating regime of the thesis*, not a confound. The regime to verify is that the bottleneck exists (the gate above), not that it doesn't.

## 5. The thesis plot and the scaling risks that are actually real

The claim is a statement about a *relationship*: hypernetwork margin versus wheel size at matched inference FLOPs (the wheel-size sweep). Expected shape: margin grows as the wheel shrinks below task demand, and closes as the wheel approaches saturation; the deployment question is where affordable budgets sit on that curve for the task at hand. Run the headroom gate at each sweep point — it predicts the margin before any hypernetwork is trained.

When scaling up, the wheel side is not the risk. The measurable risks are on the conditioning side: **encoder bandwidth** (deeper targets on larger wheels encode finer condition structure; the encoder-limited signature in the diagnostics document is the predicted first failure at scale) and **target density affordability** (the required condition spacing steepens with target depth and prediction lead; re-measure the decorrelation bend at each new scale before budgeting data generation). Both have standing instruments; neither is a reason to keep anything small.
