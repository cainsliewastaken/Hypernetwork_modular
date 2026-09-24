# Diagnosing the Binding Constraint: Data-Starved vs Expressivity-Capped vs the Chaos Ceiling

*Method-agnostic. The question this answers: when the conditioned system's held-out error is not low enough, which of three fundamentally different limits is binding — and how to push the hypernetwork exactly as far as its size allows without confusing memorization for capability.*

---

## 1. The three limits

1. **Data-starved**: the map x → ΔW is learnable in principle, but the training targets are too sparse in condition space for the hypernetwork to interpolate between them. Signature: memorization (train ≪ test).
2. **Expressivity-capped**: the hypernetwork (encoder, trunk, or head) cannot represent the map even on its own training targets. Signature: underfitting (train stuck above target depth).
3. **The chaos ceiling (Bayes floor)**: the map's *predictable part* is exhausted. Part of each oracle solution encodes information about the answer y that the condition x does not determine — in chaotic systems at finite lead, necessarily so. No data quantity and no capacity recovers it, because it is not a function of x.

The train/validation tradeoff that appears when pushing toward oracle-level targets is **not a law**: it is the signature of operating at the frontier for the *current* target density. The frontier moves with density (verified by controlled A/B: identical target count, halved spacing, held-out flipped from net-negative to strongly positive). With enough appropriately spaced data, depth keeps converting to held-out gain — up to, and only up to, the Bayes floor.

## 2. The three-gap decomposition (log this in every run)

Held-out loss = **target depth** + **fitting gap** (train − depth) + **generalization gap** (test − train).

Each term has its own lever and they never lie in the same direction:

| term | meaning | lever |
|---|---|---|
| target depth | protocol ambition | anchor strength, EM rounds, density (deeper rungs need denser data) |
| fitting gap | pure expressivity | trunk depth first, then encoder bandwidth, then head rank |
| generalization gap | data density/count | target spacing above the decorrelation bend; more targets at valid spacing |

## 3. Signatures, from measured examples

**Expressivity-capped** looks like: train stuck far above target depth; generalization gap near zero (it generalizes everything it manages to learn); more data does nothing; the right capacity axis moves train and test together. Canonical example: an output head below the targets' intrinsic rank collapses entirely — train barely better than base, gap ≈ 0. The collapse rank grows with target depth; locate it on your targets before committing to low-rank heads.

**Data-starved** looks like: train ≈ target depth (fits perfectly); large generalization gap; capacity changes nothing or make it worse; halving spacing (not doubling count at the same spacing) fixes it. Canonical example: identical targets at two spacings — below the decorrelation bend the system *loses* to its own base; at the bend it wins strongly.

**A subtle third signature — encoder-limited**, distinct from trunk-limited: prediction error correlates with condition content *outside the encoded band* (two conditions the encoder cannot distinguish receive the same prediction while their true targets differ). Trunk capacity is fine; feature bandwidth is the wall. This is the predicted first failure at scale, because deeper targets encode finer condition structure. Test: raise encoder bandwidth at fixed everything else; in measurements it was monotone and never overfit, so it is safe to probe upward.

**One trap**: an over-capacity component can *fit* train through a learned bottleneck yet generalize worse than a plainly parameterized one at small n (a learned low-rank basis locks onto train-specific directions). "Fits train" is necessary, not sufficient; only the paired train/test movement under a capacity change is diagnostic.

## 4. The two-axis probe (mechanical procedure)

1. **Learning-curve probe (data axis)**: retrain on 50% and 75% of targets, same spacing distribution. Held-out still sloping in n → data-limited; buying capacity now is wasted. Flat in n → not data-limited.
2. **Capacity probe (model axis)**: at fixed data, halve and double capacity along each axis separately — trunk depth, encoder bandwidth, head rank, width. The axis that moves *both* train and test is where expressivity binds. (Measured ordering of returns on one system: depth > encoder bandwidth > width, with head rank mattering only near the collapse point — re-derive on yours; the probe is the deliverable, not the ordering.)

**Run the probe as a loop, not once.** Fixing a data limit routinely exposes the next expressivity limit (a memorizing head hides an undersized encoder, and vice versa). The frontier alternates: densify targets until the generalization gap closes → grow the binding capacity axis until the fitting gap closes → deepen targets (weaker anchors / more EM rounds) until the gap re-opens → repeat. This loop *is* the procedure for "pushing the hypernetwork as far as its size allows." It terminates when, simultaneously: the learning curve in n is flat, the fitting gap is ~0, and the EM loop's per-round increments have vanished with the smoothness statistic at threshold. Wherever held-out sits at that point is the empirical floor for the current encoder's information budget — and the only remaining move in the design space is giving the encoder more of x.

## 5. Estimating the chaos ceiling (so the project stays falsifiable)

The oracle number is **not** the right target: oracle solves see y, and in a chaotic system part of each solution is y-only information. The correct asymptote is the **Bayes map** — the best update predictable from x alone. Two estimators, neither requiring a trained hypernetwork:

1. **Neighbor-regression extrapolation**: regress each oracle target on its condition-neighbors' targets; plot unexplained residual vs condition spacing; extrapolate to zero spacing. The intercept is the y-only (irreducible) component.
2. **EM fixed point vs density**: run the EM loop at increasing target densities; the fixed points descend and asymptote. The asymptote is the empirical Bayes floor at the current encoder bandwidth (widen the encoder and it can move down, until the encoder carries all of x that matters).

Set expectations against this floor, not against the oracle. The gap between the floor and the oracle is a property of the physics at your prediction lead — it shrinks with shorter leads and is not a modeling failure.

## 6. The standing dashboard

Every training job should log: target depth; neighbor-target distance (with its threshold from your own annealing ladder); fitting gap; generalization gap (interpolative and extrapolative separately, time-blocked); EM per-round increment; learning-curve slope from the most recent data probe; wheel displacement and wheel-standalone score during any joint phase. Alarms: neighbor-distance at threshold, gap opening, EM increments below noise, learning curve flat while fitting gap ~0 (→ you are at the floor; stop buying data and capacity, start widening the encoder or accepting the ceiling).
