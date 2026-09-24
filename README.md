# hypercond

General code for conditioning an iterated solver (the **wheel**: diffusion denoiser, implicit/fixed-point solver, supersampler) with a **hypernetwork** that emits a per-sample weight update ΔW once per solve. Everything problem-specific sits behind one class, `hypercond.task.Task`; everything in the method stack is generic.

Every technique here is training-time. Inference cost is one hypernetwork forward per solve plus per-step application of the update.

## Layout

```
hypercond/
  task.py          Task interface — the only plug point for a problem
  update_space.py  which wheel weights move and how; resolution in the observed variable
  conditioned.py   ConditionedWheel: runs the wheel with per-sample weights (torch.func.vmap)
  hypernet.py      encoder -> guarded feature normalization -> residual trunk -> zero-init head; soup + trust cap
  oracle.py        anchored proximal solves on common-random-number banks
  distill.py       M-step regression (time-blocked early stopping, optional semi-supervised task loss)
  em.py            EM self-anchoring loop with the two-alarm stopping rule
  stages.py        base training, first-order bi-level (meta) refinement, two-timescale joint finisher
  diagnostics.py   headroom gate, twin test, decorrelation curve, annealing ladder, guidance sweep,
                   three-gap decomposition, learning-curve and capacity probes, FLOP/epsilon accounting
  pipeline.py      wiring, stage runner, checkpoints, evaluation
  dashboard.py     JSONL logging and alarms
  config.py        typed config (every section maps to a part of the reference docs)
train.py           run the stack
eval.py            evaluate a checkpoint and run diagnostics
configs/default.yaml
tests/             synthetic toy task + end-to-end smoke test (not a problem file)
```

## Plugging in a problem

Subclass `Task` and point the config at it (`task: my_pkg.tasks:MyTask` or `task: path/to/file.py:MyTask`).

```python
class MyTask(Task):
    solver_steps = 20            # K, for FLOP accounting
    condition_in_wheel = True    # does the wheel already see x? (headroom-gate verdict)

    def build_wheel(self): ...                 # nn.Module, vmap-compatible
    def datasets(self): ...                    # {"train": ds, "val": ds, ...}; items are dicts of tensors
    def condition(self, batch): ...            # hypernet input x, [B, ...]
    def make_bank(self, n, generator, purpose): ...   # "solve" | "eval" (fixed, CRN) | "train" (fresh, allocated)
    def loss(self, model, theta, batch, bank): ...    # per-sample loss [B]; call model(theta, s, *args)
    def observed_range(self): return (0.0, 1.0)       # range of s
```

Inside `loss`, the wheel is always called as `model(theta, s, *args, **kwargs)`. `theta=None` is the plain base. `s` is the inference-observed variable (noise level, iteration index, stage, scale) as a float or `[B]` tensor; the update is resolved at `s`, so ΔW(s) = g₀ + u·g₁ with the default `obs_basis: affine`. Positional tensor args with leading dimension B are split per sample; pass `s` in `args` too if the wheel consumes it.

Optional hooks: `build_encoder` (phase-preserving features; the default is flatten + linear), `update_param_names`, `target_indices` (dense patches), `neighbor_pairs` (e.g. adjacent saves on a trajectory; default is 1-NN in condition space), `evaluate` (deliverable metrics such as sampled rollout MSE), and `wheel_example_inputs` (FLOP accounting).

Two data conventions matter. Datasets should be ordered by trajectory/time, because dense target patches, learning-curve prefixes and the distillation validation tail are all taken contiguously; a shuffled dataset silently turns a dense patch into thin coverage. Evaluation splits should be time-blocked, with interpolative and extrapolative sets as separate keys.

For a diffusion wheel, `s` is τ; `make_bank(purpose="train")` is where low-τ allocation (e.g. τ ∝ u²) goes; `"solve"` and `"eval"` banks should be fixed, τ-stratified (τ, ε) sets.

## Storage (symlinks to $SCRATCH)

Home space is limited, so data and outputs live on scratch behind two git-ignored symlinks at the repo root:

```
training_data     -> /pscratch/sd/c/cainslie/training_data            (dataset: training_data/data_lowres, 20,701 .mat files)
model_checkpoints -> /pscratch/sd/c/cainslie/model_chkpts_scratch/hypernetwork_modular
```

Each run writes to `model_checkpoints/<run>/` (checkpoints, `log.jsonl`, `train_report.json`), and `eval.py` writes its reports to `model_checkpoints/<run>/eval/`. On a fresh clone, recreate the links with `ln -s` as above.

## Running

```bash
pip install -e .
python train.py --config configs/default.yaml --set task=my_pkg.tasks:MyTask out_dir=model_checkpoints/exp1 base.steps=20000
python train.py --resume model_checkpoints/exp1/latest.pt           # continues unfinished stages / EM rounds
python eval.py --ckpt model_checkpoints/exp1/latest.pt --diag gate guidance flops
python -m pytest -q tests                               # smoke test on the toy task
```

`--set` takes dotted overrides (`em.rounds=4`, `stages='[base,gate]'`). Each stage writes `after_<stage>.pt`; every run appends to `log.jsonl`.

## The pipeline

Stages run in `cfg.stages` order, defaulting to `base → meta → gate → em → joint`.

**base** trains the wheel with `theta=None`, save-on-best, then rebuilds the update-space bases from the trained weights. **meta** is first-order bi-level refinement: a short anchored solve from the base, then the adapted loss is backpropagated into the base with the adapted coefficients held fixed. The standalone score may get worse while the adapted score improves; that is expected. **gate** runs anchored oracle solves against the base, with a shuffle control, and prints a channel verdict. It predicts the margin before any hypernetwork training. **em** runs round 0 anchored at the base and later rounds anchored at, and warm-started from, H(x). Each round distills a prediction-averaged soup, refits the trust cap, logs depth, neighbor distance, fitting gap, generalization gap and increment, and stops on the alarms. The best round on the selection split is kept. **joint** trains both networks with the wheel learning rate 3–10× below the hypernet's, H L2-anchored to its distilled predictions, save-on-best, and logs wheel displacement and wheel-standalone score. It is followed by one warm-started EM refresh round.

The update space is linear in θ. Per selected operator weight it has a multiplicative per-channel gain plus an additive term: `subspace` uses fixed bases (top singular vectors) with a free r×r core, so there is no factorization gauge, and `dense` is a free matrix. Biases, norm shifts and embeddings are excluded by default because they are the memorization channel. θ = 0 reproduces the base exactly, and the head is zero-initialized, so training starts at the base. Bases are frozen once targets exist, and the runner refuses `base`/`meta` after that point.

Deployment guards are always installed in evaluation. Encoder features are standardized with train statistics under a variance floor and then clamped, and the update norm is capped by a trust region fitted to target norms. `HyperEnsemble.predict(x, gate=...)` accepts a per-sample confidence gate for rollouts.

## Diagnostics (`eval.py --diag ...`)

| name | question it answers |
|---|---|
| `gate` | Is there headroom at this wheel size, and from which channel? |
| `twin` | Is target scatter selection noise? (same sample, jittered inits, two anchor strengths; distance + barrier) |
| `decorrelation` | Where is the spacing bend (condition correlation below which targets stop sharing structure)? |
| `ladder` | Depth vs neighbor distance across anchor strengths; with `--ladder-distill`, locates *your* threshold |
| `guidance` | Are amplitudes calibrated? (fires only after heavy shrinkage; then distill the scale, never pay 2× wheel) |
| `learning_curve` | Data axis: still sloping in n → data-limited |
| `capacity` | Model axis: which of depth / encoder_dim / width / head_rank moves train *and* test |
| `flops` | ε = hypernet / (K·wheel): the size of the matched-FLOP plain-wheel competitor |

Every eval also reports base, conditioned and shuffled losses per split, plus the three-gap decomposition (depth + fitting gap + generalization gap) against the latest targets.

The compute-matched control is not automated, because it is a separate training run. Train a plain wheel of size S·(1+ε+overhead) on the stack's full training budget, using `stages='[base]'` with a larger wheel from your task, and compare.

## Deliberately absent

These were measured negatives, and the code does not offer them: target mixup, low-rank/PCA denoising of targets, runtime guidance, recurrent blending of updates across autoregressive steps, function-space distillation inside the EM loop, unanchored deep solves as targets, and chained warm-starting between neighbors. Every solve is anchored and warm-started from its own anchor.

## Practical notes

`ConditionedWheel` materializes per-sample weights, so memory is B × (updated parameters). Use `vmap_chunk` or a smaller oracle batch for large wheels. The wheel must be vmap-compatible, which rules out BatchNorm running-stat updates and data-dependent Python control flow. `em.neighbor_threshold: 1.0` is the value measured on other systems; measure yours with the ladder before trusting the alarm. On the toy task the FLOP report shows ε ≫ 1 because the wheel has about 500 parameters. The thesis only applies where ε is a few percent.
