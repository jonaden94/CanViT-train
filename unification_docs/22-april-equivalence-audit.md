# 22 — Is today's pretraining still the original? (April 2026 equivalence audit)

**Date:** 2026-09-07. **Nature:** verification record. **Verdict: yes — no unwanted
divergence, no bug.**

**Question (owner).** This repo is the result of merging five repos. Does *basic random
distill pretraining* still compute what the original did, before any of that started?

**Reference.** The state when the owner took the project over from Yohaï:

* `CanViT-PyTorch` @ `cc77f4f8277945cc10423d31118ad3439d85d7e8` — 2026-04-27
* `CanViT-pretrain` @ `1a36eecda04a05cf0c199351c3d2301fcce1ad81` — 2026-04-26

250 commits and 4.5 months of divergence. Both commits are still reachable: core's from the
retired `CanViT-PyTorch` clone, pretrain's from this repo's own history (it *is* this repo,
renamed twice).

Explicitly out of scope, at the owner's direction: the data pipeline, viz/logging, launchers,
and the added policy/joint machinery — except where one of them feeds a number into the
training path. (Target standardization is the one that does; see §5.)

---

## 1. Method

Six steps. The order matters — steps 1 and 3 are what keep this cheap and interpretable.

1. **Split the timeline at an existing certified baseline.** `parity_probe.py` plus
   `harness/tests/test_rollout_parity.py`'s digest `9a0100a1a3de3acd` already certified
   *July pre-cutover → today*. So only **April → July pre-cutover** was ever unexamined —
   and that is precisely the owner's own modification period, before any parity harness
   existed. Do not re-verify a leg that is already gated.
2. **Feasibility before design.** April's code imports cleanly under today's `.venv-cu126`
   with `PYTHONSAFEPATH=1` and an explicit `PYTHONPATH` into `git archive` snapshots. Had it
   not, the whole numeric approach would have collapsed to a read-only comparison, and the
   evidence would have been much weaker — worth knowing on step 2, not step 6.
3. **Eliminate confounders before comparing.** Compare `state_dict` under one seed FIRST. If
   initialization differs, every later loss mismatch is uninterpretable. (It did not differ.)
4. **Deterministic synthetic probe.** No dataset is needed and none exists in the old format —
   `parity_probe.py` was built for exactly this: tiny CPU model, pinned RNG, synthetic
   batches. "Inject artificial data" is the existing design, not a workaround.
5. **Analytical pass.** The probe exercises ONE path (uniform patcher, CPU, no AMP, no
   compile, no DDP, `chunk_size=2`, two branches, 25 steps). Bit-identity there is *silence*
   about everything else. Every changed line must be classified: additive-and-gated /
   order-preserving refactor / real change.
6. **Compare config defaults separately.** Recipe drift is not code drift, and only the
   latter is a bug.

### Instruments (`april_equivalence/`, all re-runnable)

| file | what it does |
|---|---|
| `parity_probe_april.py` | mirror of `parity_probe.py` driving April's `train/step.py::training_step` |
| `cmp_state.py` | same-seed `state_dict` dump, `old`/`new` — the confounder check |
| `cmp_vitb16.py` | same, at the **production** config (`vitb16`, teacher_dim 768, canvas 32) |
| `lr_compare.py` | April's `SequentialLR` vs today's `LambdaLR` over 120k steps |
| `record_april.json` | the April loss stream, full precision |
| `state_dict_*.json` | the four dumps, for diffing without a GPU |

Run the April side with:

```bash
S=<scratch>; R=/mnt/vast-nhr/projects/nib00021/jonathan/repos
mkdir -p $S/pretrain $S/pytorch
git -C $R/canvit           archive 1a36eec | tar -x -C $S/pretrain
git -C $R/CanViT-PyTorch   archive cc77f4f | tar -x -C $S/pytorch
cd $R/canvit && PYTHONSAFEPATH=1 PYTHONPATH="$S/pytorch:$S/pretrain" \
  .venv-cu126/bin/python unification_docs/april_equivalence/parity_probe_april.py out.json
```

---

## 2. Numerical results

| check | result |
|---|---|
| 25-step loss stream, April `training_step` vs today's `run_rollout` | **byte-identical**, `9a0100a1a3de3acd`, 0/25 per-step mismatches |
| Model at init, probe config (`vits16`, teacher_dim 384, canvas 8) | **bit-identical** — 239 tensors, every key/shape/value |
| Model at init, **production** config (`vitb16`, teacher_dim 768, canvas 32) | **bit-identical** — 96,538,624 params, 239 tensors |
| LR schedule over 120k steps, production recipe | worst **2.5e-13 relative** |
| `harness/optim/scheduler.py`, `harness/optim/ema.py` | **byte-identical files** to April's |
| `rope`, `vpe`, `standardizers`, `teacher`, `viewpoint` | zero-line diffs |

**This re-derived a digest the codebase said could not be re-derived.** `parity_probe.py`'s
docstring states the original implementation "is gone… the recorded constant in the test is
now the sole reference". It is not gone — it is in git history, and the constant now has an
independent second derivation.

---

## 3. Analytical pass — every changed line in core, classified

The probe says the numbers match. This section says *why*, which is the difference between
"matches" and "will keep matching".

### Additive and gated off by default

`CanViTConfig` gained `patcher_name="uniform"`, `foveated_patcher`, `square_patcher`,
`vit_modulation` (disabled), `n_canvas_self_attn_blocks=0`, `canvas_self_attn_mlp_ratios=[]`.
**No pre-existing default was changed.** Each of these adds parameters when enabled — and
both models produced 239 identical tensors, so they are *provably* inert at default rather
than merely intended to be.

Also additive: `vitb8/7/6` and `*_modulate` backbone variants; `PatchEmbed(stride=…)`
defaulting to `patch_size`; `uniform_grid_coords` for overlapping patches.

### Order-preserving refactors

* **Patching moved into the model.** April: the caller ran `sample_at_viewpoint` and handed a
  `[B,3,128,128]` crop to `patch_embed`. Today: the model takes the full image and the uniform
  patcher crops internally. Same result, verified.
* **Prediction heads moved inside `forward`.** April's trainer called `model.predict_*` from
  `compute_loss`. Today they run inside `CanViTForPretraining.forward`. Same ops, same order —
  and it exists to fix a DDP bug: heads called outside the wrapped forward do not get their
  gradients AllReduced (manifests as ~√N grad-norm scaling on head params only). Latent in
  April, which had no DDP.
* **`mod=None` plumbing through attention.** `_apply_local_mod` returns its inputs untouched
  and `gate_a is None` when unmodulated, so the expression reduces to April's exactly.
* **`ViTBlock(modulated=False)`** is April's line verbatim:
  `x + ls1(attn(norm1(x), rope))`, then `x + ls2(mlp(norm2(x)))`.
* **`glimpse_size_px`.** April: `grid × patch_size`. Today:
  `(grid−1) × patch_stride + patch_size` — identical whenever `patch_stride` is unset, which
  is the default. Only differs if overlapping patches are opted into.
* **LR schedule.** `SequentialLR(LinearLR→ConstantLR)` → `LambdaLR(_lr_lambda)`. A
  reimplementation, agreeing to 2.5e-13 (float associativity: a multiplicative recurrence vs
  a closed form). Note `harness/optim/scheduler.py` still exists but is now only a **test
  reference**; `build.py::_lr_lambda` is the production path.

### Checkpoint/serialization only

Schema normalization + migration, `patch_stride` round-trip, nested-config coercion. Load
path, not train path.

### Unchanged where it counts

Loss is line-for-line April's `compute_loss` (`F.mse_loss` ×2, `torch.stack(active).sum()`).
Optimizer is one AdamW group, `betas=(0.9, 0.999)` unoverridden, `task_weight=1.0`. Step order
is `zero_grad → [DDP allreduce] → clip → step → scheduler.step`, matching April with the
allreduce deliberately placed before clipping to preserve 1-GPU semantics.
`backward_pass_autocast="off"` in both. The `random`/`full_scene` viewpoint samplers are
untouched; `viewpoint.py` gained only foveated additions.

---

## 4. Hyperparameters

Identical across the board: `peak_lr` 4e-4, `warmup_steps` 100k, `start_lr` 1e-7,
`weight_decay` 1e-4, `chunk_size` 2, `continue_prob` 0.5, 1 full + 1 random branch,
`min_viewpoint_scale` 0.05, `grad_clip` 1.0, `ema_alpha` 0.1, canvas 32, glimpse grid 8,
scene 512, `steps_per_job` 4992, `compile`/`amp`/`non_blocking_transfer` all on.

Validation protocol also matches: `eval_policy="auto"` resolves to `coarse_to_fine` for
uniform distill, which is April's quadtree.

---

## 5. The two real differences — both intended, neither a bug

**1. Normalizer pooling.** April used **one** shard, chosen from the *seed-dependent*
schedule slice, so target statistics were a function of the seed. Today pools **four** from
the sorted list, so they are seed-independent; sampling error ~1.3% → ~0.65% of a std, and
shard-to-shard variation was measured at 1.05× split-half noise (shards are effectively
i.i.d., so pooling behaves like drawing more samples). Removing the seed-dependence is a bug
fix.

> **Consequence:** absolute loss scale and `*_cos_norm` are **not comparable** across
> 2026-07-28. `*_cos_raw` and downstream probe accuracy are.

**2. `batch_size` → `batch_size_per_gpu`.** April's loop had no DDP at all. Identical on one
GPU at 64; on N GPUs use 64/N to match the original global batch. The stochastic trajectory
length is broadcast from rank 0 (`harness/rollout/engine.py:189-192`), so multi-GPU takes the
same trajectory as single-GPU rather than a per-rank mixture.

Everything else is an API rename (`forward(glimpse=)` → `forward(image=)`) or
validation-only.

---

## 6. What was NOT established — so it is not claimed

* **Real-data end-to-end equivalence.** The probe is synthetic. The old data format is not
  available, so this is the strongest evidence obtainable, but it is not a matched production
  run.
* **Cross-torch-version reproduction.** Both sides ran under today's torch 2.11.0+cu126. This
  certifies *code* equivalence; it does **not** claim April's historical numbers reproduce on
  today's stack.
* The data pipeline, and April's validation-metric formulas (the viewpoint protocol was
  checked; the per-metric formulas were not).
* Paths the probe never takes: AMP/bf16, `torch.compile`, DDP, `chunk_size=1`, foveated.

## 7. Defect found while doing this

`parity_probe.py` would not run: `ModuleNotFoundError: canvit_train`. The P1 rename
(`3d7eb1f`) had broken it, along with 21 other scripts under `unification_docs/`, because
that rename deliberately skipped this directory on the rule *"dated docs are records, they
keep the old paths"*.

That rule is right for prose and **wrong for executable code** — a script that cannot run
preserves no history that git does not already hold. Fixed in `0d82713`. The refined rule:
**prose keeps historical names; code gets renamed, because code is run, not read as history.**
`capability_matrix.py` had already been fixed in P1 for exactly this reason (a test executes
it); this generalises what was then treated as a one-off.
