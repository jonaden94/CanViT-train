# 20 — Merging CanViT-eval into CanViT-train

**Date:** 2026-08-31. **Nature:** plan + record. Stage 0 has been EXECUTED (see §5);
Stages 1-6 have not.
**Goal:** retire `CanViT-eval` entirely; one repo does all training and all evaluation.
`CanViT-PyTorch` follows in a second phase (§8), after which the repo is renamed `canvit`.

## 1. Why, in one paragraph

Not tidiness. `EntropyGuidedC2F` exists in both repos, 108 diff lines apart, and one of
them produced a published number. `in1k/rollout.py` imports `consumes_full_image`,
`derive_glimpse_px` and `make_random_viewpoints` from `..ade20k.rollout` — a task reaching
into another task's package, in violation of this repo's own layout rule, purely because
there is no shared episode runner. And this repo *writes* `pretrain_view_scale` and
`teacher_name` into published distill checkpoints while reading neither — `canvit_eval` is
the only consumer, which is how the exp22 eval-scale break happened — while for in1k it
publishes an empty metadata block and a directory that cannot be loaded back at all
(measured: §5 F6). The merge fixes three real
defects that happen to share a root cause.

## 2. The structural insight the plan is built on

`canvit_eval/episode.py::run_episode` is the abstraction this repo is missing.

It runs the glimpse loop once, handles the uniform-pre-crop vs foveated-full-image routing
(`consumes_full_image`), derives `glimpse_px` from the model with a hard token-count guard,
applies `override_scale`, and returns the full `CanViTOutput` per step. Each task then reads
what it needs from those steps.

This repo instead has **three** eval rollouts:

| task | where | returns |
|---|---|---|
| ade20k | `ade20k/rollout.py::rollout_canvas_hidden` | canvas_hidden per timestep |
| in1k | `in1k/rollout.py::rollout_cls_tokens` | cls tokens per timestep |
| distill | `distill/viz/validate.py::validate` (nested under **viz/**) | cos/recon series |

`rollout_canvas_hidden` and `rollout_cls_tokens` differ only in the readout. That in1k had
to cross-import rather than duplicate is the evidence that the generalization is missing,
not that it is unwanted.

**Target:** one shared episode runner in `harness/rollout/`; per-task **readout** and
per-task **reduction** stay in `distill/`, `ade20k/`, `in1k/`. Share the loop, not the
metrics. This is the same split `canvit_eval` already has, so it is a demonstrated design
rather than an invented one.

## 3. What is in scope, and what is deliberately dropped

**Ported:**

- the generic episode runner (§2)
- `resolve_view_scale` + `teacher_probe_for_model` — the two footgun-closers (§5, stage 2)
- the three policies this repo does not expose (`fine_to_coarse`, `full_then_random`,
  `repeated_full_scene`) — **already in `canvit_pytorch.policies`**, so this is wiring
- `reconstruction` — as ONE implementation shared with distill validation
- `ade20k-seg-dinov3` — the DINOv3 teacher baseline (single passive forward, mIoU at t0)
- `ade20k_obj`'s **per-row IoU metric only**, as an optional per-(image, class, timestep)
  output of the existing ade20k eval — NOT its three-stage cached-feature pipeline
- `bench/pt/` (844 LOC inference-latency benchmarking) plus an archived baseline (§6)
- `tests/test_view_scale.py`, `tests/test_iou_equivalence.py`

**Dropped, decided 2026-08-31:**

| dropped | why |
|---|---|
| `batch.py` (409 LOC sweep driver) | subprocess matrix + skip-existing duplicates what `slurm/runs/` + run dirs already do; importing a second orchestration model is the cost, not the benefit |
| `ade20k_obj`'s staged pipeline | same objection: a second notion of "run with cached intermediates" |
| 23 `slurm_nhr/*.sbatch` | 19 are exp19–exp22 history; new launchers get written on demand against the new interface |
| `notebooks/` + `results/` (1.1 GB) | owner wants a new way of viewing results built against the new JSON output; old results stay readable in the archived repo |
| `provenance.py` anonymization | no paper-submission requirement; keep `run_metadata`'s full-config record |
| `statsmodels` | runtime dependency that **nothing imports** |
| `tests/test_batch.py` | tests the dropped sweep driver |

`fovi` does **not** fold in. End state is two repos plus archives.

## 4. Output format

Scalars + metadata → **JSON**. `.pt` only where actual tensors are needed (`ade20k_obj`
per-row data, DINOv3 feature caches). Today `ade20k_seg`, `in1k_clf` and `reconstruction`
all `torch.save` a dict of scalars, which cannot be diffed, grepped or read without torch.

Note `pandas` currently sits in canvit_eval's **dev** dependency-group while
`ade20k_obj/iou.py` imports it at module level — that task only works in a dev install.
Porting the per-row metric means either a real `pandas` dependency or writing parquet
through `pyarrow`, which this repo already has.

## 5. Stages

Each stage is independently shippable, independently gated, and leaves the repo working.
Do not collapse them: if a metric moves, the whole value of staging is knowing which stage
moved it.

### Stage 0 — Capture the baseline — EXECUTED 2026-08-31

Everything in this section is measured, not planned. **Every later stage is gated against it.**

**Where it ran.** Interactively, on a **MIG 3g.40gb slice** of an A100-80GB (driver 570.211;
`CanViT-train/.venv-cu126` = torch 2.11.0+cu126, `CanViT-eval/.venv` = 2.11.0+cu128). The
production runs compared against ran on a **full A100-SXM4-40GB** (`ggpu104`). That
difference is load-bearing — see F1.

**Instruments.** `scripts/eval_ade20k_checkpoint.py` (already in the repo) for ade20k; two
throwaway drivers for in1k and distill, deliberately kept OUT of the repo since Stage 0
changes no code. All three build the task, load the checkpoint with `strict=True`, build
only the val loader, and call `task.evaluate` — the production eval path, nothing
reimplemented. `git diff 716051a..HEAD -- '*.py'` touches only `scripts/` and
`unification_docs/`, so `canvit_train/` at HEAD is identical to the commit these runs were
pinned to and the logged numbers are a valid baseline for HEAD.

#### Checkpoints — revised from the plan, with a reason

The plan named the three *latest* checkpoints. Two are end-of-array-job writes at steps
where no validation ever ran, so re-evaluating them yields a number with nothing to check it
against. Substituting the nearest checkpoint whose step **has a logged eval** buys a free,
independent validation of the instrument before anything is ported:

| task | checkpoint | step | why this one |
|---|---|---|---|
| distill | `jon_exp32_pretrain_lrdrop/exp32-fovi/checkpoints/step-1916928.pt` | 1916928 | the next array job's step-0 eval logged this exact step |
| in1k | `jon_exp33_in1k_finetune/in1k-fovi-ti-1196k/checkpoints/best.pt` | 400000 | the 10000-step eval cadence lands on it |
| ade20k | `jon_exp34_ade20k_probe/ade20k-fovi-ti-1196k/checkpoints/best.pt` | 30500 | single-job run, so `best.pt` is a real eval step |

distill's `latest.pt` (step 2007040) is recorded too, without a logged counterpart. All
three are foveated at `pretrain_view_scale = 2.0` — the harder case, as intended.

#### F1 — The gate is EXACT on one machine, ~1e-5 across machines

Repeating a deterministic eval on the same GPU is **bit-identical**: `fixation_grid` on
ade20k reproduced all ten timesteps at `+0.00e+00`, in1k returned `0.83632 / 0.97006` twice,
distill returned `0.925767719745636` twice. Comparing the same deterministic quantity to the
*production log* does not:

| quantity | production (full A100) | here (MIG slice) | Δ |
|---|---|---|---|
| ade20k `miou_t0` @ 30500 | 0.3768244010757639 | 0.3768156179363609 | −8.8e-6 |
| distill `val_metric` @ 1916928 | 0.9257643967866898 | 0.9257677197456360 | +3.3e-6 |

Same code, same weights, same data — different SM count, so different bf16 kernel and
reduction choices. **Consequence for every later stage: gate by re-running before-and-after
on the SAME machine and demanding exact equality. Never gate against a number in an old
log** — that comparison has a ~1e-5 floor unrelated to the refactor.

#### F2 — Only some eval policies can carry a gate

`random`, `coarse_to_fine` and `fine_to_coarse` draw viewpoints from the global RNG and
nothing seeds them. Two back-to-back ade20k runs of `random` on identical weights:

| | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Δ(run1−run2) | **0.0e+00** | −1.3e-03 | −6.6e-05 | −2.6e-03 | +8.4e-04 | +6.0e-04 | +1.6e-03 | +9.0e-04 | −4.0e-04 | −2.0e-05 |

t0 is exact because every policy opens on the deterministic full-scene anchor; t1..t9 carry a
**±3e-3 band**. On in1k the same policy (the one exp33 actually used) has a ±5e-4 band on
top1 over 50k images: `0.83702 / 0.96934` here vs `0.83658 / 0.96962` logged.

**Deterministic, and therefore the gates:** `fixation_grid` (fixed internal seed) and `full`
with an explicit scale pin. Both verified bit-identical on repeat, on all three tasks.

#### The baseline

Foveated arms, `fixed_scale = 2.0`, `resize_mode = squish`, `scene_size = 512`,
`canvas_grid = 32`.

**ade20k** — probe step 30500, T = 10, `eval_batch_size = 32`:

| policy | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 |
|---|---|---|---|---|---|---|---|---|---|---|
| `fixation_grid` **(GATE)** | 0.3768156 | 0.4035657 | 0.4186816 | 0.4234078 | 0.4369073 | 0.4451849 | 0.4481932 | 0.4488288 | 0.4514489 | 0.4544373 |
| `full` + pin 2.0 **(GATE)** | 0.3768156 | 0.3918174 | 0.3932582 | 0.3918565 | 0.3903892 | 0.3891984 | 0.3882820 | 0.3877608 | 0.3869647 | 0.3861163 |
| `random` (what exp34 logged; ±3e-3) | 0.3768156 | 0.4012284 | 0.4137365 | 0.4232487 | 0.4313182 | 0.4347763 | 0.4380502 | 0.4412822 | 0.4432737 | 0.4455040 |
| `full`, NO pin (see F5) | 0.2788730 | 0.2883050 | 0.2860560 | 0.2810370 | 0.2761970 | 0.2717320 | 0.2678090 | 0.2643220 | 0.2612410 | 0.2584140 |

**in1k** — finetune step 400000, T = 4, `eval_batch_size = 64`, full 50k val:

| policy | top1 | top5 |
|---|---|---|
| `fixation_grid` **(GATE)** | 0.83632 | 0.97006 |
| `random` (what exp33 logged; ±5e-4) | 0.83702 | 0.96934 |
| `full`, NO pin (see F5) | 0.72248 | 0.91346 |

**distill** — `val_metric` = `val/scene_cos_raw_t9`, `fixation_grid` (what `auto` resolves to
for foveated), 256-image seeded val subset, T = 10:

| checkpoint | step | val_metric **(GATE)** |
|---|---|---|
| `step-1916928.pt` | 1916928 | 0.925767719745636 |
| `latest.pt` | 2007040 | 0.925623193383217 |

#### Cross-repo agreement: canvit_train ≡ canvit_eval, exactly

The plan's condition was "if they disagree, **stop and diagnose before porting anything**".
On the deterministic policy they do not disagree at all. ADE20K, probe step 30500,
`canvit_train --eval-policy full --override-scale 2.0` against
`canvit_eval ade20k-seg-canvit --episode.policy repeated_full_scene --episode.override-scale 2.0`:

**all ten timesteps identical at `+0.00e+00`.** Not "within tolerance" — the same float.

That is stronger than expected and it de-risks the port: the two episode runners, the two
mIoU reductions and the two dataset/transform paths are already one implementation in
practice (`mIoUAccumulator` and the ADE20K val dataset/transforms live in `canvit_pytorch`;
both repos upsample-then-argmax). §2's claim that `run_episode` is the abstraction this repo
is missing is now confirmed rather than asserted.

**The in1k cross-repo check could not be run at the time — blocked by F6.** It was
unblocked by Stage 0b and then agreed exactly; the numbers are in that stage.

#### F3 — `random` is a DIFFERENT POLICY in the two repos. Do not merge the name.

Same checkpoint, `random` + pin 2.0. The two repos disagree by **0.0205 at t0**, then converge:

| | t0 | t1 | t2 | t9 |
|---|---|---|---|---|
| canvit_train `random` | 0.376816 | 0.401228 | 0.413736 | 0.445504 |
| canvit_eval `random` | 0.356355 | 0.398676 | 0.414968 | 0.445384 |

Two independent causes, both real:

1. **The t0 anchor.** canvit_train's `random` opens on the full-scene anchor
   (`start_with_full_scene=True`); canvit_eval's opens on a random glimpse — its
   `full_then_random` is the anchored one. The whole t0 gap is this.
2. **Patcher-awareness.** canvit_train routes a foveated/square model through
   `RandomSelector`, i.e. the scale/center law the backbone was *pretrained* under;
   canvit_eval always uses core's uniform safe-box `random_viewpoints`. For a foveated
   model **canvit_train's is the correct one** — same footgun as job 15025338, documented
   at `ade20k/rollout.py::make_random_viewpoints`.

So §5's "decide which is correct" is answered: canvit_train, for foveated. And the merged
CLI must not let one `--policy random` mean two things — name them apart, or make the
anchor an explicit flag.

#### F4 — `eval_override_scale` exists on ade20k ONLY

`open_loop_viewpoints(..., override_scale=...)` supports the pin for every task, but only
`Ade20kConfig` exposes a field for it (`config.py:125`). `In1kConfig` and distill's `Config`
have none, so neither can pin the eval view-scale from config — and in1k is precisely the
task whose `HISTORICAL_DEFAULTS` entry is the *documented* OOD footgun (foveated → unpinned
`coarse_to_fine`). Stage 3 must unify this knob across the three tasks while leaving each
task's default alone — the pattern already blessed in `unify-the-knob-not-the-defaults`.

#### F5 — `HISTORICAL_DEFAULTS`' own advice is wrong for foveated, and it is expensive

Its note reads: *"Pass `--cfg.eval-policy fixation_grid` (or `full`) for a scale-pinned
foveated deploy."* `full` is **not** scale-pinned: it is `repeated_full_scene`, emitting
`Viewpoint.full_scene` at **scale 1.0**, whereas a `mode='fixed'` foveated model's own FULL
anchor sits at `fixed_scale` (2.0 here — proven by `random`'s t0 matching `fixation_grid`'s
t0 exactly). Measured cost of following the docstring:

* ade20k: −0.098 mIoU at t0, widening to **−0.128 at t9**, with the curve *decaying* as
  glimpses accumulate — the textbook OOD signature.
* in1k: top1 **0.72248 vs 0.83632**, i.e. −0.114.

`full` is in-distribution only with an explicit `--cfg.eval-override-scale`, which per F4
in1k cannot even express. Fix the docstring; the knob is the actual remedy.

#### F6 — `to_hf`'s in1k output cannot be loaded back. Root cause is in core.

Two defects, one blocking:

**(a) The published classifier is unloadable.** `python -m canvit_train.checkpoint.to_hf`
on an in1k checkpoint writes a directory whose `from_pretrained` raises:

```
TypeError: CanViTForImageClassification.__init__() missing 1 required keyword-only argument: 'model_config'
config.json = {"backbone_name": "vitb16", "glimpse_grid_size": 8, "n_classes": 1000}
```

Reproduced in BOTH venvs (huggingface_hub 1.11.0 and 1.7.1), so it is not a version skew.
Root cause is in **CanViT-PyTorch**, not in `to_hf`: `from_pretrained_with_new_head` passes
`model_config={k: v for k, v in vars(cfg).items() ...}` — nested *dataclass instances*, not
plain dicts — so `PyTorchModelHubMixin` cannot JSON-encode it and **silently drops the key**.
Verified directly: a freshly built classifier has
`_hub_mixin_config = ['backbone_name', 'glimpse_grid_size', 'n_classes']`. Three sites pass
`vars(cfg)` this way — `classification/__init__.py:198` and `:273`, `segmentation/__init__.py:154`
— so every `save_pretrained` of a model built by those constructors is affected. The two
sites that pass `mc["canvit"]` (an already-flattened dict) are fine, which is why the
hand-assembled pretraining layout works.

The in1k HF export was added *because* "an in1k finetune could not be handed to
CanViT-eval". **It still cannot** — it has never worked end to end. `test_to_hf.py` covers
only `is_classifier_checkpoint` dispatch and never round-trips the artifact, so the test
pins the belief rather than the format: the `fixture-drift-hid-real-bug` pattern again.

**(b) The classifier layout carries no metadata at all.** `classifier_to_hf` delegates to
`save_pretrained`, which writes the model config and nothing else:

```
exp33_in1k_400000_hf/config.json  -> metadata keys: []   pretrain_view_scale: null   teacher_name: None
exp32_fovi_1916928_hf/config.json -> metadata keys: [dataset, git_commit, pretrain_view_scale, source_pt, step, teacher_name, timestamp]
```

So both of canvit_eval's footgun-closers are inert on any classifier: `resolve_view_scale`
finds no `pretrain_view_scale` and would silently evaluate a foveated finetune at the
policy's scales (F5 shows that costs 0.11 top1); `teacher_probe_for_model` finds no
`teacher_name` and falls back to ViT-B — correct here only by luck. §1's "this repo writes
both fields while reading neither" was too generous: for in1k it does not write them either.

**Consequence: (a) is a prerequisite, not a stage.** Fix it in core before Stage 2, with a
round-trip test (`build → save_pretrained → from_pretrained → same logits`) rather than a
dispatch assertion. Stage 2 then has a **writer half** as well as the reader port.

#### F7 — `pretrain_view_scale` has two schemas

Local `.pt` metadata records a bare **float** (`2.0`; `distill/task.py:630`,
`ade20k/task.py:484`). The HF `config.json` records the **dict** canvit_eval parses
(`{patcher_name, mode, distribution, fixed_scale, min_scale, max_scale}`), reconstructed by
`extract_pretrain_view_scale` from `training_config_history`.
`resolve_scale_from_metadata` returns "not recorded" for anything that is not a dict, so a
resolver ported as-is is a silent no-op on every local checkpoint. Accept both forms, and
build the fixtures from the real producer of each.

#### F8 — distill's `validate` returns one scalar and logs the rest

`validate()` returns `scene_cos_raw[-1]` and pushes the whole per-timestep
`scene_cos_raw/norm`, `cls_cos_raw/norm` and `in1k_tts_top1` series straight into the
tracker. With `tracker="none"` a standalone eval can capture **only** `val_metric` — which
is why distill's row above has one number and not ten. Stage 3's `harness.evaluate` must
have `evaluate` **return** the series and leave logging to the caller, or the Stage-1b gate
for distill is a single scalar.

#### F9 — Stage 4's reconstruction gate, as written, cannot be met

It says "reconstruction matches distill's `val/scene_cos_raw_t9` on the same checkpoint".
The two compute the same quantity over **different images**: distill validates on a fixed
256-image subset drawn by a seeded permutation of ImageNet-1k val (`val_seed=0`,
`n_val_samples=256`), while `canvit_eval/tasks/reconstruction.py`'s `ImageDirDataset` rglobs
a directory and sorts by filename. Restate the gate: the ported reconstruction, **pointed at
distill's own fixed-subset loader**, matches the distill series. Sharing that loader is the
point of the merge anyway.

#### F10 — Two incidental findings, out of scope, recorded so they are not lost

* **`best.pt` in a multi-job array run is not the global best.** `harness/loop.py`'s
  `best_so_far` is a local initialised to `None` per job, so each array task's first
  validation always rewrites `best.pt`. Demonstrated on exp33: top1 peaked at step 380000
  (0.83714, array task 42) but `best.pt` holds step 400000 (0.83658, task 44). Single-job
  runs (exp30/exp34 ade20k) are unaffected; the in1k arrays (exp25/exp29/exp33) are not.
* **`scripts/eval_ade20k_checkpoint.py` is a Stage-3 precursor for one task.** It already
  does for ade20k what `harness.evaluate` will do for all three, and its header already
  carries the C2F/foveated-scale warning Stage 3 needs. Stage 3 subsumes and deletes it; that
  header is the best available draft of the new CLI's help text.

### Stage 0b — PREREQUISITE: make the in1k HF layout loadable (core) — EXECUTED 2026-08-31

Surfaced by Stage 0, F6(a): `save_pretrained` on any classifier or segmentation model built
by `from_pretrained_with_new_head` / `from_pretrained_with_probe` drops `model_config`, so
`from_pretrained` on the result raises. Three sites in **CanViT-PyTorch** pass
`vars(cfg)` (nested dataclass instances) where a flattened dict is required:
`classification/__init__.py:198`, `:273`, `segmentation/__init__.py:154`.

This is a core fix, so strictly it belongs to Phase 2 (§8) — but it blocks the in1k half of
this merge (canvit_eval cannot load an in1k finetune at all, so there is no cross-repo
number to gate against), so it goes first.

**Done in `CanViT-PyTorch@2679d6b`.** The scope turned out to be wider than F6 read:
the **segmentation** wrapper is affected too, and for uniform models as well as foveated —
i.e. *every* probe or classifier this stack has ever published was unloadable, not just the
in1k branch. `serialize_canvit_config` was added as the explicit inverse of
`rebuild_canvit_config` and the three sites route through it. No runtime behaviour change:
the wrapper ends up with the config `from_pretrained` has always built from a plain dict.

**Gate — passed, all on one machine (F1):**

* 4 new round-trip tests in `tests/test_checkpoint_sources.py` (build → `save_pretrained` →
  `from_pretrained` → same cfg, same tensors, **identical logits**), for the classifier and
  the segmentation wrapper, uniform and foveated. All four fail before the fix.
* `canvit_pytorch` 125 passed (121 + the 4 new); `canvit_train` 313 passed.
* Both Stage-0 rows that run through the changed constructors reproduce **bit-identically**:
  ade20k `fixation_grid` (all ten timesteps) and in1k `fixation_grid` (0.83632 / 0.97006).
* **The check Stage 0 could not run now runs and agrees exactly.** in1k under
  `repeated_full_scene`, unpinned, on the same finetune: top1 **0.72248** / top5 **0.91346**
  in *both* repos — 36124/50000 either way. So the cross-repo agreement established for
  ade20k holds for in1k too, and §2's shared-runner premise is confirmed on both tasks.

F6(b) — the classifier layout carrying no `metadata` block — is **not** fixed here and stays
Stage 2's writer half.

### Stage 1 — Lift the shared episode runner — EXECUTED 2026-09-01

Move `consumes_full_image`, `derive_glimpse_px` and the glimpse loop into
`harness/rollout/`; give it `run_episode`'s shape (return per-step outputs, let callers
read). Rewire ade20k and in1k onto it. Delete the `in1k → ade20k` cross-import.

**distill is deliberately NOT rewired here.** `distill/viz/validate.py::validate` is not a
rollout — it is a whole validation *phase*: teacher targets, the per-timestep cos/recon
series, the IN1k linear-probe readout, curve plots and the PCA figure, each on its own
historical cadence. Treating all three tasks as one move is the premature-unification trap
in §6.

**Stage 1b is CANCELLED — its premise was wrong (checked 2026-09-01).** §2's table claimed
distill owns a third eval rollout. It does not: `validate()` delegates to
`CanViT.forward_reduce` (`canvit_pytorch/model/base/impl.py:422`), a fold over viewpoints
that already lives in **core**. There is no distill-side loop to extract.

What the stack actually has is two loops, split along a line the plan did not see:

| | `CanViT.forward_reduce` (core) | `harness/rollout/episode.py::run_episode` |
|---|---|---|
| shape | fold (`init_fn`/`step_fn`) | list of readouts |
| image per glimpse | the SAME full image every time | a per-viewpoint **crop** |
| modulation | hoisted out of the loop | recomputed per glimpse |
| used by | distill | ade20k, in1k |

The image row is the whole reason both exist, and it is a **patcher convention**, not a task
difference. `UniformPatcher` built with `glimpse_size_px=<N>` crops internally
(`patcher/uniform.py:50`), which is how the pretraining model is built — so distill hands it
the full scene. The downstream wrappers are built with `glimpse_size_px=None`, so the patcher
expects an already-cropped glimpse and the CALLER must crop. `consumes_full_image` /
`derive_glimpse_px` / `sample_at_viewpoint` in `run_episode` exist purely to compensate for
that. `forward_reduce` passes one fixed `image`, so it cannot express the caller-crop case.

The real unification is therefore to give the downstream wrappers a `glimpse_size_px`-aware
patcher so they crop internally like distill, after which `run_episode` collapses into
`forward_reduce`. That is core surgery changing what every downstream wrapper does, and
**its measurable payoff today is zero**: the only behavioural difference is the modulation
hoist, and `vit_modulation.enabled` is `false` in every checkpoint in flight (checked on
exp22's `config.json`), so `token_modulation` is `None` and the hoist is a no-op. Not worth
the risk now. Revisit if modulation is ever enabled — at which point ade20k/in1k eval starts
recomputing it once per glimpse, identical in value and wasteful in time.

Recorded rather than acted on, per §6's premature-unification warning.

Moving `validate.py` out of `viz/` — a validation rollout nested under a *visualization*
subpackage — is a structural fix worth doing, but it is cosmetic and must not ride along
with a numeric change. Do it in its own commit, before or after, never during.

**Stage 1 done.** `harness/rollout/episode.py` holds `consumes_full_image`,
`derive_glimpse_px` and `run_episode`; `ade20k/rollout.py` and `in1k/rollout.py` are now
just their readouts (39 and 69 lines, from 126 and 81). `make_random_viewpoints` moved to
`harness/rollout/eval_viewpoints.py`, which had been reaching into the ade20k package for
it — a second cross-import the plan had not spotted. Both `in1k → ade20k` imports are gone
(`in1k/rollout.py` AND `in1k/task.py`; the plan named one).

`run_episode` takes a `readout` callback rather than returning per-step `CanViTOutput`s the
way `canvit_eval/episode.py` does: both callers keep only a small projection of each step,
and ten full foveated ADE20K states are gigabytes. Same loop, same sharing, memory
unchanged. Closed-loop rollouts (`_policy_rollout`, `_entropy_c2f_rollout`,
`_policy_rollout_cls`) are NOT folded in — their viewpoints depend on the live state, so
they cannot take a precomputed list; they already share `deploy_rollout_viewpoints`'
`advance` callback. `in1k/rollout.py::eval_viewpoints` is a third, now-unused policy
dispatcher reached only by its own test; left alone because the policy surface is Stage 3's
subject, not this stage's.

**Gate — PASSED.** 313 tests (unchanged count), and all four Stage-0 **(GATE)** rows
reproduce **bit-identically**: ade20k `fixation_grid` and `full`+pin 2.0 (ten timesteps
each), in1k `fixation_grid` and in1k `full` unpinned. Fresh before/after on the same MIG
slice, per F1. Artifacts: `stage0_baseline/gate1_*.json`.

**Gate (Stage 1):** the Stage-0 rows marked **(GATE)** reproduce **bit-identically** —
ade20k `fixation_grid` and `full`+pin 2.0 (ten timesteps each), in1k `fixation_grid`
(top1/top5) and in1k `full` unpinned. Per F1 this must be a fresh before-and-after on ONE
machine; per F2 the `random` rows cannot gate anything past t0. This is pure code motion, so
anything that moves at all is a bug introduced here.

### Stage 2 — Close the two footguns — WRITER HALF EXECUTED 2026-09-01, reader deferred to Stage 3

**Resequenced, for a reason found while starting it.** The plan had both resolvers landing
here. But `resolve_view_scale`'s only legitimate caller is standalone eval, which is Stage 3
— wiring auto-resolution into *training-time* validation would silently re-base exp25/exp29/
exp33, which this plan forbids elsewhere. And `teacher_probe_for_model` is largely already
here: `distill/probe.py::PROBE_REGISTRY` maps `teacher_name` to probe repo **and
resolution**, which canvit_eval's `TEACHER_REGISTRY` does not carry. So landing the readers
now would mean committing unused functions. They move to Stage 3, where the caller exists.

What went in instead is the half that has a live defect behind it and a real gate.

**1. Downstream checkpoints now record the view scale, in ONE shape.** New
`checkpoint/downstream_pretrain_view_scale` produces the same dict
`extract_pretrain_view_scale` builds for distill, and both downstream tasks call it. Three
things it fixes, all found by looking at real checkpoints rather than at the plan:

* **in1k recorded the scale NOWHERE** — not `metadata.pretrain_view_scale`, not
  `training_config_history` (whose in1k entries hold only mode/n_timesteps/scene_size/task/
  train_spec). F7 said "two schemas"; the truth was two schemas and one absence.
* **ade20k recorded a bare float**, so a reader written against the published dict form is a
  silent no-op on it. Every pre-2026-09-01 ade20k checkpoint is still in that state, so the
  reader must accept both — `_as_view_scale_dict` does.
* **ade20k recorded it unconditionally**, i.e. `1.0` for a UNIFORM run, whose view scale is
  meaningless. `extract_pretrain_view_scale`'s contract is that `None` means "unknown, never
  read as 1.0"; emitting 1.0 for uniform broke exactly that. Now gated on the patcher.

**2. `to_hf`'s classifier layout carries a metadata block** (F6b). `save_pretrained` records
only `__init__` kwargs, so it is merged into config.json afterwards — verified safe, the
mixin filters config.json to the signature on the way back, which is how the pretraining
layout has always worked. For the scale it tries the payload first, then **the backbone repo
in `model_config["model_repo"]`**, which is what rescues exp25/exp29/exp33 — so this fixes
existing checkpoints, not only future ones. `teacher_name` has no first step at all (a
downstream checkpoint never recorded it) and always comes from the backbone. Both fall back
to `None` and a loud warning naming the measured cost, never to a guess.

**3. The off-scale warning at the point of use** (F5). `open_loop_viewpoints` now warns when
a fixed-scale foveated model's trajectory does not sit at its training scale. It checks the
**generated scales, not the policy name** — a name table goes stale the moment a policy or an
override is added, and it has to stay right for the combinations Stage 3 opens up. Silent for
uniform models and for sampled-scale modes, which are scale-robust by construction. This is
the `documented-drift-still-ships` lesson applied: `HISTORICAL_DEFAULTS`' docstring already
warned about this and it happened anyway.

**Gate — PASSED.**

* 325 tests (313 + 12 new: 5 on the metadata block, 7 on the warning).
* All four Stage-0 **(GATE)** rows still **bit-identical** — the warning sits in the numeric
  path, so this is not a formality.
* **The two repos now agree exactly on the same measurement.** Re-export exp33's finetune
  and ask canvit_eval for one glimpse with no scale argument: it auto-resolves 2.0 from the
  new metadata and reports **top1 0.81258 / top5 0.95806** — identical, to `+0.00e+00` on
  both, to canvit_train's `fixation_grid` at T=1. Before the fix the same command reported
  **0.72580**.

  **That 0.087 is a measurement error, not a gain.** The weights are byte-identical either
  way. 0.72580 was not a measurement of this model at all — it was a measurement of the
  MISMATCH between the model and glimpses at a scale it never trained on, a question nobody
  asked. It is quoted here only as evidence that the missing metadata block was a real
  defect rather than a cosmetic one; a 0.087 discrepancy is hard to wave away. Nothing in
  this stage makes any model better, and the direction of the change is not the point:
  measuring off-scale degradation deliberately is a legitimate experiment, which is exactly
  why the off-scale case WARNS rather than refuses. The goal is that an eval measures what
  was asked for and records what it did — never that its numbers come out high.

**Still open, deliberately:** existing published HF directories do not gain the field
retroactively — they need re-exporting. Nothing in flight depends on one (exp36 pins its own
commit and reads `.pt` files directly).

### Stage 2 (original text, for the record) — Close the two footguns

Port `resolve_view_scale` (auto-derive the eval view-scale from the checkpoint's recorded
`pretrain_view_scale`) and `teacher_probe_for_model` (auto-select teacher + in1k probe from
`teacher_name`, with a registry and a loud fallback).

**This stage has a WRITER half, added after Stage 0.** The premise "both read metadata this
repo already writes" is false for in1k: `classifier_to_hf` publishes an empty `metadata`
block, so both closers are inert on every classifier (F6(b)). And the reader must accept
BOTH recorded schemas — the local `.pt` float and the HF dict (F7).

**Gate:** (a) a run passing the scale explicitly is unchanged; (b) a run passing nothing
resolves to the same value and logs the decision; (c) pre-metadata checkpoints stay
unchanged (no-op without the field); (d) an in1k checkpoint round-tripped through `to_hf`
now carries `pretrain_view_scale` + `teacher_name`, and the resolver fires on it — checked
against the F5 numbers, which are what "inert" costs: 0.114 top1 on in1k, 0.128 mIoU on
ade20k.

### Stage 3 — `harness.evaluate` — EXECUTED 2026-09-02 (3a policy surface, 3b entry point)

`python -m canvit_train.harness.evaluate <distill|ade20k|in1k> --opts.ckpt … --cfg.…`, tyro
subcommands over each task's own config dataclass. It builds the task, restores the
checkpoint strictly, takes the val loader from the new `task.build_val_loader()` — the SAME
one `build_loaders` hands the training loop — and calls `task.evaluate`. Nothing here is a
parallel implementation, which is why it cannot drift from the numbers a run logs.

`build_val_loader` is the seam that makes that true, and it removed real duplication: the
ade20k val loader is now built in one place instead of inside `make_ade20k_loaders`, so
standalone eval no longer has to construct the train split just to reach the val set.

**`auto` is refused** for a standalone measurement, with a message naming the hazard
concretely. `HISTORICAL_DEFAULTS` stays the *training* default, untouched.

**One config per invocation.** `scripts/eval_ade20k_checkpoint.py` is deleted — it was this
for one task. Its live consumer `scripts/eval_ade20k_c2f.sh` now loops the four arms through
`harness.evaluate`, which is exactly where a loop belongs: the owner's own orchestration, in
the shell, one artifact per measurement.

**Gate — PASSED.** 340 tests, and standalone reproduces the Stage-0 baseline
**bit-identically on all three tasks**: ade20k `fixation_grid` and `full`+pin 2.0 (ten
timesteps each), in1k `fixation_grid`, distill `val_metric`. That is the gate this stage was
always going to be judged on, and the numbers came from the pre-existing drivers, so the new
entry point is proven against code it replaces. Artifacts: `stage0_baseline/gate3b_*.json`.

**3c — the readers and F8, done 2026-09-02.**

*The readers do not pin anything.* Stage 2 deferred them here and the framing changed on the
way: which trajectory to measure is the user's choice and this code does not make it (owner,
2026-09-02). What a checkpoint knows is the scale it was pretrained at and the teacher that
supervised it, and making the user retype either is how they get mistyped — a mistyped view
scale being the silent failure this whole merge exists to close.

So `adopt_checkpoint_provenance` fills in `foveated_scale` and `teacher_name` **only where
the user left the dataclass default**, logs what it adopted, and records it in the artifact.
Pinning stays explicit via `--cfg.eval-override-scale`. The payoff is that the off-scale
warning now compares against the model's REAL training scale instead of a config default of
1.0 — so `--cfg.eval-policy full` on exp34 with nothing else typed says "pretrained at 2.0
but this trajectory uses 1" and names the remedy, where before it had no way to know.

`read_pretraining_provenance` is shared with `to_hf` (one resolution order: the payload's own
record, including ade20k's legacy float, then the backbone repo — the only route for
exp25/exp29/exp33). `teacher_name` feeds `distill/probe.py::PROBE_REGISTRY`, which already
carried the probe repo AND its resolution, so canvit_eval's `teacher_probe_for_model` needed
no port beyond this read.

*F8.* `validate` now RETURNS its scalars and the caller logs them. The key names are
character-for-character what it used to log, and distill declares `metrics_prefix = "val"` so
`run.py` keeps the historical namespace — otherwise every exp22–exp32 dashboard would break.
A standalone distill record went from **1 metric to 56** (per-timestep `scene_cos_raw/norm`,
`cls_cos_raw/norm`, `in1k_tts_top1`, plus the teacher probe top-1).

**Gate — PASSED.** 347 tests, and all four standalone rows still **bit-identical** to the
Stage-0 baseline. F8 rewrote distill's return path, so that was not a formality.

**Stage 3 is complete.**

### Stage 3 (original text, for the record) — `harness.evaluate`

`python -m canvit_train.harness.evaluate <distill|ade20k|in1k> --cfg…`, mirroring
`harness.run`: tyro subcommands over each task's own config dataclass, which is already
this repo's idiom (`harness/cli.py`) and canvit_eval's. Expose the three missing policies.

**No default policy — require an explicit choice.** The default table
(`HISTORICAL_DEFAULTS`) stays for *training-time* validation, where it protects
comparability with exp22–exp36; standalone eval refuses to guess. The error message must
name the hazard concretely (a foveated model under a scale-varying policy such as
`coarse_to_fine` is out of distribution — mIoU *falls* as glimpses accumulate). A docstring
would not have caught the in1k/foveated case; the message at the point of use might.

Training-time validation and standalone eval share the episode runner and differ only in
config: subset vs full val set, glimpse count, DDP vs single-GPU. Nothing stops X glimpses
in training and Y standalone.

#### The policy surface: orthogonal axes, named presets (owner decision, 2026-08-31)

F3 forced the question "what does `--policy random` mean" and the answer is that `random`
was never one thing. It bundles independent choices. Decompose them:

**Corrected 2026-09-02, after reading the generators.** The first draft of this section
listed `center` and `scale` as independent axes. They are not: `random_viewpoints` draws
`centers = rand * (1 - scale)`, i.e. the safe-box center law is a JOINT distribution over
(center, scale), and the quadtree's centers and scales both come off the same tree. A
four-axis table looks tidier and lies. What is actually there:

| | what it is | values |
|---|---|---|
| **preset** | the joint (center, scale) trajectory — the thing that has a name and a published number | `coarse_to_fine` · `fine_to_coarse` · `random` · `full` · `fixation_grid` · `entropy_coarse_to_fine` · `policy` |
| `t0` | whether the run opens on the full-scene anchor, where that is not already the trajectory's own first element | `full_anchor` · `trajectory` · unset |
| `order` | quadtree only | coarse→fine · fine→coarse (exposed as two presets) |
| `override_scale` | post-hoc pin of every scale, keeping the generated CENTERS | float · unset |

`override_scale` is a **modifier, not an axis**: it replaces scales that were already drawn,
so canvit_eval's `random`+pin is "safe-box centers at their natural scales, then re-scaled" —
a coherent and useful measurement (Stage 0 tracked it against canvit_train's foveated
`random` to ~1e-3 from t1 on), but not "safe-box centers designed for scale 2.0". Saying so
matters: a reader who believes the axes are independent will over-trust an exotic combination.

So the surface is **the preset names people already use, plus three modifiers** — no new
vocabulary, no rename of the flags in ~50 launchers, and every historical protocol still
named by one word. `eval_override_scale` stops being ade20k-only (F4) and becomes the
modifier on all three tasks.

**Presets stay the primary interface.** `--eval-policy <name>` expands to an axis tuple;
any axis is individually overridable on top. The named presets are exactly the trajectories
that already have numbers, and each one is **gated on expanding to a bit-identical result**
— that is what makes "this config reproduces the exp33 protocol" a test rather than a claim.

**Any combination the code can execute is allowed.** Untested is not forbidden: it warns,
runs, and gets its resolved tuple recorded. Two hard errors only:

1. **A flag that would be silently ignored.** A `center` law passed alongside a closed-loop
   policy (`policy`, `entropy_coarse_to_fine`) does nothing — the center comes from the
   scorer or the entropy map. Same for `order` on non-quadtree centers. Accepting those
   quietly is worse than refusing: you get a number and believe it came from the config you
   typed. Note closed-loop is only PARTLY non-decomposable — the `scale` axis still applies,
   and for foveated it is already pinned to the training scale (what exp36 does).
2. **A genuine contradiction**, e.g. the scale pinned to two different values.

Off-scale foveated is NOT an error. It warns with the measured cost (F5: −0.114 top1 on
in1k, −0.128 mIoU on ade20k) and stamps the mismatch into the output — because measuring OOD
degradation is a legitimate experiment, and because `HISTORICAL_DEFAULTS["in1k"]` keeps that
default on purpose to stay comparable with exp25/exp29/exp33.

**No protocol is fixed, for training-time validation either (owner, 2026-09-02).** Many
validation settings are adequate and picking one for the user is not this code's job. What
it owes them is visibility: the RESOLVED axis tuple goes into the validation log line and
into every output record, the off-scale warning names the remedy with the right value for
that specific model, and standalone eval refuses to guess. Note the historical default was
never actually reached for the case that worried us — exp25/exp29/exp33 all set
`CFG_EVAL_POLICY=random` with `fixed-scale 2.0` explicitly, so every foveated in1k curve was
in-distribution already. The unpinned-C2F entry is a latent default, not a live one.

**One config per invocation (owner + design, 2026-09-02).** `harness.evaluate` takes ONE
policy and writes ONE artifact. It was tempting to accept a list, since the model and dataset
load once per process — but that is an efficiency argument, the weakest reason to complicate
an interface, and everything else opposes it: a list forces a second CLI mode and a nested
output schema, leaves failure semantics murky when config 3 of 4 dies, and a loop over
policies inside the process is precisely the small orchestrator `batch.py` was dropped for.

The reuse seam is the **library**, not the CLI: `evaluate(...) -> dict` is a clean function,
so comparing four policies on one checkpoint is a short loop in a notebook — which is where
the owner said results get inspected. If load time ever dominates a real workflow, the answer
is a documented library-level loop, not a CLI mode.

**Provenance, not restriction, is what protects comparability.** Every eval output records
the fully resolved axis tuple and whether it matched a known preset. That is what makes "is
this number comparable to exp33?" answerable later, for any combination however exotic; and
it makes promoting a combination worth keeping into a preset a one-line change with a
recorded reference number.

The `random` collision resolves as: keep canvit_train's meaning under the historical name
(every published number was measured under it), and reach canvit_eval's via the axes
(`center=safebox`, `t0=random`) or its own preset name.

Three further requirements Stage 0 added:

* **`--eval-override-scale` on all three tasks** (F4) — today only ade20k has it, and in1k,
  the one task whose historical default is the OOD footgun, cannot express the remedy.
  Unify the knob, keep each task's default (`unify-the-knob-not-the-defaults`).
* **`evaluate` must RETURN the per-timestep series, not log it** (F8) — distill currently
  returns one scalar and pushes ten series into the tracker, which is why Stage 0's distill
  baseline is a single number.

Also fix the `HISTORICAL_DEFAULTS` docstring here (F5): its "(or `full`)" advice is
in-distribution only with an explicit pin.

**Gate:** standalone reproduces training-time validation exactly when handed the same
checkpoint, policy, glimpse count and val subset — on ONE machine (F1), under a
deterministic policy (F2).

### Stage 4 — The eval-only capabilities — 4a (reconstruction) EXECUTED 2026-09-02

**Reconstruction was not ported, and must not be: canvit_eval's version is WRONG.**

The plan called for "reconstruction as ONE implementation with two entry points", treating
it as the canary for whether the refactor is working. It caught something. The two
implementations do not compute the same quantity under the name `scene_cos_raw`:

| | what it compares |
|---|---|
| `distill/validate.py` | `cos(scene_normalizer.destandardize(pred), raw_teacher)` |
| `canvit_eval/tasks/reconstruction.py` | `cos(pred, raw_teacher)` — no destandardize anywhere in the file |

Measured on exp32-fovi step-1916928, one batch of 64 at t9
(`stage0_baseline/recon_diff.py`):

```
scene_cos_raw  canvit_eval (no destandardize) = 0.647881
scene_cos_raw  canvit_train (destandardized)  = 0.926994   difference +0.279
scene_cos_norm (identical in both)            = 0.857282
```

**distill's is correct**, established two independent ways
(`stage0_baseline/recon_space.py`).

*From the training code:* the target is `scene_norm(raw_patches)` and the MSE loss compares
`scene_pred` against it, so `predict_teacher_scene` is trained to output NORMALIZED features.

*From the tensors themselves*, without reference to that code — per-channel moments:

| tensor | std | ‖vec‖ |
|---|---|---|
| raw teacher patches | 0.329 | 11.58 |
| normalized teacher (`scene_norm`) | 0.988 | 27.39 |
| **`predict_teacher_scene` output** | **0.849** | **23.58** |
| `destandardize(pred)` | 0.291 | 10.79 |

The prediction sits with the NORMALIZED teacher; destandardizing moves it onto the raw one.

And the cosine pattern is symmetric, which is the clincher:

```
pred                vs norm_teacher   0.857   matched
destandardize(pred) vs raw_teacher    0.927   matched
pred                vs raw_teacher    0.648   MISMATCHED (canvit_eval)
destandardize(pred) vs norm_teacher   0.599   MISMATCHED
```

Both mismatched pairings are low. Had `pred` been in raw space the pattern would invert.

**Nothing of this repo's own history is affected.** `scene_cos_raw = 0.927` is the
destandardized comparison, i.e. what distill has logged since exp22 — the correct number all
along. Only canvit_eval's separate reconstruction task crossed the spaces, and it is the
thing being retired.

So any `scene_cos_raw` / `cls_cos_raw` published from canvit_eval's reconstruction task is
understated. `*_cos_norm` from it is fine.

**What actually shipped.** Nothing was ported: after F8 made `validate` return its series,
`python -m canvit_train.harness.evaluate distill` IS the reconstruction task, in its correct
form. The only capability canvit_eval had that distill validation lacked was the image
SOURCE — its `ImageDirDataset` rglobs an arbitrary folder, whereas `IndexedImageFolder`
needs class subdirectories and ADE20K's val images are flat. So `--cfg.val-image-dir` was
added, with `FlatImageDir` (recursive, filename-sorted, label `-1`).

That `-1` exposed a second defect: `labels_are_in1k` checked only the UPPER bound (it was
written to tell IN1k from IN21k), so `-1` passed and the IN1k probe readout would have
reported an accuracy against garbage labels instead of being skipped. Now bounded both ends.

Verified end to end — reconstruction on ADE20K val images, scale adopted from the
checkpoint, IN1k readout correctly absent (`stage0_baseline/stage4_recon_ade20k.json`):
`scene_cos_raw` 0.887 → 0.918 across t0→t9, below the 0.926 it reaches on ImageNet val,
which is the domain shift one would expect.

**Gate — PASSED.** 351 tests, four standalone rows still bit-identical.

**4b — `ade20k-seg-dinov3`, the teacher baseline, ported 2026-09-02.**
`harness.evaluate ade20k-dinov3`: the DINOv3 teacher's own ADE20K score, one passive forward,
mIoU at t0 — the reference line the CanViT numbers are read against. It has no checkpoint and
no episode (hence its own opts rather than `EvalOpts`), but it shares
`make_ade20k_val_loader` and `eval_probe_on_batch`, so the baseline and the model it bounds
are measured identically rather than merely similarly.

**Gate — PASSED, bit-identical.** Both implementations over the full 2000-image val set with
a shared probe: `0.0006143441136078409` on each side, Δ `+0.000e+00`. The probe is a
*synthetic* one, and deliberately so: a random probe is the correct instrument for an
equivalence test, since it proves the two implementations compute the same function without
needing a good one. 354 tests.

**Recorded while doing it: no probe in this stack can read DINOv3 features.** Both cached
ADE20K probes (`probe-ade20k-40k-s512-c32/c64-in21k`) are 1024-d and `feat_type:
canvas_hidden`, i.e. trained on CanViT CANVAS features; DINOv3-B/16 patches are 768-d. So a
*real* baseline number needs a probe trained on teacher features that nobody here has
published. The capability is ported and correct; the artifact to feed it does not exist yet.

**4c — per-row IoU, done 2026-09-02.** `--cfg.per-row-iou-out <path.parquet>` on the ade20k
eval writes one row per (timestep, image, class) with raw `inter_px` / `union_px` /
`gt_area_px`. A dataset mIoU says how good the model is; these say on WHICH images and
classes, and how that moves as glimpses accumulate — the question a glimpse policy is
actually about, and one an aggregate cannot answer.

Only the METRIC was ported, as §3 decided — `ade20k_obj`'s three-stage cached-feature
pipeline stays behind. The rows ride the predictions `eval_probe_on_batch` already computes
(it now returns them instead of discarding them), so the cost is one scatter-add per batch
and no extra forward. Off by default. Parquet via **pyarrow**, a real dependency here;
canvit_eval used pandas, which sits in its *dev* group while `ade20k_obj/iou.py` imports it
at module level — so that task only ever ran in a dev install.

Counts are stored, never a per-row ratio: `inter/union` is undefined for a class absent from
an image, and averaging per-image ratios is not the dataset mIoU.

**Gate — PASSED.** On exp34's probe the rows aggregate to the accumulator's `miou_t{t}` at
**all ten timesteps to ≤1.1e-16**, i.e. summation order only. Plus a `np.bincount` reference
test for the scatter-add itself — which is `canvit_eval/tests/test_iou_equivalence.py`, so
Stage 5 has one less item. 360 tests, four bit-identity rows unmoved (`eval_probe_on_batch`
is in the numeric path, so that mattered).

Writing it surfaced one real defect in the port: `miou_from_rows` summed float32 counts and
drifted ~1e-9 from the accumulator. Counts are integers; it sums in float64 now.

**Stage 4 is complete.**

### Stage 4 (original text, for the record) — The eval-only capabilities

`reconstruction` as a **single** implementation with two entry points (distill validation
already computes cosine-to-teacher; two implementations would recreate the duplication this
merge exists to remove — treat this as the canary for whether the refactor is working).
Then `ade20k-seg-dinov3`, then the per-row IoU output.

**Gate:** reconstruction, **pointed at distill's own fixed-subset val loader**, matches
distill's `val/scene_cos_raw_t9` on the same checkpoint (0.925767719745636 at step 1916928).
The unqualified version of this gate is unmeetable — the two paths cover different images
(F9). The DINOv3 baseline matches canvit_eval's number; per-row IoU aggregates to the same
dataset mIoU the existing path reports.

### Stage 5 — Tests and bench — EXECUTED 2026-09-02

**`test_iou_equivalence.py`** came in with Stage 4c, alongside the metric it covers.

**`test_view_scale.py` ported by CASE, not verbatim.** Its subject,
`resolve_scale_from_metadata`, returned a scale to PIN; this repo adopts the training scale
into the config and leaves pinning explicit (owner, 2026-09-02), so a verbatim port would
test semantics that deliberately no longer exist. Every situation it enumerated still has to
be right, and four were uncovered: the **square** patcher (scale-sensitive for the same
`fix_size = scale * H` reason and the one likeliest to be forgotten), multi-scale modes
(`per_rollout` / `per_glimpse` — adopting the *mode* is what keeps the off-scale warning
quiet for a scale-robust model), a checkpoint with no metadata at all, and `mode=fixed` with
no value recorded. 365 tests.

**`bench/pt/` went to `CanViT-PyTorch`, not here.** It benchmarks `canvit_pytorch` and its
only non-core import was two DINOv3 repo ids, so hosting it in the *training* repo would
invert the dependency exactly the way §8 warns about for eval. Committed as
`CanViT-PyTorch@3a0dcc2`.

**It did not run.** `run.py` called `CanViT.forward(glimpse=...)`, but bare `CanViT` takes
`image=` — only the downstream wrappers use `glimpse=`. Every CanViT cell raised `TypeError`,
so it had been broken against current core for some time. Which is what a benchmark with no
stored baseline looks like from outside: indistinguishable from one nobody runs.

`matrix.py` was KEPT despite being a subprocess sweep. The objection to `batch.py` and to a
multi-config eval CLI was duplicating orchestration `slurm/runs/` already has; this is GPU/CPU
idle gating and CPU-topology-aware thread pinning, which a shell loop cannot do and nothing
here duplicates.

Baseline committed from this MIG slice, 300 iterations per cell — `canvit` 512px cg32
amp-bf16 **11.08 ms** (CI 11.08–11.09), `dinov3-vitb16` 512px amp-bf16 **8.98 ms**. A latency
baseline is valid ONLY on the hardware that made it, so the device is in the filename, the
JSONL meta and the README — F1's machine-local rule, and more binding for timings than for
metrics.

**Stage 5 is complete.** Only Stage 6 (docstrings + retire) remains.

### Stage 5 (original text, for the record) — Tests and bench

Port `test_view_scale.py` and `test_iou_equivalence.py`. Port `bench/pt/` (its only
canvit_eval dependency is two DINOv3 repo constants) and **commit a baseline JSONL** — it
has no stored baseline, so as shipped it measures "now" rather than "did this regress".
It is complementary to `unification_docs/throughput_ab.py`, which measures training
throughput, not inference latency.

### Stage 6 — Retire

Rewrite the **15 source docstrings** that cite `canvit_eval` as the behavioural
specification (*"Routing follows canvit_eval/episode.py"*, *"canvit_eval's rule"*, *"the
resize canvit_eval uses"*) — otherwise they become pointers into an archive, the same
staleness pattern as the old `CanViT-pretrain` paths. Then mark `CanViT-eval` read-only in
`CLAUDE.md` alongside `CanViT-specialize` and `CanViT-PyTorch-RL`, and update memory.

## 6. Risks

**A gate is only a gate on one machine.** Stage 0 measured a ~1e-5 spread on a fully
deterministic quantity between a full A100 and a MIG slice of one (F1). Every stage's gate
therefore means "re-run before and after, here, and demand exact equality" — never "compare
to the number in the log".

**The regression surface is metric values, not exceptions.** Precedent: `68b635f` fixed an
argmax/upsample ordering and re-based **every** earlier ADE20K mIoU by ~0.19. Nothing
crashed; the numbers were simply wrong for months. Unit tests cannot catch this class of
change — only comparing values on a real checkpoint can, which is why every stage has a
numeric gate and why Stage 0 exists.

**Premature unification is the specific failure mode.** The three eval paths differ in ways
that are load-bearing and currently tested: ade20k's reduction order, distill's teacher
targets, in1k's top-k. Forcing one generic runner over the *reductions* would bury those.
Share the loop only.

**exp36 timing.** Its ten seeds are pinned to `716051a`, so they are immune to this work.
But their numbers are measured under today's eval; if the gate ever fails and is accepted
as a rebase, exp36 stops being comparable to exp30/34/35.

**Two dataset-path mechanisms** must be reconciled: canvit_eval resolves `ADE20K_ROOT` /
`IMAGENET_VAL` with known-path fallbacks; this repo uses `.envrc.grete` exports.

## 7. Corrections to earlier claims about canvit_eval

Recorded because they were stated confidently and were wrong:

- **It has tests** — 348 LOC in `tests/`, not zero. Only the in-package `test_*.py` count
  is zero.
- **The three "extra" policies are not its code** — all five static generators already live
  in `canvit_pytorch.policies`; it merely exposes them, and `full_then_random` is
  `random_viewpoints(start_with_full_scene=True)`.
- **Its `EntropyGuidedC2F` is the *unguarded* copy.** This repo's port is pinned by
  `harness/tests/test_entropy_c2f.py` and validated to max Δ0.05 against paper Table 4.
  Deleting canvit_eval loses an independent second implementation, not a live guard.
- **It writes `.pt`, not JSON sidecars** — json+parquet appear only in `ade20k_obj`.

## 8. Phase 2 — core, and the rename

`CanViT-PyTorch` merges **after** this, never before: with eval still in a separate repo,
moving core into canvit_train would force eval to depend on the *training* repo, which is
worse layering than today. Then rename the repo/package to `canvit`.

Known costs: commit pinning loses an axis (`TRAIN`/`PYTORCH`/`FOVI` → two), historical
launchers keep resolving because the old repos remain, and upstream merges become
impractical — currently free, since `upstream/main` is 0 commits ahead of both forks.
Published HF checkpoints are unaffected: `config.json` records architecture only, no module
paths; `library_name` / `repo_url` are cosmetic model-card metadata.
