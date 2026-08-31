# 20 — Merging CanViT-eval into CanViT-train

**Date:** 2026-08-31. **Nature:** plan, not a record. Nothing below has been executed.
**Goal:** retire `CanViT-eval` entirely; one repo does all training and all evaluation.
`CanViT-PyTorch` follows in a second phase (§8), after which the repo is renamed `canvit`.

## 1. Why, in one paragraph

Not tidiness. `EntropyGuidedC2F` exists in both repos, 108 diff lines apart, and one of
them produced a published number. `in1k/rollout.py` imports `consumes_full_image`,
`derive_glimpse_px` and `make_random_viewpoints` from `..ade20k.rollout` — a task reaching
into another task's package, in violation of this repo's own layout rule, purely because
there is no shared episode runner. And this repo *writes* `pretrain_view_scale` and
`teacher_name` into every published checkpoint while reading neither; `canvit_eval` is the
only consumer, which is how the exp22 eval-scale break happened. The merge fixes three real
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

### Stage 0 — Capture the baseline (no code changes)

Record every eval number the current code produces on these three checkpoints, both paths
where both exist. Write it into this file as a table. **Everything after this is checked
against it.** Checkpoints chosen 2026-08-31 (owner) — the arms this repo was most recently
validated on, all foveated, which is the harder case:

| task | checkpoint |
|---|---|
| distill | `logs/jon_exp32_pretrain_lrdrop/exp32-fovi/checkpoints/latest.pt` (= `step-2007040.pt`) |
| in1k | `logs/jon_exp33_in1k_finetune/in1k-fovi-ti-1196k/checkpoints/latest.pt` (= `step-401408.pt`) |
| ade20k | `logs/jon_exp34_ade20k_probe/ade20k-fovi-ti-1196k/checkpoints/best.pt` |

The ade20k one is also exp36's reward model, so its `miou_t0` is independently known
(0.377) — a free cross-check on the harness before any of this starts.

Also run `canvit_eval` on the same checkpoints and record whether the two repos agree.
Expectation: they do — `preds_from_logits` upsamples-then-argmaxes with a docstring
cross-referencing `canvit_eval/tasks/ade20k_seg.py`, which does `F.interpolate` then
`.argmax(dim=1)`, and both use core's `mIoUAccumulator`. If they disagree, **stop and
diagnose before porting anything**: find where the difference comes from and decide which
is correct. An unexplained disagreement here invalidates the gate for every later stage.

### Stage 1 — Lift the shared episode runner

Move `consumes_full_image`, `derive_glimpse_px` and the glimpse loop into
`harness/rollout/`; give it `run_episode`'s shape (return per-step outputs, let callers
read). Rewire ade20k and in1k onto it. Delete the `in1k → ade20k` cross-import.

**distill is deliberately NOT rewired here.** `distill/viz/validate.py::validate` is not a
rollout — it is a whole validation *phase*: teacher targets, the per-timestep cos/recon
series, the IN1k linear-probe readout, curve plots and the PCA figure, each on its own
historical cadence. Only the glimpse loop inside it is shareable. Extracting that is a
second, separately-gated step (**Stage 1b**) precisely because the surrounding phase is
where distill's task-specific correctness lives. Treating all three tasks as one move is
the premature-unification trap in §6.

Moving `validate.py` out of `viz/` — a validation rollout nested under a *visualization*
subpackage — is a structural fix worth doing, but it is cosmetic and must not ride along
with a numeric change. Do it in its own commit, before or after, never during.

**Gate (1 and 1b):** every Stage-0 number reproduces exactly. This is pure code motion;
anything that moves is a bug introduced here.

### Stage 2 — Close the two footguns

Port `resolve_view_scale` (auto-derive the eval view-scale from the checkpoint's recorded
`pretrain_view_scale`) and `teacher_probe_for_model` (auto-select teacher + in1k probe from
`teacher_name`, with a registry and a loud fallback). Both read metadata this repo already
writes in `checkpoint/to_hf.py`.

**Gate:** a run passing the scale explicitly is unchanged; a run passing nothing now
resolves to the same value and logs the decision. Pre-metadata checkpoints stay unchanged
(the resolver is a no-op without the field).

### Stage 3 — `harness.evaluate`

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

**Gate:** standalone reproduces training-time validation exactly when handed the same
checkpoint, policy, glimpse count and val subset.

### Stage 4 — The eval-only capabilities

`reconstruction` as a **single** implementation with two entry points (distill validation
already computes cosine-to-teacher; two implementations would recreate the duplication this
merge exists to remove — treat this as the canary for whether the refactor is working).
Then `ade20k-seg-dinov3`, then the per-row IoU output.

**Gate:** reconstruction matches distill's `val/scene_cos_raw_t9` on the same checkpoint;
the DINOv3 baseline matches canvit_eval's number; per-row IoU aggregates to the same
dataset mIoU the existing path reports.

### Stage 5 — Tests and bench

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
