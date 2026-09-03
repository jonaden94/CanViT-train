# 21 — Merging CanViT-PyTorch into CanViT-train (Phase 2)

**Date:** 2026-09-03. **Nature:** plan. Nothing below has been executed.
**Goal:** one repo holds the model and everything that trains or evaluates it; then rename it
`canvit`. Phase 1 (eval) is complete — `unification_docs/20-eval-merge.md`.

**Owner decision, 2026-09-03: upstream `m2b3` will never be pulled again.** §8 of doc 20
listed that as this phase's main cost. It is not a cost.

## 1. What this is, and what it is not

This is a **packaging** change. Not one line of model or training code needs to change, and
if any numeric output moves, that is a bug introduced here — the same standing this repo gave
Stage 1 of the eval merge.

That makes the risk profile unusual and worth stating plainly: the danger is not wrong
numbers, it is **two copies of `canvit_pytorch` on `sys.path` and the wrong one winning,
silently**. §5 is mostly about that.

## 2. The surface, measured

| | count |
|---|---|
| launchers pinning `PYTORCH_COMMIT` | **116** |
| distinct core commits pinned by live `slurm/runs/` launchers | 4 (`d616b7b`, `1f5121b`, `017ce9b`, `3277048`) |
| files in `canvit_train` importing `canvit_pytorch` | **51**, across ~12 submodules |
| `canvit_train` tests runnable with NO GPU | **361 of 365** |
| the 4 that are not | `test_task_digests.py` — they assert GPU-recorded hashes |
| core's own suite | 125 (needs `fovi`, so run it from `CanViT-train/.venv-cu126`) |

## 3. Architecture: ONE package, `canvit`, with core under `canvit/core/`

`canvit_train` → **`canvit`**, and everything that was `canvit_pytorch` moves under
**`canvit/core/`**. So `canvit.core.model`, `canvit.core.patcher`, `canvit.core.teacher`
alongside `canvit.harness`, `canvit.distill`, `canvit.ade20k`, `canvit.in1k`.

**This replaces an earlier draft of this section that argued for two top-level packages
(`canvit_train/` + `canvit_pytorch/`) side by side. The owner rejected it, correctly.** The
reasoning is recorded because two of the three arguments for the rejected design were bad,
and the same mistakes are easy to make again:

* *"Zero import churn"* — a one-time TRANSITION cost dressed as an architecture property. 51
  files is a mechanical rewrite and it is the cheapest kind to verify: imports fail at import
  time, the loudest failure mode there is, with 361 CPU tests catching it. It should not drive
  a decision the owner lives with for years.
* *"Layering stays grep-checkable"* — **not true as a differentiator.** With subpackages it is
  one grep either way:
  `grep -rn "from canvit\.\(harness\|distill\|ade20k\|in1k\)" canvit/core/`.
* *"The rename decouples"* — an argument for DEFERRING the rename, not against doing it. The
  rename was in the owner's original ask.

And the case for one package, underweighted at first: `canvit_pytorch`'s `_pytorch` suffix
encodes a framework split that no longer exists (there was a TPU sibling once), and two
top-level packages in one repo permanently announce "these were once two projects" when the
end state is one.

**`core`, not `model`.** Core is not just the model — it carries `patcher/`, `teacher/`,
`backbone/`, `policies/`, `probes/`, `metrics.py`, `preprocess/`, `viewpoint/`, `rope/`,
`standardizers/`, `data/`. And it already HAS a `model/` inside it, so `canvit/model/` would
become `canvit/model/model/`.

### Two facts checked 2026-09-03 that make this safe

**No name collisions.** Core's 25 top-level entries and `canvit_train`'s five (`ade20k`,
`checkpoint`, `distill`, `harness`, `in1k`) are disjoint — only `__init__.py` overlaps. So
even a fully flat merge would have worked; `core/` is chosen for the layer boundary, not
forced by collisions.

**No checkpoint pickles a project class, so the rename breaks nothing.** Disassembling
`best.pt` and `step-1916928.pt`: the only `GLOBAL` opcodes are `collections OrderedDict`,
`torch._utils _rebuild_tensor_v2`, `torch FloatStorage`, `torch LongStorage`. The single
`canvit_train` occurrence is a plain provenance STRING
(`/local/jobs/…/canvit_train/harness/run.py`), not a class reference. This was the one thing
that could have made the rename expensive — every existing checkpoint unloadable — and it
does not apply. Re-check with `pickletools.dis` if the checkpoint schema ever gains an
object field.

The four read-only repos (`CanViT-specialize`, `CanViT-eval`, `CanViT-PyTorch-RL`, and core
itself) import `canvit_pytorch` from their OWN venvs, which point at the old clone that stays
on disk (§4). They are unaffected by any renaming here.

`fovi` does **not** fold in — same conclusion as doc 20 §3. Pinning therefore goes from three
axes to two (`TRAIN_COMMIT` + `FOVI_COMMIT`), as doc 20 §8 predicted.

## 4. Pinning: CanViT-PyTorch stays on disk, read-only

116 launchers `git archive` a core commit out of `../CanViT-PyTorch/.git`. Delete that clone
and 116 historical runs stop being reproducible.

So it joins the three repos that are **already** read-only references. That costs nothing —
the pattern exists, `CLAUDE.md` already documents it, and `harness_train.sbatch` already
treats `PYTORCH_COMMIT` as optional (`if [ -n "${PYTORCH_COMMIT:-}" ]`). Old launchers keep
their pin and keep working; new ones simply omit it.

**One new hazard needs a guard.** The sbatch prepends the snapshots in this order, so the
final `PYTHONPATH` is `fovi : CanViT-PyTorch : CanViT-train`:

```
export PYTHONPATH="$_CODE_DIR/CanViT-train${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH="$_CODE_DIR/CanViT-PyTorch${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH="$_CODE_DIR/fovi${PYTHONPATH:+:$PYTHONPATH}"
```

A pinned old `CanViT-PyTorch` snapshot therefore **shadows** the `canvit_pytorch` inside a
post-merge `CanViT-train` snapshot. For an old launcher reproducing an old run that is exactly
right. For a NEW launcher that sets `PYTORCH_COMMIT` out of habit it is a silent time-travel
bug: today's trainer against last month's model code.

Guard: when the pinned `canvit_train` snapshot **contains** `canvit_pytorch/` *and*
`PYTORCH_COMMIT` is set, say so loudly and name which core actually wins. That mirrors the
`_PKG` auto-detection already in that file for the `canvit_pretrain` rename — detect the
situation, state it, do not guess.

## 5. The two shadowing traps, which are the real risk

**Trap A — the editable-install `.pth`.** `.venv-cu126` resolves core through
`_editable_impl_canvit_pytorch.pth`, whose single line is
`/…/repos/CanViT-PyTorch`. `.pth` files are processed in **alphabetical order**, so
`_editable_impl_canvit_pytorch` is read *before* `_editable_impl_canvit_train`. Move core into
the CanViT-train tree without re-syncing and every process in that venv keeps importing the
OLD clone — which still exists, still imports, and is one commit behind forever.

There is no error. The only detection is to ask:

```python
import canvit_pytorch; print(canvit_pytorch.__file__)
```

That assertion goes in the test suite, not in a checklist.

**Trap B — `PYTHONPATH`**, §4 above. Same failure mode, different mechanism.

Both traps share a shape worth naming: after this merge, "which `canvit_pytorch` am I running"
stops being a question with one obvious answer. Every stage below ends by answering it
explicitly.

## 6. Stages

### Working arrangement (decided 2026-09-03, no GPU available)

**All of this happens on a branch, `phase2-core-merge`, and `main` is not touched until P3
passes.** The risk of building without the numeric gate is not that the code is wrong — it is
that someone pins or submits against a commit that was never gated. A branch removes that
outright, and `main` keeping the pre-merge tree is also what lets P0 be recorded LATER: when a
GPU appears, record P0 from `main` and the same measurements from the branch, then compare.
So P0 is no longer an ordering constraint, only a merge precondition.

Cost of deferring, named so it is managed: the diff accumulates unverified. Mitigation is to
keep P1/P2 as several small commits each with its CPU gate recorded, so a later numeric
failure is bisectable rather than a haystack.

**A V100 is enough** when one appears. `.venv-cu126` is the sm_70-compatible build — the
pyproject calls it "cu126 (V100+A100)". But V100 is pre-Ampere, so **there is no bfloat16**
and the harness falls back to float16 with a warning (`run.py`: "bfloat16 needs sm_80+").
V100 numbers therefore will NOT match the A100/MIG figures in doc 20's Stage-0 tables. That
is fine here because every gate below is same-machine before/after — but it does mean P0 must
be re-recorded on whatever GPU is used, which F1 required anyway.

### P0 — Re-record the baseline ON THE TARGET MACHINE (needs GPU)

The four `test_task_digests.py` hashes were recorded on a GPU at `8554c1f`. Comparing them
across hardware is precisely the trap doc 20 F1 exists to prevent, so they cannot gate this
phase until they are re-established here. Capture, in one session on one GPU:

* the four task digests,
* the four bit-identity eval rows (`stage0_baseline/gate6.sh`).

**Nothing else in this phase may be called done before this exists.**

### P1 — Move core in (CPU-gatable)

`canvit_pytorch/*` → `canvit/core/*`; `canvit_train/*` → `canvit/*`; core's `tests/` and
`bench/` in too. Rewrite the 51 import sites (`canvit_pytorch.X` → `canvit.core.X`) and
`canvit_train.X` → `canvit.X`. Merge core's dependencies and extras (`demo` / `policy` /
`fovi`) into one `pyproject.toml` with `packages = ["canvit"]`; drop the
`canvit-pytorch = { path = "../CanViT-PyTorch", editable = true }` source; re-sync the venvs.

**Gate:** 361 CPU tests + core's 125, AND a new test asserting `canvit.core.__file__`
resolves inside this repo (Trap A). The import-provenance assertion is the load-bearing half:
both copies of core are functionally identical TODAY, so the tests pass either way and only
asking where the module came from can catch the shadow.

### P2 — Pinning (CPU)

New launchers stop setting `PYTORCH_COMMIT`; `harness_train.sbatch` gains the §4 guard.
Historical launchers are **not** touched — same rule as the `canvit_pretrain` pins.

**Gate:** `git archive` a post-merge commit into a temp dir and import `canvit_pytorch` and
`canvit_train` from it with `PYTHONSAFEPATH=1`; then archive an OLD `TRAIN_COMMIT` plus its
`PYTORCH_COMMIT` and confirm the old core still wins. Both are CPU-only.

### P3 — The numeric gate (needs GPU)

Four digests and four eval rows reproduce P0 **bit-identically**. This is a packaging change,
so anything that moves is a defect introduced here.

### P4 — Archive core, rename the repo

Mark `CanViT-PyTorch` read-only in `CLAUDE.md` with an `ARCHIVED.md` redirect (keeping it on
disk for the 116 pins), then rename the repo `CanViT-train` → `canvit`. The **package** names
stay as they are; see §3.

Note the rename invalidates `_REPO_BASE`-relative paths in the sbatch and the
`cd "$(dirname ...)/../../.."` idiom in every `slurm/runs/*.sh`. That is mechanical but it is
116 files' worth of blast radius, so it goes last and alone.

## 7. What could make this not worth doing

Recorded so the decision stays visible rather than assumed:

* Upstream merges become impractical — **accepted, owner 2026-09-03**, upstream will not be
  pulled again.
* Core stops being independently installable by anyone outside this project. Nobody is.
* `CanViT-PyTorch` must live on disk indefinitely for the pins. It already must, for the same
  reason three other repos do.
* The rename's blast radius (§P4) is the largest mechanical change in the phase and buys only
  a name. It is separable — P0–P3 deliver the merge; P4 can be deferred or skipped without
  leaving anything half-done.

Published HF checkpoints are unaffected: `config.json` records architecture only, no module
paths (doc 20 §8, verified during the eval merge).
