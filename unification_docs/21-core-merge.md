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

## 8. P1 survey — measured 2026-09-03, before any edit was made

Everything below was established by reading the two trees, **not** from the plan above. It is
recorded because it cost a session's worth of surveying and because three items change what
P1 should do.

### 8.1 The CPU gate is 365/365 — and §2 was wrong about why four tests fail

§2 of this plan claimed `test_task_digests.py` "assert[s] GPU-recorded hashes" and so cannot
run without a GPU. **That is false.** The file sets `_DEVICE = torch.device("cpu")`; the
digests are CPU tests, and its own docstring names the venv to run them in
(`.venv-cu126/bin/python -m pytest …`).

What actually happens:

| venv | torch | `pytest canvit_train` |
|---|---|---|
| `.venv` | 2.11.0+**cu130** | 361 passed, **4 failed**, 744s |
| `.venv-cu126` | 2.11.0+**cu126** | **365 passed**, 257s |

The four digests were recorded under cu126 and the two builds' **CPU** kernels do not agree
bit-for-bit, so running them under cu130 fails them. Nothing about a GPU is involved. (An
earlier revision of this section repeated §2's error and attributed the four failures to
GPU-recorded hashes; it was wrong for the same reason.)

**So P1's gate is `.venv-cu126` → 365 passed, 0 failed.** A gate with no expected failures is
much harder to misread than "361 plus four known exceptions", and it needs no GPU. Use the
same venv for the numeric rows, which is what doc 20 did.

### 8.2 Trap A is defused BY the rename — and this is why `uv sync` is not on P1's critical path

§5 assumed the editable install could silently keep serving the old core. Checked: the three
`.pth` files in each venv are **plain path lines adding repo ROOTS** to `sys.path` —
`/…/repos/CanViT-PyTorch`, `/…/repos/CanViT-train`, `/…/repos/fovi`. So after the move, with
no re-sync at all:

* `canvit` resolves from the CanViT-train root, which is already on `sys.path`, and
  `canvit.core` comes with it;
* the stale CanViT-PyTorch root still offers the top-level name `canvit_pytorch`, but **no
  file imports that name any more**, so it is inert, not shadowing.

The trap was real only for the design that kept the name `canvit_pytorch` — there,
alphabetical `.pth` order (`…_canvit_pytorch` before `…_canvit_train`) would have handed every
import to the old clone with no error. **The rename the owner asked for removes the failure
mode.** That matters practically: `uv sync` needs the network, which is not guaranteed here,
and P1's gate is meaningful without it.

Still add both tests — they cost nothing and they are the only detection if the layout changes
again: (a) `canvit.core.__file__` resolves under this repo, (b) no file under `canvit/` imports
`canvit_pytorch`. Leave the stale `.pth` and `canvit_pytorch-0.1.9.dist-info` alone; the next
`uv sync` clears them.

### 8.3 `_PKG` needs a third branch

`slurm/harness_train.sbatch:121-124` is a two-way detect (`canvit_train`, falling back to
`canvit_pretrain` when the pinned snapshot contains that directory). After P1 it is a
three-way: default `canvit`, fall back to `canvit_train`, then `canvit_pretrain`. Same
principle as before — detect from the snapshot, never hardcode.

### 8.4 Rewrite scope for `canvit_train` → `canvit`

**Rewrite:** the package tree (66 files), `pyproject.toml` (name, `packages`, `testpaths`,
`per-file-ignores`), `README.md`, `slurm/README.md`, `readme_docs/q_policy_foveated.md`,
`scripts/*.py` and `scripts/*.sh`, `slurm/harness_train.sbatch`.

**Leave:** `slurm/runs/**` and `slurm/archive/**` — all 29 hits inspected, every one is a
comment or pinned history; `unification_docs/**`; `readme_docs/verification_runs.md` (a
record); `.gitignore:39`, where `slurm/canvit_train_state.venv-cu126` is an **archived scrap
script's state file**, not the package.

A blanket sed over the package is correct, not sloppy: the 46 non-import occurrences are
docstrings, `monkeypatch.setattr` dotted targets, and `logging.getLogger` names — all dotted
module paths, all of which must move.

One piece of **pre-existing** rot the sed will carry over rather than fix:
`harness/config.py:95` cites `canvit_train.train.rl.VPG`, a path deleted by the `fe35b62`
restructure. Flagged, not touched.

### 8.5 Disposition of core's non-package contents — the part §6 did not plan

| core path | destination | why |
|---|---|---|
| `canvit_pytorch/` | `canvit/core/` | 25 top-level entries, disjoint from canvit_train's 5 |
| `tests/` (3 files) | `canvit/core/tests/` | matches this repo's convention (`canvit/harness/tests/`, which has an `__init__.py`); keeps `testpaths` single-valued and avoids rootdir module-name collisions |
| `test_data/` (2 images, 405K) | repo root `test_data/` | referenced by **relative** path — `tests/test_classification.py:20`, `demos/basic.py:32`, `demos/classify.py:80` — so it must sit at the rootdir and those tests only pass when pytest runs from the repo root. Pre-existing fragility, preserved deliberately. |
| `bench/` | repo root `bench/` | the CanViT-eval benchmark adopted in `3a0dcc2`, plus its 2 baseline jsonl |
| `demos/` (2 files) | repo root `demos/` | live examples; depend on the relative `test_data/` |
| `assets/` (928K) | repo root `assets/` | README images |
| `other_papers/` (1 PDF, 6.2M) | repo root `papers/` | this repo already has `papers/fourier.pdf`; one folder, not two |
| `README.md` (260 lines) | merged into the root `README.md` | **corrected — see §9.3.** `readme_docs/_README.md` states its own charter: "The README describes the repository; these documents describe *procedures*." A model/API reference is not a procedure, and after the merge the model IS part of the repository, so the front page is its home. Keeping it at the root also leaves its four relative links (`assets/`, `test_data/`, `demos/`) valid with no rewrites. |
| `LICENSE.md` | drop | same MIT text and copyright line as this repo's `LICENSE` |
| `.github/workflows/release.yml` | drop | it publishes the `canvit-pytorch` distribution; the merged package is not published |
| `.python-version`, `.gitignore`, `uv.lock` | merge, never copy | the lock is regenerated by the sync |
| `canvit_paper/`, `.claude/` | leave in place | untracked (14M), not git content |

### 8.6 pyproject merge, concretely

Core's base deps (`huggingface-hub>=1.3.2`, `numpy>=2.2.0,<2.4.0`, `torch>=2.9.0`,
`torchvision>=0.22.0`, `safetensors>=0.7.0`) fold into `canvit`'s. Keeping the torch bounds in
base deps **preserves the current resolution** — they were already there transitively via
`canvit-pytorch` — while the conflicting `cuda` / `cu126` groups go on pinning the build.

Core's three extras:

* `fovi` → plain dep. Train always asked for `canvit-pytorch[fovi]`, so it was never optional.
* `policy` (timm) → plain dep. **Checked: this is a no-op for the resolution** — `timm` is an
  unconditional requirement of `fovi` (`uv.lock:1071`, `:1102`), so it is installed either
  way. Fold it in regardless, so the dependency is declared where it is used instead of
  inherited by accident.
* `demo` → stays an extra, minus whatever base already covers.

Core's pytest config must come along or its tests change selection: `markers = [slow,
network]` and `addopts = "-m 'not slow'"`.

### 8.7 Ordering: a GPU appeared, so P0 goes first after all

The "record P0 later" arrangement in §6 existed only to work around having no GPU. One is now
available, so the original ordering is restored: **record P0 on `main` first**, on the GPU
that will also run P3, then branch. That removes the deferral cost §6 had to manage, and it
respects F1 — the digests and the four eval rows are only comparable within one machine.

## 9. P0 — EXECUTED 2026-09-03. The pre-merge baseline exists.

Machine: **MIG 1g.20gb slice of an A100-80GB**, node `ggpu137`, driver 570.211.01,
`.venv-cu126`. Raw artifacts + the re-runnable invocation:
`unification_docs/phase2_baseline/`.

Two components, both from `main` @ `d5d78eb`:

1. **Test suite** — `.venv-cu126/bin/python -m pytest canvit_train -q` → **365 passed, 0
   failed, 257s** (§8.1).
2. **Four `harness.evaluate` rows** — the same four configs as
   `stage0_baseline/gate3b.sh`, since those are the ones that go through the merged entry
   point. All four exited 0; 24 min wall clock.

### 9.1 The hardware answer: 20 GB is ample, and three of four rows did not move at all

Peak GPU memory over the whole run: **12 513 MiB of 19 968 (63%)**, reached by the ade20k
rows (batch 32, 512 px scene, 32×32 canvas, 10 timesteps — the heaviest config in the set).
in1k and distill sit near 5 GiB. A 20 GB slice is therefore not a constraint on any gate in
this phase; only wall clock is, and only because 1g is one third of the SMs.

Comparing to doc 20's **3g.40gb** figures — a different slice of the same GPU model:

| row | 1g.20gb (P0) | 3g.40gb (doc 20) | diff |
|---|---|---|---|
| ade20k `fixation_grid`, all 12 mIoU scalars | — | — | **0, exactly** |
| ade20k `full`+pin 2.0, all 12 mIoU scalars | — | — | **0, exactly** |
| in1k `fixation_grid` top1 / top5 | 0.83632 / 0.97006 | 0.83632 / 0.97006 | **0, exactly** |
| distill `val_metric` | 0.9257651567459106 | 0.925767719745636 | 2.56e-06 |

### 9.2 This SHARPENS F1 rather than contradicting it — and it changes P3's gate

F1 (doc 20 §5) says bit-identity holds only within one GPU, with ~1e-5 across GPU types. The
table above looks like a counterexample. It is not. **The three exact rows are exact because
their metrics are integer-derived, not because the arithmetic agreed:**

* in1k top1/top5 are `correct / N` — a count over a fixed denominator. It is exact unless a
  *prediction flips*, which takes far more than 1e-6 of drift.
* ade20k mIoU comes from `mIoUAccumulator`'s integer confusion counts, summed in float64. Same
  story: the float drift has to reach the argmax before the metric notices.
* distill's `val_metric` is a **mean of cosine similarities** — a float reduction with nothing
  quantizing it. It is the only one of the four that exposes raw drift, and it moved 2.56e-06,
  right in F1's band.

So the four rows are not four samples of the same quantity. Three are *robust* statistics that
tolerate the hardware offset; one is a *sensitive* one that reports it. **P3 must gate them
differently:**

* **ade20k (24 scalars) and in1k (2 scalars): require exact equality.** They are
  integer-derived, so on one machine any movement at all means a real change in predictions —
  which is precisely the defect class a packaging change could introduce. No tolerance.
* **distill (56 scalars): require agreement to ~1e-5**, not bit-identity. Demanding exactness
  there would make the gate fail for reasons that have nothing to do with the merge. If it
  moves by more than that on the *same* slice, that is a real finding.

A packaging change should of course move nothing at all, and P3 is run on this same slice, so
in practice all 82 scalars should be identical. The point of splitting the rule is that when
something does move, the tolerance says immediately whether it is the hardware or the merge.

### 9.3 Two errors in §8 found while executing P0

* **§8.1 / §2 on the four digest tests** — corrected in place. They are CPU tests; the
  failures were a cu130-vs-cu126 venv mismatch. The gate is 365/365, no GPU needed.
* **§8.5 on core's README** — corrected in place. It goes into the root `README.md`, not
  `readme_docs/`, because `readme_docs/_README.md` defines that directory as *procedures* and a
  model/API reference is not one. Keeping it at the root also preserves its four relative
  links. If the merged front page then reads as too long, splitting it is a separate and easy
  change; it should not be pre-empted here.

One expected, non-defect difference: `gate3b_distill.json` holds **1** metric, `p0_distill.json`
holds **56**. Stage 4 of the eval merge widened `distill/validate.py` from returning
`scene_cos_raw[-1]` to returning the full scalar dict, after gate3b was recorded. The new set
is a strict superset — `val_metric` is still there and is the row compared above.

**P0 is complete, so the §6 precondition is satisfied and P1 may proceed.**
