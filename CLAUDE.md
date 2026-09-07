# Coding guidelines

**Invoke the `andrej-karpathy-skills:karpathy-guidelines` skill at the start of every
session, before planning or writing code**, and follow it throughout. Short form: state
assumptions and ask when unsure; simplest code that works; surgical changes only; define a
verifiable success criterion and check it before claiming done. Never call work
"complete"/"a drop-in" without saying what the criterion was — and when porting or
replacing code, read the target end to end rather than planning from a `grep`.

# What this repo is

**CanViT** — the Canvas Vision Transformer, plus everything that trains and evaluates it.
One repo, one package.

Two entry points, and only two:

```bash
python -m canvit.harness.run      <distill|ade20k|in1k> --preset <default|probe|finetune|policy_only|joint>
python -m canvit.harness.evaluate <distill|ade20k|ade20k-dinov3|in1k> --cfg.eval-policy <policy>
```

| task | trains | data |
|---|---|---|
| `distill` | passive→active dense distillation from DINOv3 — the pretraining objective | IN21k WebDataset shards + a val image folder |
| `ade20k` | ADE20K segmentation probe / finetune | ADE20K (`$ADE20K_ROOT`) |
| `in1k` | ImageNet-1k linear probe / full finetune | IN1k WebDataset + val image folder |

On top of any of the three, the **viewpoint-selection policy** can be trained by RL, alone
against a frozen model (`policy_only`) or jointly with the task (`joint`).

`--cfg.eval-policy` is **required** for standalone eval — no `auto` guess. A foveated model
is only in-distribution at its training view scale, so pin it
(`--cfg.foveated-scale.fixed-scale`); off-scale glimpses make the metric *fall* as glimpses
accumulate.

## Package layout

One rule: **`harness/` holds the entry point and everything shared by more than one task;
each task folder holds only what is specific to that task.**

```
canvit/
├── core/        THE MODEL — canvas ViT, patchers, HF-hub classes, probes, teacher,
│                standardizers, viewpoint, rope. Imports nothing from the layers below.
├── harness/     entry point + shared primitives
│                run cli loop spec config flat; rollout/ policy/ optim/ infra/ viz/
├── distill/     DINOv3 latent distillation: loss, model, probe, data/, viz/
├── ade20k/      ADE20K segmentation: data, metrics, rollout, viz
├── in1k/        ImageNet-1k: data, metrics, model, eval, rollout
└── checkpoint/  checkpoint I/O and `to_hf` / `probe_to_hf` — publishing to the HF layout
```

`core/` must not import from `harness/`, `distill/`, `ade20k/` or `in1k/`. One grep checks it:

```bash
grep -rn "from canvit\.\(harness\|distill\|ade20k\|in1k\)" canvit/core/
```

Also at the repo root: `slurm/` (launchers), `unification_docs/` (design notes + history),
`readme_docs/` (campaign procedures), `bench/pt/` (inference benchmark), `demos/`,
`test_data/`.

# The other repos on disk

```
repos/
├── canvit/   this repo                       LIVE
└── fovi/     foveated-vision geometry        LIVE — the only sibling dependency
```

Four clones are kept **read-only, as fallback references. Do not edit them**, and prefer
this repo's equivalent:

| clone | superseded by | why it is kept |
|---|---|---|
| `CanViT-PyTorch/` | `canvit.core` | ~116 launchers `git archive` a core commit out of its `.git`. **Must stay on disk permanently.** |
| `CanViT-eval/` | `canvit.harness.evaluate` | its `results/` are the historical record — but see its `ARCHIVED.md`: the `reconstruction` task's `*_cos_raw` numbers are wrong |
| `CanViT-specialize/` | `canvit.{ade20k,in1k}` | pre-unification reference for the downstream recipes |
| `CanViT-PyTorch-RL/` | `canvit.core.policy` + `canvit.harness.policy` | pre-unification reference for the RL recipes; has its own `CLAUDE.md` governing work inside it |

# Gotchas

Each of these has cost someone real time.

- **There is no `canvit_pytorch` and no `canvit_train` package.** The model is
  `canvit.core`. Code importing the old names is pre-2026-09-03 and belongs to a pinned
  snapshot, not to this tree.
- **Historical launchers under `slurm/runs/` and `slurm/archive/` say `canvit_pretrain` or
  `canvit_train` ON PURPOSE** — they pin pre-rename commits whose snapshot carries that
  package directory. Do not "modernize" them; `harness_train.sbatch` detects which name a
  snapshot holds.
- **Those same launchers hardcode `repos/CanViT-train/...` checkpoint paths**, which stopped
  resolving when the repo was renamed on 2026-09-03 (no symlink was left behind): ~32 under
  `slurm/runs/` (exp23–exp36) and ~199 under `slurm/archive/`. **Decided (owner,
  2026-09-07): leave them.** They are the record of what those jobs ran, and new work gets
  new launchers; repoint the paths to `repos/canvit/logs/...` by hand only when you actually
  re-run one. Nothing fails silently — a guarded launcher refuses to submit (exp36's does),
  and the rest would die on a missing file.
- **`PYTORCH_COMMIT` is load-bearing for those old launchers but must NOT be set by new
  ones.** `TRAIN_COMMIT` now pins model and trainer together; setting `PYTORCH_COMMIT` on a
  post-merge pin has no effect. The launcher warns in both failure directions.
- **Use `.venv-cu126` for the test suite**, always. The four `test_task_digests.py` digests
  pin *CPU* numerics against hashes recorded under that torch build; `.venv` (cu130) fails
  exactly those four for that reason alone.
- **Eval gates are machine-local.** Bit-identity holds only within one GPU (≈1e-5 across
  GPU types, and MIG slice size counts). Old job logs cannot gate a refactor.
- **The wandb entity `cidas_goettingen` is shared** with other people's projects. Pin runs
  by id, assert on names, never iterate `api.projects(entity)` and act.
- **`git mv` on a directory carries untracked `__pycache__` with it**, and preserves mtimes,
  so Python keeps using stale bytecode whose paths no longer exist. Purge after any rename.

# Environments

Each repo has its **own** uv-managed venv; a venv is an editable install of its own repo, so
running it directly picks up your edits — no `PYTHONPATH` gymnastics.

```bash
.venv-cu126/bin/python -m pytest canvit          # 519 tests
```

torch uses the GPU only if its CUDA build is `<=` the node's driver. Many compute nodes here
cap at **CUDA 12.8** (A100/MIG, driver 570.x), where a `cu130` build is GPU-dead.

| venv | torch | use |
|---|---|---|
| `.venv` | cu130 | CUDA-13-capable nodes |
| `.venv-cu126` | cu126 | 12.8 nodes (A100/MIG) — **and all test runs** |

Rebuild the cu126 env with:
`UV_PROJECT_ENVIRONMENT=.venv-cu126 uv sync --no-group cuda --group cu126`

`[tool.uv.sources]` links `fovi` as a **relative-path editable install**
(`fovi = { path = "../fovi", editable = true }`), so the two repos must be cloned as
siblings. Swap to `{ git = ... }` to install without it.

# Datasets and env variables

Dataset roots, caches and the run-artifact directory come from the environment. For project
`nib00021` on Grete they are filled in already in **`.envrc.grete`**; `slurm/env.sh` sources
it directly, so submitting needs no `direnv`. For interactive work,
`source .envrc.grete` (or `cp .envrc.grete .envrc && direnv allow`).

The variables that matter: `LOGS_DIR`, `WEBDATASET_DIR`, `VAL_DIR`, `VAL_INDEX_DIR`,
`ADE20K_ROOT`, `IN1K_TRAIN_DIR`, `IN1K_VAL_DIR`, `HF_HOME`, `HF_TOKEN`, `WANDB_DIR`,
`WANDB_PROJECT`, `WANDB_ENTITY`. Full table and first-time setup: `README.md`.

# Commit pinning — why editing the repo is safe during live jobs

The venvs are editable installs, so in principle a running job would pick up a mid-run edit.
`slurm/harness_train.sbatch` neutralizes that: each run pins exact commits via
**`TRAIN_COMMIT`** and **`FOVI_COMMIT`**, and the sbatch `git archive`s them **offline** into
the per-job `$TMPDIR/canvit_src`, prepending that snapshot to `PYTHONPATH` with
`PYTHONSAFEPATH=1` so it wins over the cwd and the editable install.

- You can **edit the repo freely** while pinned jobs are in flight. Only newly submitted
  jobs that pin a new commit pick up changes.
- `git archive` reads the local `.git` only — no network, no SSH, works for private repos,
  and does not touch HEAD or the working tree.
- Launcher scripts and `.envrc.grete` are read at submit time, so editing them is always
  safe.
- Legacy spellings still accepted: `PRETRAIN_COMMIT` (≈48 launchers) and `PYTORCH_COMMIT`
  (≈116) — see Gotchas.
- **Workflow:** edit → commit/push → submit runs pinning the new hash.

# Guardrails

- **Claude does not run `git commit` / `push` / `pull`, and does not submit or cancel SLURM
  jobs, unless the user explicitly asks.** (In practice the user asks often; just don't do
  it unprompted.)
- **Never persist a `PYTHONPATH` override** in `~/.bashrc`, `~/.profile`, a SLURM script or
  any sourced file, and never `pip install -e` one repo into another's venv in a way that
  rewrites the editable `.pth`. Both leak in-progress edits into unrelated runs.
- Stage explicit paths. No `git add -A`, `git commit -a`, `git checkout`, `git stash`.
- HuggingFace: cached downloads fine, **no uploads**.

# History

This repo is the result of merging five repos into one. The narrative lives in
`unification_docs/`, not here:

| doc | what happened |
|---|---|
| `17-harness-consolidation.md` | specialize + RL folded in; one trainer; `CanViT-pretrain` → `CanViT-train` |
| `18-package-restructure.md` | `train/` + `tasks/` dissolved into `harness/` + per-task dirs |
| `20-eval-merge.md` | `CanViT-eval` folded in; one eval entry point |
| `21-core-merge.md` | `CanViT-PyTorch` folded in as `canvit/core`; repo renamed `canvit` |

`00-master-plan.md` §5 lists what was deliberately **not** ported.

# Notes

- Persistent memory for these sessions lives under
  `~/.claude/projects/<encoded-repo-path>/memory/` (`MEMORY.md` is the index).
