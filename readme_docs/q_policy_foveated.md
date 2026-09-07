# Training the Q viewpoint policy for a foveated model

Trains the ADE20K viewpoint policy (Q-regression) on our foveated CanViT backbone and the
ADE20K probe trained on it. The policy learns **where to look next**; the backbone and probe
stay frozen.

> **Run once, exploratory.** This recipe has been run end to end exactly one time — exp36,
> ten seeds, 2026-08-31 — and it works: the policy beats random viewpoints on every seed.
> That is a *first result*, not a reference number. It has no independent replication, no
> second backbone/probe pair, and nothing to be checked against, so treat it as evidence
> that the foveated path trains, and treat your own run as a smoke test all the same (see
> [Checking a run](#checking-a-run)). What exp36 measured is at the end:
> [What the first run showed](#what-the-first-run-showed-exp36).

## Run it

```bash
for s in 0 1 2 3 4 5 6 7 8 9; do
  SEED=$s bash slurm/runs/exp36_policy_qreg_fovi/policy-qreg-fovi-s0.sh
done
```

One A100 per seed, 9000 steps. The equivalent uniform runs took 75 min each at canvas 64;
this one runs at canvas 32, so expect that or less.

The launcher refuses to submit unless the probe run has finished (it checks for the probe's
final `step-40000.pt`, not `best.pt`, which appears at the first evaluation and would hand
the policy a probe a few hundred steps old). That probe run **is** finished, so the guard is
already satisfied.

## The two checkpoints it uses

Absolute paths, readable by anyone in the `HPC_nib00021` project — the files are mode 640
and every parent directory is group-readable, so a collaborator can point at them directly
without copying:

| flag | value |
|---|---|
| `--cfg.model-repo` | `/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit/logs/jon_exp22_full_runs/exp22-fovi-teacherinit-lrdrop-1196k/checkpoints/step-155648.pt` |
| `--cfg.probe-repo` | `/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit/logs/jon_exp34_ade20k_probe/ade20k-fovi-ti-1196k/checkpoints/best.pt` |

Both are training checkpoints passed straight to the flags — no HF export, no conversion
step. Both flags accept a training `.pt`, a local HF directory, or a Hub id.

Verified by building the model this recipe specifies: the backbone comes out foveated at
scale 2.0 on a canvas grid of 32, and all 9 probe-head tensors are bit-identical to the ones
in the probe checkpoint, i.e. the reward model really is the trained probe and not a fresh
random head.

The probe is **the reward model**, not a detail: the reward is the fraction of the probe's
cross-entropy that a glimpse removes, so a mismatched head makes the reward pure noise and
the policy learns nothing.

## What the policy chooses

Every glimpse is the **same foveation pattern**, and only the fixation centre changes — t0
included, which is simply the centred one. There is no full-image glimpse anywhere in the
rollout.

The foveation window is `fix_size = scale × H`, so at scale 2.0 it spans twice the image
side. The policy is choosing where to spend resolution, not what is visible. Its action
space is therefore a grid of **centres only, with no scale dimension**.

## Three settings that must match the checkpoints

Each fails **silently** — a worse number, not an error.

| setting | value here | why |
|---|---|---|
| `--cfg.canvas-grid` | `32` | must equal the grid the probe was trained at, or the reward model sees a canvas resolution it never saw. Easy to get wrong by copying: the uniform recipe uses 64 because *its* probe was trained at 64. |
| `--cfg.foveated-scale.fixed-scale` | `2.0` | must equal the backbone's pretraining scale, or every glimpse is out of distribution and mIoU decays as glimpses accumulate |
| `--cfg.resize-mode` | `squish` | the protocol every other CanViT number in this repo is measured under |

Change the probe, and `canvas-grid` has to change with it.

## Checking a run

**Step 1 — is the pairing right?** `eval/miou_t0` is measured *before* any policy action, so
it depends only on the frozen backbone, the probe, the resize and the eval path — never on
the policy. For this pair it must land at **0.377**. If it does not, backbone and probe are
mismatched and every later timestep is meaningless; fix that before reading anything else.

**Step 2 — does it beat random?** These two checkpoints under *random* viewpoints, which is
the bar a learned policy has to clear:

| | t0 | t1 | t2 | t3 | t4 |
|---|---|---|---|---|---|
| random viewpoints | 0.377 | 0.403 | 0.415 | 0.424 | **0.428** |

Both rows above were measured on 2026-08-02 with exactly this recipe's settings — full
ADE20K val, `n_timesteps 5`, `canvas_grid 32`, `squish`, `fixed_scale 2.0` — against
`best.pt`, so they are directly comparable to what the policy run reports:

```bash
python -m canvit.harness.evaluate ade20k \
  --opts.ckpt <probe .pt> --cfg.model-repo <backbone .pt> \
  --cfg.eval-policy random --cfg.n-timesteps 5 --cfg.canvas-grid 32 \
  --cfg.resize-mode squish --cfg.foveated-scale.fixed-scale 2.0
```

A trained policy should beat 0.428 at t4, and should beat random *earliest* — the claim is
that it reaches a given mIoU in fewer glimpses, so the gap at t1–t2 matters more than at t4.

Also check the **shape**: mIoU must rise monotonically across t1–t4. Falling mIoU as
glimpses accumulate is the signature of a scale mismatch between the rollout and the
backbone's pretraining scale.

**Do not borrow the uniform policy figures** (`ce_mean` ≈ 0.686, `miou_final` ≈ 0.448):
different backbone, different probe, canvas grid 64. They do not transfer, and a correct run
here would look broken against them.

## What the first run showed (exp36)

`jon_exp36_policy_qreg_fovi`, ten seeds, 2026-08-31 (jobs 15654175–15654184, ~65 min each on
one A100). The first Q-policy training ever completed on a foveated backbone. Numbers are the
last evaluation of each run (step 8000), mean ± sd over the ten seeds, next to the random
baseline from [Checking a run](#checking-a-run):

| | t0 | t1 | t2 | t3 | t4 |
|---|---|---|---|---|---|
| random viewpoints | 0.377 | 0.403 | 0.415 | 0.424 | 0.428 |
| exp36 policy | 0.3768 ± 0.0000 | 0.4191 ± 0.0013 | 0.4359 ± 0.0014 | 0.4447 ± 0.0023 | **0.4500 ± 0.0033** |

`ce_mean` 0.6952 ± 0.0016.

Both checks in [Checking a run](#checking-a-run) pass. `miou_t0` came out at 0.3768 on all
ten seeds — identical to the pairing figure, as it must be, since t0 precedes any policy
action — so backbone and probe were paired correctly. And the policy clears random at t4 on
every seed, worst 0.4455.

The **earliness** is the part worth noting, because reaching a given mIoU in fewer glimpses
is the whole claim: the policy passes random's five-glimpse score (0.428) by **t2**. mIoU also
rises monotonically t0→t4, which is what rules out a scale mismatch.

### How much this establishes

Little, deliberately. One campaign, one backbone/probe pair, one hyperparameter setting, and
the ± above is seed spread within that single setting — it is not a run-to-run band for the
recipe, and there is no earlier foveated-policy result to compare against. The comparison to
random is sound (t0 agrees to four decimals, and both rows were measured after the mIoU
re-base of 2026-07-29, `68b635f`), but it is a comparison against an open-loop baseline, not
a verification of anything.

Do not promote these into a gate or an expected result. If a later run misses them, that is a
difference to investigate, not a regression — and note that the launcher's checkpoint paths
predate the 2026-09-03 repo rename, so a re-run needs them repointed first.
