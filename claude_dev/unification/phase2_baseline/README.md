# Phase-2 P0 baseline — raw artifacts (2026-09-03)

The pre-merge reference that P3 must reproduce after `canvit_pytorch` moves into
`canvit/core/`. Read `../21-core-merge.md` §9 for what it means; this directory is the raw
evidence plus the exact invocation.

* `p0_*.json` — the four `canvit_train.harness.evaluate` rows, `main` @ `d5d78eb`.
* `run_p0.sh` — the exact invocation, re-runnable as `bash run_p0.sh <out-dir>`. Same four
  configs as `../stage0_baseline/gate3b.sh`; it also samples `nvidia-smi` so peak VRAM is a
  number rather than an absence of OOM.

**Machine: MIG 1g.20gb slice of an A100-80GB, node `ggpu137`, driver 570.211.01,
`.venv-cu126` (torch 2.11.0+cu126).** Doc 20's Stage-0 tables were recorded on a **3g.40gb**
slice, so this is a different slice of the same GPU model — one third the SMs and half the
memory. Peak GPU memory across all four rows: **12 513 MiB of 19 968**, reached by the
ade20k rows; in1k and distill sit near 5 GiB.

Wall clock: ade20k rows 2m53s / 2m51s, in1k 16m00s (50 000 images), distill 2m09s (256
samples). 24 minutes total.

## P3 (2026-09-03) — the gate, on the same slice

`p3_*.json` are the same four rows re-run on the merged tree (`ba6ed9c`), and
`run_p3.sh` is `run_p0.sh` with one substitution — the module path — so the two differ in
exactly the thing under test.

`compare.py` checks them. **82 scalars, 0 violations, worst |diff| = 0.000e+00**, and the
resolved `protocol` block matches on all four rows. Peak GPU memory was 12 513 MiB in both
runs. Wall clock 22m41s vs 23m53s.

Distill's 56 scalars were allowed 1e-5 (§9.2) and came out exact anyway, so the tolerance
was never used.
