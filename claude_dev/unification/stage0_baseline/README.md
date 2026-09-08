# Stage-0 baseline — raw artifacts (2026-08-31)

The numbers behind `../20-eval-merge.md` §5 Stage 0. Every later stage of the CanViT-eval
merge is gated against these. Read the doc for what they mean and which rows are gates; this
directory is only the raw evidence plus enough to regenerate it.

* `ade20k_*.json`, `in1k_*.json`, `distill_*.json` — `canvit_train` at `716051a`/HEAD.
* `ce_ade20k_*.json` — `canvit_eval`, transcribed out of its `.pt` output.
* `ade20k.sh`, `batch2.sh`, `batch3.sh` — the exact invocations, in order.
* `eval_in1k.py`, `eval_distill.py` — throwaway drivers, deliberately not part of the
  package. Stage 3's `harness.evaluate` replaces them; they are here so the baseline is
  reproducible in the meantime. `scripts/eval_ade20k_checkpoint.py` served the ade20k rows.

Measured on a **MIG 3g.40gb slice** of an A100-80GB. Same-machine repeats of a deterministic
policy are bit-identical; the same quantity on a full A100 differs by ~1e-5 (§5 F1), so these
files are a reference for re-runs *on that same slice*, not an absolute truth.
