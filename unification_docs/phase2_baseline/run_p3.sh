#!/bin/bash
# P3 numeric gate (doc 21 P3). Byte-identical to run_p0.sh except for the module path
# (canvit_train.harness.evaluate -> canvit.harness.evaluate), which is the whole point:
# the same four rows, same machine, same venv, through the renamed package.
# Regenerate with:  sed 's/canvit_train\.harness\.evaluate/canvit.harness.evaluate/' run_p0.sh
# bit-identically after the core merge. Same four rows as
# unification_docs/stage0_baseline/gate3b.sh, re-recorded because F1 makes them
# machine-local and this is a MIG 1g.20gb slice, not the 3g.40gb one doc 20 used.
set -uo pipefail
OUT="$1"                       # where the JSONs go
cd /user/henrich1/u25995/jonathan/repos/CanViT-train
export ADE20K_ROOT=/mnt/vast-nhr/projects/nib00021/jonathan/datasets/zhoubolei--scene_parse_150/ADEChallengeData2016
export HF_HOME="$HOME/.cache/huggingface" TORCH_HOME="$HOME/.cache/torch"
B=logs/jon_exp22_full_runs/exp22-fovi-teacherinit-lrdrop-1196k/checkpoints/step-155648-hf
E=".venv-cu126/bin/python -m canvit.harness.evaluate"

# Sample GPU memory so "is 20GB enough" gets a margin, not just an absence of OOM.
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 2; done ) > "$OUT/vram_samples.txt" 2>/dev/null &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

row() { echo "### $* @ $(date +%T)"; }

row "1 ade20k fixation_grid"
$E ade20k --opts.ckpt logs/jon_exp34_ade20k_probe/ade20k-fovi-ti-1196k/checkpoints/best.pt \
  --opts.out "$OUT/p0_ade20k_fixgrid.json" --cfg.model-repo "$B" \
  --cfg.eval-policy fixation_grid --cfg.foveated-scale.fixed-scale 2.0 \
  --cfg.resize-mode squish --cfg.n-timesteps 10 --cfg.canvas-grid 32 \
  --cfg.scene-size 512 --cfg.eval-batch-size 32 --cfg.num-workers 8 >"$OUT/log_1.txt" 2>&1
echo "  exit=$?"

row "2 ade20k full+pin2.0"
$E ade20k --opts.ckpt logs/jon_exp34_ade20k_probe/ade20k-fovi-ti-1196k/checkpoints/best.pt \
  --opts.out "$OUT/p0_ade20k_full_pin2.json" --cfg.model-repo "$B" \
  --cfg.eval-policy full --cfg.eval-override-scale 2.0 --cfg.foveated-scale.fixed-scale 2.0 \
  --cfg.resize-mode squish --cfg.n-timesteps 10 --cfg.canvas-grid 32 \
  --cfg.scene-size 512 --cfg.eval-batch-size 32 --cfg.num-workers 8 >"$OUT/log_2.txt" 2>&1
echo "  exit=$?"

row "3 in1k fixation_grid"
$E in1k --opts.ckpt logs/jon_exp33_in1k_finetune/in1k-fovi-ti-1196k/checkpoints/best.pt \
  --opts.out "$OUT/p0_in1k_fixgrid.json" --cfg.model-repo "$B" \
  --cfg.probe-repo canvit/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe \
  --cfg.mode finetune --cfg.eval-policy fixation_grid --cfg.n-timesteps 4 \
  --cfg.foveated-scale.fixed-scale 2.0 --cfg.min-vp-scale 0.05 \
  --cfg.eval-batch-size 64 --cfg.num-workers 8 >"$OUT/log_3.txt" 2>&1
echo "  exit=$?"

row "4 distill fixation_grid"
$E distill --opts.ckpt logs/jon_exp32_pretrain_lrdrop/exp32-fovi/checkpoints/step-1916928.pt \
  --opts.out "$OUT/p0_distill.json" \
  --cfg.eval-policy fixation_grid --cfg.foveated-scale.fixed-scale 2.0 \
  --cfg.val-dir /mnt/vast-nhr/projects/nib00021/jonathan/datasets/imagenet1k-val \
  --cfg.val-index-dir /mnt/vast-nhr/projects/nib00021/jonathan/repos/_data_cache \
  --cfg.webdataset-dir /nonexistent >"$OUT/log_4.txt" 2>&1
echo "  exit=$?"

kill $SAMPLER 2>/dev/null
echo "### peak GPU memory used (MiB): $(sort -n "$OUT/vram_samples.txt" | tail -1)"
echo "### ALLDONE $(date +%T)"
