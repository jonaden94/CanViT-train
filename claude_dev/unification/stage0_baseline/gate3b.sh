set -uo pipefail
S=/mnt/vast-nhr/projects/nib00021/jonathan/tmp/_claude_tmp/stage0
cd /user/henrich1/u25995/jonathan/repos/CanViT-train
export ADE20K_ROOT=/mnt/vast-nhr/projects/nib00021/jonathan/datasets/zhoubolei--scene_parse_150/ADEChallengeData2016
export HF_HOME="$HOME/.cache/huggingface" TORCH_HOME="$HOME/.cache/torch"
B=logs/jon_exp22_full_runs/exp22-fovi-teacherinit-lrdrop-1196k/checkpoints/step-155648-hf
E=".venv-cu126/bin/python -m canvit_train.harness.evaluate"

echo "### suite"; .venv-cu126/bin/python -m pytest -q canvit_train 2>&1 | tail -3

echo "### 1 ade20k fixation_grid via harness.evaluate"
$E ade20k --opts.ckpt logs/jon_exp34_ade20k_probe/ade20k-fovi-ti-1196k/checkpoints/best.pt \
  --opts.out "$S/gate3b_ade20k_fixgrid.json" --cfg.model-repo "$B" \
  --cfg.eval-policy fixation_grid --cfg.foveated-scale.fixed-scale 2.0 \
  --cfg.resize-mode squish --cfg.n-timesteps 10 --cfg.canvas-grid 32 \
  --cfg.scene-size 512 --cfg.eval-batch-size 32 --cfg.num-workers 8 >/dev/null

echo "### 2 ade20k full+pin2.0 via harness.evaluate"
$E ade20k --opts.ckpt logs/jon_exp34_ade20k_probe/ade20k-fovi-ti-1196k/checkpoints/best.pt \
  --opts.out "$S/gate3b_ade20k_full_pin2.json" --cfg.model-repo "$B" \
  --cfg.eval-policy full --cfg.eval-override-scale 2.0 --cfg.foveated-scale.fixed-scale 2.0 \
  --cfg.resize-mode squish --cfg.n-timesteps 10 --cfg.canvas-grid 32 \
  --cfg.scene-size 512 --cfg.eval-batch-size 32 --cfg.num-workers 8 >/dev/null

echo "### 3 in1k fixation_grid via harness.evaluate"
$E in1k --opts.ckpt logs/jon_exp33_in1k_finetune/in1k-fovi-ti-1196k/checkpoints/best.pt \
  --opts.out "$S/gate3b_in1k_fixgrid.json" --cfg.model-repo "$B" \
  --cfg.probe-repo canvit/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe \
  --cfg.mode finetune --cfg.eval-policy fixation_grid --cfg.n-timesteps 4 \
  --cfg.foveated-scale.fixed-scale 2.0 --cfg.min-vp-scale 0.05 \
  --cfg.eval-batch-size 64 --cfg.num-workers 8 >/dev/null

echo "### 4 distill fixation_grid via harness.evaluate"
$E distill --opts.ckpt logs/jon_exp32_pretrain_lrdrop/exp32-fovi/checkpoints/step-1916928.pt \
  --opts.out "$S/gate3b_distill.json" \
  --cfg.eval-policy fixation_grid --cfg.foveated-scale.fixed-scale 2.0 \
  --cfg.val-dir /mnt/vast-nhr/projects/nib00021/jonathan/datasets/imagenet1k-val \
  --cfg.val-index-dir /mnt/vast-nhr/projects/nib00021/jonathan/repos/_data_cache \
  --cfg.webdataset-dir /nonexistent >/dev/null
echo "### ALLDONE"
