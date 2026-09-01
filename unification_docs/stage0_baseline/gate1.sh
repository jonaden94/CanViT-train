set -uo pipefail
S=/mnt/vast-nhr/projects/nib00021/jonathan/tmp/_claude_tmp/stage0
TR=/user/henrich1/u25995/jonathan/repos/CanViT-train
export ADE20K_ROOT=/mnt/vast-nhr/projects/nib00021/jonathan/datasets/zhoubolei--scene_parse_150/ADEChallengeData2016
export HF_HOME="$HOME/.cache/huggingface" TORCH_HOME="$HOME/.cache/torch"
B=$TR/logs/jon_exp22_full_runs/exp22-fovi-teacherinit-lrdrop-1196k/checkpoints/step-155648-hf
cd $TR
ade() { .venv-cu126/bin/python scripts/eval_ade20k_checkpoint.py \
  --ckpt logs/jon_exp34_ade20k_probe/ade20k-fovi-ti-1196k/checkpoints/best.pt \
  --model-repo "$B" --n-timesteps 10 --fixed-scale 2.0 --resize-mode squish \
  --canvas-grid 32 --scene-size 512 --eval-batch-size 32 --num-workers 8 \
  --out "$S/gate1_$1.json" "${@:2}" >/dev/null; }
echo "### 1 ade20k fixation_grid"; ade ade20k_fixgrid --eval-policy fixation_grid
echo "### 2 ade20k full+pin2.0";   ade ade20k_full_pin2 --eval-policy full --override-scale 2.0
echo "### 3 in1k fixation_grid"
.venv-cu126/bin/python "$S/eval_in1k.py" \
  --ckpt logs/jon_exp33_in1k_finetune/in1k-fovi-ti-1196k/checkpoints/best.pt \
  --model-repo "$B" --probe-repo canvit/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe \
  --mode finetune --eval-policy fixation_grid --n-timesteps 4 --fixed-scale 2.0 \
  --min-vp-scale 0.05 --eval-batch-size 64 --num-workers 8 \
  --out "$S/gate1_in1k_fixgrid.json" >/dev/null
echo "### 4 in1k full, unpinned (the cross-repo row)"
.venv-cu126/bin/python "$S/eval_in1k.py" \
  --ckpt logs/jon_exp33_in1k_finetune/in1k-fovi-ti-1196k/checkpoints/best.pt \
  --model-repo "$B" --probe-repo canvit/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe \
  --mode finetune --eval-policy full --n-timesteps 4 --fixed-scale 2.0 \
  --min-vp-scale 0.05 --eval-batch-size 64 --num-workers 8 \
  --out "$S/gate1_in1k_full_noscale.json" >/dev/null
echo "### DONE"
