set -euo pipefail
cd /user/henrich1/u25995/jonathan/repos/CanViT-train
S=/mnt/vast-nhr/projects/nib00021/jonathan/tmp/_claude_tmp/stage0
export ADE20K_ROOT=/mnt/vast-nhr/projects/nib00021/jonathan/datasets/zhoubolei--scene_parse_150/ADEChallengeData2016
export HF_HOME="$HOME/.cache/huggingface" TORCH_HOME="$HOME/.cache/torch"
REPO=logs/jon_exp22_full_runs/exp22-fovi-teacherinit-lrdrop-1196k/checkpoints/step-155648-hf
CKPT=logs/jon_exp34_ade20k_probe/ade20k-fovi-ti-1196k/checkpoints/best.pt
run() {  # $1=tag  $2..=extra flags
  local tag="$1"; shift
  echo "### $tag"
  .venv-cu126/bin/python scripts/eval_ade20k_checkpoint.py \
    --ckpt "$CKPT" --model-repo "$REPO" \
    --n-timesteps 10 --fixed-scale 2.0 --resize-mode squish \
    --canvas-grid 32 --scene-size 512 --eval-batch-size 32 --num-workers 8 \
    --out "$S/ade20k_$tag.json" "$@"
}
run random_run1     --eval-policy random
run random_run2     --eval-policy random
run fixation_run1   --eval-policy fixation_grid
run fixation_run2   --eval-policy fixation_grid
echo "### DONE"
