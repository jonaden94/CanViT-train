set -uo pipefail
S=/mnt/vast-nhr/projects/nib00021/jonathan/tmp/_claude_tmp/stage0
TR=/user/henrich1/u25995/jonathan/repos/CanViT-train
EV=/user/henrich1/u25995/jonathan/repos/CanViT-eval
export ADE20K_ROOT=/mnt/vast-nhr/projects/nib00021/jonathan/datasets/zhoubolei--scene_parse_150/ADEChallengeData2016
export HF_HOME="$HOME/.cache/huggingface" TORCH_HOME="$HOME/.cache/torch"
BACKBONE=$TR/logs/jon_exp22_full_runs/exp22-fovi-teacherinit-lrdrop-1196k/checkpoints/step-155648-hf
IN1KCKPT=$TR/logs/jon_exp33_in1k_finetune/in1k-fovi-ti-1196k/checkpoints/best.pt
VAL=/mnt/vast-nhr/projects/nib00021/jonathan/datasets/imagenet1k-val

in1k() { echo "### in1k $1"; cd $TR && .venv-cu126/bin/python "$S/eval_in1k.py" \
  --ckpt "$IN1KCKPT" --model-repo "$BACKBONE" \
  --probe-repo canvit/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe \
  --mode finetune --n-timesteps 4 --fixed-scale 2.0 --min-vp-scale 0.05 \
  --eval-batch-size 64 --num-workers 8 --eval-policy "$2" --out "$S/in1k_$1.json"; }

in1k fixgrid_run1 fixation_grid
in1k fixgrid_run2 fixation_grid
in1k full_noscale  full

echo "### canvit_eval in1k repeated_full_scene (unpinned; in1k HF metadata is empty so no auto-pin)"
cd $EV && .venv/bin/python -m canvit_eval in1k-clf --mode finetuned \
  --episode.model-repo "$S/exp33_in1k_400000_hf" \
  --episode.policy repeated_full_scene --episode.n-timesteps 4 \
  --episode.canvas-grid 32 --scene-size 512 --batch-size 64 --num-workers 8 \
  --val-dir "$VAL" --output "$S/ce_in1k_full_noscale.pt"

echo "### canvit_train ade20k full WITHOUT pin (demonstrates the 'or full' doc defect)"
cd $TR && .venv-cu126/bin/python scripts/eval_ade20k_checkpoint.py \
  --ckpt logs/jon_exp34_ade20k_probe/ade20k-fovi-ti-1196k/checkpoints/best.pt \
  --model-repo "$BACKBONE" --eval-policy full \
  --n-timesteps 10 --fixed-scale 2.0 --resize-mode squish --canvas-grid 32 \
  --scene-size 512 --eval-batch-size 32 --num-workers 8 \
  --out "$S/ade20k_full_noscale.json"

echo "### DONE"
