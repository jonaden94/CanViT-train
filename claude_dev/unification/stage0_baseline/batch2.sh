set -uo pipefail
S=/mnt/vast-nhr/projects/nib00021/jonathan/tmp/_claude_tmp/stage0
TR=/user/henrich1/u25995/jonathan/repos/CanViT-train
EV=/user/henrich1/u25995/jonathan/repos/CanViT-eval
export ADE20K_ROOT=/mnt/vast-nhr/projects/nib00021/jonathan/datasets/zhoubolei--scene_parse_150/ADEChallengeData2016
export HF_HOME="$HOME/.cache/huggingface" TORCH_HOME="$HOME/.cache/torch"
BACKBONE=$TR/logs/jon_exp22_full_runs/exp22-fovi-teacherinit-lrdrop-1196k/checkpoints/step-155648-hf
PROBEHF=$S/exp34_fovi_ti_probe_hf

echo "### 1 canvit_train ade20k full+pin2.0"
cd $TR && .venv-cu126/bin/python scripts/eval_ade20k_checkpoint.py \
  --ckpt logs/jon_exp34_ade20k_probe/ade20k-fovi-ti-1196k/checkpoints/best.pt \
  --model-repo "$BACKBONE" --eval-policy full --override-scale 2.0 \
  --n-timesteps 10 --fixed-scale 2.0 --resize-mode squish --canvas-grid 32 \
  --scene-size 512 --eval-batch-size 32 --num-workers 8 \
  --out "$S/ade20k_full_pin2_run1.json"

echo "### 2 canvit_eval ade20k repeated_full_scene+pin2.0"
cd $EV && .venv/bin/python -m canvit_eval ade20k-seg-canvit \
  --probe-repo "$PROBEHF" --episode.model-repo "$BACKBONE" \
  --episode.policy repeated_full_scene --episode.n-timesteps 10 \
  --episode.canvas-grid 32 --episode.override-scale 2.0 \
  --scene-size 512 --resize-mode squish --batch-size 32 --num-workers 8 \
  --output "$S/ce_ade20k_full_pin2.pt"

echo "### 3 canvit_eval ade20k random+pin2.0"
cd $EV && .venv/bin/python -m canvit_eval ade20k-seg-canvit \
  --probe-repo "$PROBEHF" --episode.model-repo "$BACKBONE" \
  --episode.policy random --episode.n-timesteps 10 \
  --episode.canvas-grid 32 --episode.override-scale 2.0 \
  --scene-size 512 --resize-mode squish --batch-size 32 --num-workers 8 \
  --output "$S/ce_ade20k_random_pin2.pt"

echo "### 4 canvit_train distill step-1916928 (has a logged eval)"
cd $TR && .venv-cu126/bin/python "$S/eval_distill.py" \
  --ckpt logs/jon_exp32_pretrain_lrdrop/exp32-fovi/checkpoints/step-1916928.pt \
  --val-dir /mnt/vast-nhr/projects/nib00021/jonathan/datasets/imagenet1k-val \
  --val-index-dir /mnt/vast-nhr/projects/nib00021/jonathan/repos/_data_cache \
  --out "$S/distill_1916928_run1.json"

echo "### 5 canvit_train distill step-1916928 repeat"
cd $TR && .venv-cu126/bin/python "$S/eval_distill.py" \
  --ckpt logs/jon_exp32_pretrain_lrdrop/exp32-fovi/checkpoints/step-1916928.pt \
  --val-dir /mnt/vast-nhr/projects/nib00021/jonathan/datasets/imagenet1k-val \
  --val-index-dir /mnt/vast-nhr/projects/nib00021/jonathan/repos/_data_cache \
  --out "$S/distill_1916928_run2.json"

echo "### 6 canvit_train distill latest (step-2007040)"
cd $TR && .venv-cu126/bin/python "$S/eval_distill.py" \
  --ckpt logs/jon_exp32_pretrain_lrdrop/exp32-fovi/checkpoints/latest.pt \
  --val-dir /mnt/vast-nhr/projects/nib00021/jonathan/datasets/imagenet1k-val \
  --val-index-dir /mnt/vast-nhr/projects/nib00021/jonathan/repos/_data_cache \
  --out "$S/distill_2007040_run1.json"

echo "### 7 canvit_train in1k best.pt (step 400000, has a logged eval)"
cd $TR && .venv-cu126/bin/python "$S/eval_in1k.py" \
  --ckpt logs/jon_exp33_in1k_finetune/in1k-fovi-ti-1196k/checkpoints/best.pt \
  --model-repo "$BACKBONE" \
  --probe-repo canvit/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe \
  --mode finetune --eval-policy random --n-timesteps 4 --fixed-scale 2.0 \
  --min-vp-scale 0.05 --eval-batch-size 64 --num-workers 8 \
  --out "$S/in1k_400000_run1.json"

echo "### DONE"
