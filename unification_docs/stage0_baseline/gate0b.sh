set -uo pipefail
S=/mnt/vast-nhr/projects/nib00021/jonathan/tmp/_claude_tmp/stage0
TR=/user/henrich1/u25995/jonathan/repos/CanViT-train
EV=/user/henrich1/u25995/jonathan/repos/CanViT-eval
export ADE20K_ROOT=/mnt/vast-nhr/projects/nib00021/jonathan/datasets/zhoubolei--scene_parse_150/ADEChallengeData2016
export HF_HOME="$HOME/.cache/huggingface" TORCH_HOME="$HOME/.cache/torch"
BACKBONE=$TR/logs/jon_exp22_full_runs/exp22-fovi-teacherinit-lrdrop-1196k/checkpoints/step-155648-hf
VAL=/mnt/vast-nhr/projects/nib00021/jonathan/datasets/imagenet1k-val

echo "### A canvit_train test suite (core changed underneath it)"
cd $TR && .venv-cu126/bin/python -m pytest -q canvit_train 2>&1 | tail -4

echo "### B GATE ade20k fixation_grid (must equal Stage 0 bit-for-bit)"
cd $TR && .venv-cu126/bin/python scripts/eval_ade20k_checkpoint.py \
  --ckpt logs/jon_exp34_ade20k_probe/ade20k-fovi-ti-1196k/checkpoints/best.pt \
  --model-repo "$BACKBONE" --eval-policy fixation_grid \
  --n-timesteps 10 --fixed-scale 2.0 --resize-mode squish --canvas-grid 32 \
  --scene-size 512 --eval-batch-size 32 --num-workers 8 \
  --out "$S/gate0b_ade20k_fixgrid.json" >/dev/null

echo "### C GATE in1k fixation_grid (must equal Stage 0 bit-for-bit)"
cd $TR && .venv-cu126/bin/python "$S/eval_in1k.py" \
  --ckpt logs/jon_exp33_in1k_finetune/in1k-fovi-ti-1196k/checkpoints/best.pt \
  --model-repo "$BACKBONE" \
  --probe-repo canvit/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe \
  --mode finetune --eval-policy fixation_grid --n-timesteps 4 --fixed-scale 2.0 \
  --min-vp-scale 0.05 --eval-batch-size 64 --num-workers 8 \
  --out "$S/gate0b_in1k_fixgrid.json" >/dev/null

echo "### D re-export in1k to HF and load it back"
cd $TR && rm -rf "$S/exp33_in1k_400000_hf_fixed"
.venv-cu126/bin/python -m canvit_train.checkpoint.to_hf \
  --pt-path logs/jon_exp33_in1k_finetune/in1k-fovi-ti-1196k/checkpoints/best.pt \
  --out-dir "$S/exp33_in1k_400000_hf_fixed" 2>&1 | tail -3
python3 -c "
import json;d=json.load(open('$S/exp33_in1k_400000_hf_fixed/config.json'))
print('config.json keys:', sorted(d.keys()))
print('has model_config:', 'model_config' in d)"

echo "### E the check Stage 0 could NOT run: canvit_eval in1k, unpinned"
cd $EV && .venv/bin/python -m canvit_eval in1k-clf --mode finetuned \
  --episode.model-repo "$S/exp33_in1k_400000_hf_fixed" \
  --episode.policy repeated_full_scene --episode.n-timesteps 4 \
  --episode.canvas-grid 32 --scene-size 512 --batch-size 64 --num-workers 8 \
  --val-dir "$VAL" --output "$S/ce_in1k_full_noscale.pt" 2>&1 | tail -8

echo "### DONE"
