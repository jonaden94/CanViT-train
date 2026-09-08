set -uo pipefail
S=/mnt/vast-nhr/projects/nib00021/jonathan/tmp/_claude_tmp/stage0
TR=/user/henrich1/u25995/jonathan/repos/CanViT-train
EV=/user/henrich1/u25995/jonathan/repos/CanViT-eval
export ADE20K_ROOT=/mnt/vast-nhr/projects/nib00021/jonathan/datasets/zhoubolei--scene_parse_150/ADEChallengeData2016
export HF_HOME="$HOME/.cache/huggingface" TORCH_HOME="$HOME/.cache/torch"
P="$S/synthetic_dv3_seg_probe"

echo "### A canvit_train ade20k-dinov3"
cd $TR && .venv-cu126/bin/python -m canvit_train.harness.evaluate ade20k-dinov3 \
  --opts.probe-repo "$P" --opts.eval-resolution 512 \
  --cfg.resize-mode squish --cfg.scene-size 512 --cfg.eval-batch-size 32 --cfg.num-workers 8 \
  --out "$S/gate4b_train_dinov3.json" >/dev/null

echo "### B canvit_eval ade20k-seg-dinov3 (the implementation being ported)"
cd $EV && .venv/bin/python -m canvit_eval ade20k-seg-dinov3 \
  --probe-repo "$P" --eval-resolution 512 \
  --scene-size 512 --resize-mode squish --batch-size 32 --num-workers 8 \
  --output "$S/gate4b_eval_dinov3.pt" 2>&1 | tail -2
echo "### ALLDONE"
