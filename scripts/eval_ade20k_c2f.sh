#!/bin/bash
# Re-evaluate all four finished ADE20K probes of a run group under COARSE-TO-FINE.
#
# The probe groups train and validate under `random` viewpoints (ADE20K's historical
# default, inherited from the specialize probe, which trained on random views). This
# answers the separate question "what do these models score under the C2F deploy
# convention?".
#
# n_timesteps=21 = canvit_eval's EpisodeConfig default for policy="coarse_to_fine", so
# these are the numbers CanViT-eval would report.
#
# The two FOVEATED arms pass --override-scale 2.0: C2F's quadtree scales {1.0, 0.5, 0.25}
# are ones a fixed-scale foveated backbone never trained on (`fix_size = scale * H`), so
# unpinned every glimpse is out of distribution. Pinning keeps C2F's CENTERS and fixes the
# scale at the pretraining value -- canvit_eval's `override_scale` semantics.
#
# The four backbones below are fixed: every probe group is trained from the same four
# exp22 pretrains, so only the probe run group changes between campaigns.
#
# Usage: bash scripts/eval_ade20k_c2f.sh [run_group] [output_dir]
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GROUP=${1:-jon_exp34_ade20k_probe}
# Run artifacts follow $LOGS_DIR, not this clone: a second project member points LOGS_DIR
# at a directory they own (README, first-time setup), so their probe runs are not under
# ./logs at all. Falls back to ./logs, which is what LOGS_DIR is for the owner.
RUNS_DIR="${LOGS_DIR:-$PWD/logs}"
OUT=${2:-$RUNS_DIR/$GROUP/_c2f_eval}
mkdir -p "$OUT"

export ADE20K_ROOT=${ADE20K_ROOT:-/mnt/vast-nhr/projects/nib00021/jonathan/datasets/zhoubolei--scene_parse_150/ADEChallengeData2016}
export HF_HUB_OFFLINE=1
PY=.venv-cu126/bin/python
# ABSOLUTE, not derived from this clone: logs/ is gitignored, so the four exp22 backbones
# exist in exactly one place on the cluster (group-readable to HPC_nib00021) and a second
# member's clone has no logs/ at all. Override with EXP22_DIR if you hold your own copies.
SRC="${EXP22_DIR:-/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit/logs/jon_exp22_full_runs}"

# run_name : pretrained backbone the probe was trained on : override-scale ("" = uniform)
RUNS=(
  "ade20k-uni16ti-803k:exp22-uniform16-teacherinit-lrdrop2-803k/checkpoints/step-16384-hf:"
  "ade20k-uni16-1516k:exp22-uniform16-lrdrop-1516k/checkpoints/step-319488-hf:"
  "ade20k-fovi-ti-1196k:exp22-fovi-teacherinit-lrdrop-1196k/checkpoints/step-155648-hf:2.0"
  "ade20k-fovi-1901k:exp22-fovi/checkpoints/step-1900544-hf:2.0"
)

for entry in "${RUNS[@]}"; do
    IFS=: read -r run repo scale <<<"$entry"
    echo "=== $run ${scale:+(foveated, scale pinned to $scale)} ==="
    extra=()
    if [ -n "$scale" ]; then
        extra=(--cfg.eval-override-scale "$scale" --cfg.foveated-scale.fixed-scale "$scale")
    fi
    # `harness.evaluate` replaced scripts/eval_ade20k_checkpoint.py (2026-09-02): the
    # per-task driver became one entry point for all three tasks, proven bit-identical
    # against this script's own numbers. The loop over arms stays HERE, in the shell --
    # one measurement per invocation, one artifact each (eval-merge doc §5, Stage 3).
    # eval-batch-size is pinned to 16 rather than left at the config default of 32: C2F
    # shuffles within each quadtree level, so the batch size changes the RNG pattern and
    # this script's earlier outputs were taken at 16.
    $PY -m canvit.harness.evaluate ade20k \
        --opts.ckpt "$RUNS_DIR/$GROUP/$run/checkpoints/best.pt" \
        --opts.out "$OUT/$run.json" \
        --cfg.model-repo "$SRC/$repo" \
        --cfg.eval-policy coarse_to_fine --cfg.n-timesteps 21 \
        --cfg.eval-batch-size 16 --cfg.num-workers 8 \
        "${extra[@]}" >/dev/null
done

echo "=== done -> $OUT ==="
