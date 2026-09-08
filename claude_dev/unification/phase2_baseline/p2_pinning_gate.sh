#!/bin/bash
# P2 gate: does the pinning mechanism still work on BOTH sides of the core merge?
set -uo pipefail
W="$1"; REPO_BASE=/user/henrich1/u25995/jonathan/repos
VENV="$REPO_BASE/CanViT-train/.venv-cu126/bin/python"
rm -rf "$W"; mkdir -p "$W"

# ---------- Case A: a POST-merge pin. TRAIN_COMMIT alone must supply model + trainer ------
A="$W/postmerge"; mkdir -p "$A/CanViT-train"
NEW=$(git -C "$REPO_BASE/CanViT-train" rev-parse HEAD)
git -C "$REPO_BASE/CanViT-train" archive "$NEW" | tar -x -C "$A/CanViT-train"
echo "### Case A: post-merge snapshot $NEW"
PYTHONSAFEPATH=1 PYTHONPATH="$A/CanViT-train" "$VENV" -c "
import canvit, canvit.core, canvit.harness.cli
print('  canvit      ->', canvit.__file__)
print('  canvit.core ->', canvit.core.__file__)
assert canvit.__file__.startswith('$A'), 'canvit did NOT come from the snapshot'
assert canvit.core.__file__.startswith('$A'), 'canvit.core did NOT come from the snapshot'
print('  OK: both resolve inside the pinned snapshot')
" 2>&1 | grep -vE "Warning|warn"

# ---------- Case B: a PRE-merge pin, exp32's real pair. Old core must still win -----------
B="$W/premerge"; mkdir -p "$B/CanViT-train" "$B/CanViT-PyTorch"
OLD_TRAIN=716051a; OLD_CORE=d616b7b
git -C "$REPO_BASE/CanViT-train"   archive "$OLD_TRAIN" | tar -x -C "$B/CanViT-train"
git -C "$REPO_BASE/CanViT-PyTorch" archive "$OLD_CORE"  | tar -x -C "$B/CanViT-PyTorch"
echo "### Case B: pre-merge pins TRAIN=$OLD_TRAIN CORE=$OLD_CORE (exp32-fovi-teacherinit)"
echo "  snapshot package dir: $(ls -d "$B/CanViT-train"/canvit* | xargs -n1 basename | tr '\n' ' ')"
# sbatch order: CanViT-train prepended first, then CanViT-PyTorch -> core wins ties
PYTHONSAFEPATH=1 PYTHONPATH="$B/CanViT-PyTorch:$B/CanViT-train" "$VENV" -c "
import canvit_train, canvit_pytorch
print('  canvit_train   ->', canvit_train.__file__)
print('  canvit_pytorch ->', canvit_pytorch.__file__)
assert canvit_train.__file__.startswith('$B'), 'old trainer did not come from its snapshot'
assert canvit_pytorch.__file__.startswith('$B/CanViT-PyTorch'), 'old core did NOT win'
print('  OK: the pinned OLD core wins, not the working tree and not the clone')
" 2>&1 | grep -vE "Warning|warn"

# ---------- Case C: the two guard branches actually fire ----------------------------------
echo "### Case C: guard messages"
guard() {  # $1 = _CODE_DIR, $2 = PYTORCH_COMMIT value
    local _CODE_DIR="$1" PYTORCH_COMMIT="$2" _PKG=canvit
    log() { echo "    $*"; }
    for _cand in canvit_train canvit_pretrain; do
        [ -d "$_CODE_DIR/CanViT-train/$_cand" ] && { _PKG=$_cand; break; }
    done
    echo "  _PKG=$_PKG PYTORCH_COMMIT='${PYTORCH_COMMIT}'"
    if [ "$_PKG" = canvit ]; then
        [ -n "${PYTORCH_COMMIT:-}" ] && log "WARNING: PYTORCH_COMMIT has NO EFFECT (post-merge pin)"
    else
        [ -z "${PYTORCH_COMMIT:-}" ] && log "WARNING: model NOT pinned (pre-merge pin, no PYTORCH_COMMIT)"
    fi
}
guard "$A" "d616b7b"   # post-merge snapshot + a stray PYTORCH_COMMIT -> should warn
guard "$A" ""          # post-merge, clean                           -> silent
guard "$B" ""          # pre-merge, no core pin                      -> should warn
guard "$B" "d616b7b"   # pre-merge, correctly pinned                 -> silent
echo "### ALLDONE"
