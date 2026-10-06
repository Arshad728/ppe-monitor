#!/usr/bin/env bash
# Phase 1 on the Mac: training data, training on the Apple GPU, evaluation, mistake review.
#
#   bash ~/Documents/ppe-monitor/scripts/mac_phase1.sh prepare   # ~10 min: data + a quick GPU training check
#   bash ~/Documents/ppe-monitor/scripts/mac_phase1.sh train     # full training, evaluation, mistake gallery
#   bash ~/Documents/ppe-monitor/scripts/mac_phase1.sh review    # ~1 min: rebuild the mistake gallery only
#
# Extra options after "train" go to scripts/train_detector.py, e.g.  ... train --epochs 40
# Keep the Mac plugged in with the lid open while training; macOS is told not to sleep meanwhile.
# Output is also saved to logs/phase1_<stage>.log.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
STAGE="${1:-}"
case "$STAGE" in
  prepare|train|review) shift ;;
  *) echo "usage: bash scripts/mac_phase1.sh prepare|train|review [training options]"; exit 2 ;;
esac
mkdir -p logs

# Only one Phase 1 run at a time. Two runs would share the GPU (each at about half speed), write
# the same model file and truncate each other's log - which is what happened on 23 Sep 2026.
LOCK="$ROOT/logs/.phase1.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  OTHER="$(cat "$LOCK/pid" 2>/dev/null)"
  if [ -n "$OTHER" ] && kill -0 "$OTHER" 2>/dev/null; then
    echo "Phase 1 is already running in another window ($(cat "$LOCK/what" 2>/dev/null), process $OTHER)."
    echo "Let it finish, or stop it with Ctrl+C in its window, then run this again."
    exit 1
  fi
  rm -rf "$LOCK" && mkdir "$LOCK" || { echo "Could not create $LOCK"; exit 1; }  # left over from a run that was killed
fi
echo $$ > "$LOCK/pid"
echo "$STAGE, started $(date '+%H:%M')" > "$LOCK/what"
trap 'rm -rf "$LOCK"' EXIT

LOG="$ROOT/logs/phase1_$STAGE.log"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

step() { printf '\n==== %s ====\n' "$*"; }
fail() { printf '\nSTOPPED: %s\nFull log: %s\n' "$*" "$LOG"; sleep 0.3; exit 1; }
echo "PPE monitor - Phase 1 ($STAGE), $(date '+%Y-%m-%d %H:%M:%S')"

# The project's Python: the conda env "ppe" or .venv that scripts/mac_setup.sh created.
PY=""
for c in "$(command -v conda 2>/dev/null)" /opt/anaconda3/bin/conda "$HOME/anaconda3/bin/conda" /opt/miniconda3/bin/conda \
         "$HOME/miniconda3/bin/conda" "$HOME/miniforge3/bin/conda"; do
  if [ -n "$c" ] && [ -x "$c" ]; then
    ENV_DIR="$("$c" env list 2>/dev/null | awk '$1=="ppe" {print $NF}')"
    if [ -n "$ENV_DIR" ] && [ -x "$ENV_DIR/bin/python" ]; then PY="$ENV_DIR/bin/python"; break; fi
  fi
done
[ -z "$PY" ] && [ -x .venv/bin/python ] && PY="$ROOT/.venv/bin/python"
[ -n "$PY" ] || fail "No project environment found. Run scripts/mac_setup.sh first."
PY_BIN_DIR="$(dirname "$PY")"
export PATH="$PY_BIN_DIR:$PATH"
echo "Python: $PY"

# Stop macOS from sleeping while a long step runs (only while it runs).
awake() { if command -v caffeinate >/dev/null 2>&1; then caffeinate -i "$@"; else "$@"; fi; }

if [ "$STAGE" = "prepare" ]; then
  step "1/3 Machine-learning packages (Ultralytics YOLO, PyTorch, torchvision)"
  "$PY" -m pip install -q -r requirements-ml.txt || fail "pip install failed (see above)"
  "$PY" -c 'import ultralytics, torch, torchvision, cv2, numpy; print("  ultralytics", ultralytics.__version__, "| torch", torch.__version__, "| torchvision", torchvision.__version__, "| opencv", cv2.__version__, "| numpy", numpy.__version__, "| Apple GPU:", torch.backends.mps.is_available())'

  step "2/3 Training data (download, merge, remove test-set leaks)"
  "$PY" scripts/prepare_data.py || fail "preparing the data failed (see above)"

  step "3/3 Quick training check on the GPU (1 epoch on a quarter of the data)"
  awake "$PY" scripts/train_detector.py --quick || fail "the quick training check failed (see above)"

  step "Done"
  echo "  Data report:   $ROOT/datasets/ppe3/report.md"
  echo "  Quick check:   $ROOT/logs/train_quick.txt"
  echo "  Next (after review):  bash scripts/mac_phase1.sh train"
elif [ "$STAGE" = "review" ]; then
  step "Mistake gallery on the test split"
  "$PY" scripts/review_mistakes.py "$@" || fail "the mistake review failed (see above)"
else
  [ -f datasets/ppe3/data.yaml ] || fail "No training data yet. Run: bash scripts/mac_phase1.sh prepare"

  step "1/2 Full training on the GPU (plugged in, lid open)"
  awake "$PY" scripts/train_detector.py "$@" || fail "training failed (see above)"

  step "2/2 Mistake gallery on the test split"
  "$PY" scripts/review_mistakes.py || fail "the mistake review failed (see above)"

  step "Done"
  echo "  Model card:      $(ls -t "$ROOT"/models/*.md 2>/dev/null | head -1)"
  echo "  Mistake gallery: $(ls -td "$ROOT"/runs/review/*/ 2>/dev/null | head -1)index.html   (open it in a browser)"
  echo "  Training curves: results.png in the newest folder under $ROOT/runs/detect/"
fi
sleep 0.3
