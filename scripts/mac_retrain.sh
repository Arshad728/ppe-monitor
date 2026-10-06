#!/usr/bin/env bash
# Fine-tune the detector with extra public datasets, so it also finds navy, brown and white helmets
# (docs/real_clips.md, docs/decisions/0007).
#
#   bash ~/Documents/ppe-monitor/scripts/mac_retrain.sh data      # ~20-30 min: download ~1.4 GB, check labels, build datasets/ppe4
#   bash ~/Documents/ppe-monitor/scripts/mac_retrain.sh train     # ~2-4 h: fine-tune (stops early when val stops improving)
#   bash ~/Documents/ppe-monitor/scripts/mac_retrain.sh compare   # ~10 min: old model against the new one
#   bash ~/Documents/ppe-monitor/scripts/mac_retrain.sh accept    # put the new model into use, if it passed
#
# `train` starts from the Phase 1 model (not from scratch) and guards against overtraining:
#   - at most 30 epochs, and it stops after 8 epochs without improvement on the validation photos;
#   - the checkpoint kept is the one best on validation, not the last;
#   - the model card (models/candidates/*.md) reports train against validation loss per epoch
#     and the gap between training and validation accuracy.
# The new model goes to models/candidates/ and is used only after `accept`.
# Keep the Mac plugged in during `train`; it is kept awake while the script runs.
# Output is also saved to logs/retrain_<stage>.log. Only one run at a time.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
STAGE="${1:-}"
case "$STAGE" in
  data|train|compare|accept) shift ;;
  *) echo "usage: bash scripts/mac_retrain.sh data|train|compare|accept"; exit 2 ;;
esac
mkdir -p logs

LOCK="$ROOT/logs/.retrain.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  OTHER="$(cat "$LOCK/pid" 2>/dev/null)"
  if [ -n "$OTHER" ] && kill -0 "$OTHER" 2>/dev/null; then
    echo "A retraining step is already running in another window ($(cat "$LOCK/what" 2>/dev/null), process $OTHER)."
    echo "Let it finish, or stop it with Ctrl+C in its window, then run this again."
    exit 1
  fi
  rm -rf "$LOCK" && mkdir "$LOCK" || { echo "Could not create $LOCK"; exit 1; }
fi
echo $$ > "$LOCK/pid"
echo "$STAGE, started $(date '+%H:%M')" > "$LOCK/what"
trap 'rm -rf "$LOCK"' EXIT

LOG="$ROOT/logs/retrain_$STAGE.log"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

step() { printf '\n==== %s ====\n' "$*"; }
fail() { printf '\nSTOPPED: %s\nFull log: %s\n' "$*" "$LOG"; sleep 0.3; exit 1; }
awake() { if command -v caffeinate >/dev/null 2>&1; then caffeinate -i "$@"; else "$@"; fi; }
echo "PPE monitor - retraining ($STAGE), $(date '+%Y-%m-%d %H:%M:%S')"

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
export PATH="$(dirname "$PY"):$PATH"
export PYTHONUNBUFFERED=1
echo "Python: $PY"
[ -f models/ppe3_yolo26n_baseline.pt ] || fail "No Phase 1 model (models/ppe3_yolo26n_baseline.pt)."
[ -f datasets/ppe3/data.yaml ] || fail "No ppe3 dataset. Phase 1 comes first: bash scripts/mac_phase1.sh prepare"

if [ "$STAGE" = "data" ]; then
  step "Extra datasets: download, check the labels, merge (datasets/ppe4)"
  awake "$PY" scripts/prepare_more_data.py || fail "building datasets/ppe4 failed (see above)"
  step "Done"
  echo "  Report:   $ROOT/datasets/ppe4/report.md"
  echo "  Pictures: $ROOT/runs/data_review/ppe4/ (kept images with their boxes; left-out ones with the reason)"
  echo "  Next: bash scripts/mac_retrain.sh train"
elif [ "$STAGE" = "train" ]; then
  [ -f datasets/ppe4/data.yaml ] || fail "No datasets/ppe4. Run: bash scripts/mac_retrain.sh data"
  step "Fine-tuning the Phase 1 model on datasets/ppe4 (at most 30 epochs, early stop after 8 without improvement)"
  echo "  Keep the Mac plugged in. It will not sleep while this runs; closing the lid may still pause it."
  awake "$PY" scripts/train_detector.py --model models/ppe3_yolo26n_baseline.pt --data datasets/ppe4/data.yaml \
      --epochs 30 --patience 8 --name finetune --out-name ppe4_yolo26n_finetune --save-dir models/candidates "$@" \
      || fail "training failed (see above)"
  step "Done"
  echo "  Model card (with the overfitting check): $ROOT/models/candidates/ppe4_yolo26n_finetune.md"
  echo "  Next: bash scripts/mac_retrain.sh compare"
elif [ "$STAGE" = "compare" ]; then
  step "The model in use against the fine-tuned one, on test photos neither trained on"
  awake "$PY" scripts/compare_models.py
  code=$?
  [ $code -eq 1 ] && fail "the comparison failed (see above)"
  step "Done"
  L="$(ls -td runs/compare/*/ | head -1)"
  echo "  Report: $ROOT/${L}report.md"
  [ $code -eq 0 ] && echo "  It passed. To use it: bash scripts/mac_retrain.sh accept" \
                  || echo "  It did not pass every check; the model in use stays."
else
  step "Putting the fine-tuned model into use (only if it passes the comparison)"
  awake "$PY" scripts/compare_models.py --accept || fail "not accepted (see above)"
  step "Making its Core ML and ONNX versions (Phase 4)"
  "$PY" scripts/export_models.py || fail "exporting the new model failed (see above)"
  step "Done"
fi
sleep 0.3
