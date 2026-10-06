#!/usr/bin/env bash
# Teach the detector that caps and hats are not helmets (after Phase 6's real-video check).
#
#   bash ~/Documents/ppe-monitor/scripts/mac_caps.sh run        # ~2-3 h: data, train and compare, one after another
#   bash ~/Documents/ppe-monitor/scripts/mac_caps.sh accept     # put the new model into use, if it passed every check
#   bash ~/Documents/ppe-monitor/scripts/mac_caps.sh accept --override ["reason"]
#                                              # ... or although a check failed, when you have decided it is worth it:
#                                              #     the report and the model card record the failed checks and why
#
# or one stage at a time:
#   bash ~/Documents/ppe-monitor/scripts/mac_caps.sh data       # ~10 min: ~1,700 Open Images photos (~570 MB) -> datasets/ppe4caps
#   bash ~/Documents/ppe-monitor/scripts/mac_caps.sh train      # ~1.5-2 h: fine-tune the model in use (at most 15 epochs)
#   bash ~/Documents/ppe-monitor/scripts/mac_caps.sh compare    # ~30 min: both models on test photos and every test clip
#
# `data` downloads photos of people in hats and caps from Open Images (Google's dataset; photos by
#   Flickr authors, CC BY 2.0), checks each against its SHA-256 in data/caps/openimages_caps.json,
#   and builds datasets/ppe4caps = datasets/ppe4 + those photos (training and validation only; the
#   held-out ones are a separate test list). How they were chosen: src/ppe_monitor/data/caps.py.
# `train` guards against overtraining: it starts from the model in use, runs at most 15 epochs and
#   stops after 5 without improvement on validation (which mixes the old photos and the new ones),
#   keeps the best checkpoint, not the last, and its model card reports training against
#   validation accuracy (the overfitting check). It fine-tunes gently: a small, falling learning
#   rate (AdamW, 0.0002 -> 0.00002) and no large start for the biases. The first attempt (29 Sep,
#   Ultralytics' defaults) knocked the helmet scores down and was not accepted: docs/caps_results.md.
# `compare` accepts the candidate only if it is no worse on every test set the models were compared
#   on before, still finds helmets as well, calls far fewer hats helmets on photos by photographers
#   it never saw, memorises no more than the model in use, and, end to end on every test clip
#   (with your real video), raises no more false alarms and misses at most 2 more violations.
# Keep the Mac plugged in; it is kept awake while this runs. Output: logs/caps_<stage>.log.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
STAGE="${1:-}"
case "$STAGE" in
  data|train|compare|accept|run) shift ;;
  *) echo "usage: bash scripts/mac_caps.sh run|accept   (or one stage: data|train|compare)"; exit 2 ;;
esac
mkdir -p logs

for other in phase6 retrain; do
  O="$ROOT/logs/.$other.lock"
  if [ -d "$O" ] && kill -0 "$(cat "$O/pid" 2>/dev/null)" 2>/dev/null; then
    echo "Another job is running ($other: $(cat "$O/what" 2>/dev/null)). Let it finish, then run this again."
    exit 1
  fi
done
LOCK="$ROOT/logs/.caps.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  OTHER="$(cat "$LOCK/pid" 2>/dev/null)"
  if [ -n "$OTHER" ] && kill -0 "$OTHER" 2>/dev/null; then
    echo "This is already running in another window ($(cat "$LOCK/what" 2>/dev/null), process $OTHER)."
    echo "Let it finish, or stop it with Ctrl+C in its window, then run this again."
    exit 1
  fi
  rm -rf "$LOCK" && mkdir "$LOCK" || { echo "Could not create $LOCK"; exit 1; }
fi
echo $$ > "$LOCK/pid"
echo "$STAGE, started $(date '+%H:%M')" > "$LOCK/what"
trap 'rm -rf "$LOCK"' EXIT

LOG="$ROOT/logs/caps_$STAGE.log"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

step() { printf '\n==== %s ====\n' "$*"; }
fail() { printf '\nSTOPPED: %s\nFull log: %s\n' "$*" "$LOG"; sleep 0.3; exit 1; }
awake() { if command -v caffeinate >/dev/null 2>&1; then caffeinate -i "$@"; else "$@"; fi; }
echo "PPE monitor - caps and hats are not helmets ($STAGE), $(date '+%Y-%m-%d %H:%M:%S')"

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

external_db() { "$PY" - <<'PYEOF' 2>/dev/null
import sys; sys.path.insert(0, "src")
from ppe_monitor.backend.settings import ServerSettings
s = ServerSettings.load().secrets
sys.exit(0 if s.get("DATABASE_URL") or s.get("PPE_DB_HOST") else 1)
PYEOF
}
need_db() {
  "$PY" -c "import sqlalchemy, psycopg, fastapi" 2>/dev/null || fail "Phase 5 isn't set up. Run: bash scripts/mac_phase5.sh setup"
  if ! external_db; then
    [ -f data/postgres/PG_VERSION ] || fail "No database yet. Run: bash scripts/mac_phase5.sh setup"
    "$PY" scripts/db.py start || fail "the database server did not start (see logs/postgres.log)"
  fi
}
need_clips() {
  "$PY" -c "import lap" 2>/dev/null || "$PY" -m pip install -q "lap==0.5.13" || fail "installing lap failed"
  [ -f datasets/event_clips/clips.json ] || { "$PY" scripts/make_event_clips.py || fail "making the still clips failed"; }
  [ -f datasets/event_clips_sway/clips.json ] || { "$PY" scripts/make_event_clips.py --motion sway || fail "making the moving clips failed"; }
  [ -f datasets/event_clips_zone/clips.json ] || { "$PY" scripts/make_event_clips.py --motion pan || fail "making the zone clips failed"; }
}
model_in_use() { ls -t models/*.pt 2>/dev/null | head -1; }
CAND="models/candidates/ppe4caps_yolo26n_caps2.pt"

do_data() {
  [ -f datasets/ppe4/data.yaml ] || fail "No datasets/ppe4 (the current training data). Run: bash scripts/mac_retrain.sh data"
  step "Photos of people in hats and caps: download, check, add to a copy of datasets/ppe4 (datasets/ppe4caps)"
  awake "$PY" scripts/prepare_caps_data.py || fail "building datasets/ppe4caps failed (see above)"
}

do_train() {
  [ -f datasets/ppe4caps/data.yaml ] || fail "No datasets/ppe4caps. Run: bash scripts/mac_caps.sh data"
  IN_USE="$(model_in_use)"
  step "Fine-tuning $(basename "$IN_USE") on datasets/ppe4caps, gently (at most 15 epochs, early stop after 5 without improvement)"
  echo "  Keep the Mac plugged in. It will not sleep while this runs; closing the lid may still pause it."
  awake "$PY" scripts/train_detector.py --model "$IN_USE" --data datasets/ppe4caps/data.yaml --epochs 15 --patience 5 \
      --optimizer AdamW --lr0 0.0002 --lrf 0.1 --warmup-epochs 1 --warmup-bias-lr 0 --close-mosaic 3 \
      --name caps2 --out-name ppe4caps_yolo26n_caps2 --save-dir models/candidates \
      || fail "training failed (see above)"
  echo "  Model card (with the overfitting check): $ROOT/models/candidates/ppe4caps_yolo26n_caps2.md"
}

do_compare() {
  [ -f "$CAND" ] || fail "No candidate yet ($CAND). Run: bash scripts/mac_caps.sh train"
  need_db
  need_clips
  IN_USE="$(model_in_use)"
  step "1/3 Core ML version of the candidate (so both models are evaluated the same way)"
  "$PY" scripts/export_models.py --weights "$CAND" --backend coreml || fail "exporting the candidate failed"
  step "2/3 Both models on every test clip, end to end (the model in use first)"
  STAMP="$(date +%Y%m%d_%H%M%S)"
  OLD_DIR="runs/evaluation/${STAMP}_in_use"
  NEW_DIR="runs/evaluation/${STAMP}_caps_candidate"
  awake "$PY" scripts/evaluate_system.py --weights "$IN_USE" --out "$OLD_DIR" || fail "evaluating the model in use failed"
  awake "$PY" scripts/evaluate_system.py --weights "$CAND" --out "$NEW_DIR" || fail "evaluating the candidate failed"
  step "3/3 The candidate against the model in use, on test photos and clips neither trained on"
  awake "$PY" scripts/compare_models.py --new "$CAND" --checks caps --evaluations "$OLD_DIR/metrics.json" "$NEW_DIR/metrics.json"
  code=$?
  [ $code -eq 1 ] && fail "the comparison failed (see above)"
  echo "$OLD_DIR/metrics.json $NEW_DIR/metrics.json" > runs/evaluation/.last_caps
  L="$(ls -td runs/compare/*/ | head -1)"
  echo "  Report: $ROOT/${L}report.md"
  echo "  End to end: $ROOT/$OLD_DIR/report.md and $ROOT/$NEW_DIR/report.md"
  [ $code -eq 0 ] && echo "  It passed every check. To use it: bash scripts/mac_caps.sh accept" \
                  || echo "  It did not pass every check; the model in use stays."
  return 0
}

if [ "$STAGE" = "data" ]; then
  do_data
elif [ "$STAGE" = "train" ]; then
  do_train
elif [ "$STAGE" = "compare" ]; then
  do_compare
elif [ "$STAGE" = "run" ]; then
  do_data
  do_train
  do_compare
else
  [ -f runs/evaluation/.last_caps ] || fail "No comparison yet. Run: bash scripts/mac_caps.sh compare"
  read -r OLD_EVAL NEW_EVAL < runs/evaluation/.last_caps
  ARGS=()
  if [ "${1:-}" = "--override" ]; then
    ARGS=(--override "${2:-accepted by $(whoami) on $(date +%Y-%m-%d) although a check failed; why: docs/caps_results.md}")
    step "Putting the new model into use, over a failed check (your decision)"
  else
    step "Putting the new model into use (only if it passes every check)"
  fi
  awake "$PY" scripts/compare_models.py --new "$CAND" --checks caps --evaluations "$OLD_EVAL" "$NEW_EVAL" --accept \
      ${ARGS[@]+"${ARGS[@]}"} || fail "not accepted (see above)"
  step "Making its Core ML and ONNX versions (Phase 4)"
  "$PY" scripts/export_models.py || fail "exporting the new model failed (see above)"
fi
step "Done"
sleep 0.3
