#!/usr/bin/env bash
# Phase 6 on the Mac: measure the whole system, learn from false alarms, package it.
#
#   bash ~/Documents/ppe-monitor/scripts/mac_phase6.sh check        # ~3 min: tests, and a short end-to-end run
#   bash ~/Documents/ppe-monitor/scripts/mac_phase6.sh evaluate     # ~7 min: every test clip -> the five headline numbers
#   bash ~/Documents/ppe-monitor/scripts/mac_phase6.sh evaluate --telegram 5   # ... and time 5 real alerts on your phone
#   bash ~/Documents/ppe-monitor/scripts/mac_phase6.sh feedback     # the false alarms marked on the dashboard -> training data
#   bash ~/Documents/ppe-monitor/scripts/mac_phase6.sh retrain      # ~1-2 h: fine-tune on them, compare, evaluate
#   bash ~/Documents/ppe-monitor/scripts/mac_phase6.sh accept       # put the retrained model into use, if it passed
#   bash ~/Documents/ppe-monitor/scripts/mac_phase6.sh docker       # needs Docker: `docker compose up`, checked end to end
#   bash ~/Documents/ppe-monitor/scripts/mac_phase6.sh annotate data/clips/<clip>.mp4   # mark the violations in a clip
#
# `evaluate` plays every test clip once as a camera, 8 at a time, through the real system (camera
#   service, database, alert worker, a stand-in for Telegram) and matches each alert to the clip's
#   ground truth: scripts/evaluate_system.py. Its database is a throw-away one (ppe_eval); the
#   dashboard's data is untouched. With --telegram N, N alerts from the public test photos (never
#   your own footage) also go to your phone, marked "Evaluation", to time real delivery.
# `feedback` and `retrain` turn the events you marked "False alarm" on the dashboard into training
#   images (scripts/feedback.py): only from real cameras or clips listed as "train" in
#   data/clips/clips_catalog.csv, never from test clips. `retrain` needs at least 10 of them.
# `docker` needs Docker Desktop, OrbStack or Colima. Docker on a Mac can't use the Apple GPU, so the
#   stack runs on the CPU there; the native setup (mac_phase5.sh run) stays the fast way on this Mac.
# Output is also saved to logs/phase6_<stage>.log. Only one Phase 6 run at a time.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
STAGE="${1:-}"
case "$STAGE" in
  check|evaluate|feedback|retrain|accept|docker|annotate) shift ;;
  *) echo "usage: bash scripts/mac_phase6.sh check|evaluate [--telegram N]|feedback|retrain|accept|docker|annotate CLIP"; exit 2 ;;
esac
mkdir -p logs

LOCK="$ROOT/logs/.phase6.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  OTHER="$(cat "$LOCK/pid" 2>/dev/null)"
  if [ -n "$OTHER" ] && kill -0 "$OTHER" 2>/dev/null; then
    echo "Phase 6 is already running in another window ($(cat "$LOCK/what" 2>/dev/null), process $OTHER)."
    echo "Let it finish, or stop it with Ctrl+C in its window, then run this again."
    exit 1
  fi
  rm -rf "$LOCK" && mkdir "$LOCK" || { echo "Could not create $LOCK"; exit 1; }
fi
echo $$ > "$LOCK/pid"
echo "$STAGE, started $(date '+%H:%M')" > "$LOCK/what"
trap 'rm -rf "$LOCK"' EXIT

LOG="$ROOT/logs/phase6_$STAGE.log"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

step() { printf '\n==== %s ====\n' "$*"; }
fail() { printf '\nSTOPPED: %s\nFull log: %s\n' "$*" "$LOG"; sleep 0.3; exit 1; }
awake() { if command -v caffeinate >/dev/null 2>&1; then caffeinate -i "$@"; else "$@"; fi; }
echo "PPE monitor - Phase 6 ($STAGE), $(date '+%Y-%m-%d %H:%M:%S')"

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

if [ "$STAGE" = "check" ]; then
  need_db
  need_clips
  step "1/2 Tests"
  if external_db; then
    "$PY" -m pytest -q || fail "some tests failed (see above)"
  else
    TEST_URL="$("$PY" - <<'PYEOF'
import sys; sys.path.insert(0, "src")
from ppe_monitor.backend import localpg
from ppe_monitor.backend.settings import ServerSettings
print(localpg.ensure_database(ServerSettings.load(), "ppe_test", fresh=True))
PYEOF
)" || fail "could not make the test database"
    PPE_TEST_DATABASE_URL="$TEST_URL" "$PY" -m pytest -q || fail "some tests failed (see above)"
  fi
  step "2/2 A short end-to-end evaluation: 10 clips, 8 cameras at a time (~1 min)"
  awake "$PY" scripts/evaluate_system.py --sets still,zone,site --limit 4 || fail "the short evaluation failed (see above)"
  step "Done"
  echo "  Next: bash scripts/mac_phase6.sh evaluate"

elif [ "$STAGE" = "evaluate" ]; then
  need_db
  need_clips
  step "The test clips through the whole system, 8 cameras at a time (all 260 clips: ~7 min)"
  echo "  Keep the Mac plugged in, and leave it alone until this is done: other work on it slows the cameras down."
  awake "$PY" scripts/evaluate_system.py "$@" || fail "the evaluation failed (see above)"
  step "Done"
  R="$(ls -td runs/evaluation/*/ | head -1)"
  echo "  Report: $ROOT/${R}report.md"

elif [ "$STAGE" = "feedback" ]; then
  need_db
  step "False alarms marked on the dashboard -> datasets/feedback"
  "$PY" scripts/feedback.py export "$@" || fail "the export failed (see above)"
  [ -f datasets/feedback/review.html ] && command -v open >/dev/null 2>&1 && open datasets/feedback/review.html
  step "Done"
  echo "  Review:  $ROOT/datasets/feedback/review.html  (to leave one out, add its id to datasets/feedback/skip.txt)"
  echo "  Next:    bash scripts/mac_phase6.sh retrain   (needs at least 10 ready)"

elif [ "$STAGE" = "retrain" ]; then
  need_db
  need_clips
  [ -f datasets/ppe4/data.yaml ] || fail "No datasets/ppe4 (the training data). Run: bash scripts/mac_retrain.sh data"
  step "1/5 The training data: the model in use's (from its model card) plus the ready false alarms (datasets/ppe5)"
  "$PY" scripts/feedback.py build
  code=$?
  if [ $code -eq 3 ]; then
    step "Nothing to retrain on yet"
    echo "  Mark false alarms on the dashboard as they happen (on real cameras, or on clips listed as train"
    echo "  in data/clips/clips_catalog.csv), then: bash scripts/mac_phase6.sh feedback, and this again."
    exit 0
  fi
  [ $code -eq 0 ] || fail "building datasets/ppe5 failed (see above)"
  IN_USE="$(model_in_use)"
  step "2/5 Fine-tuning $(basename "$IN_USE") on it (at most 10 epochs, early stop after 4 without improvement)"
  echo "  Keep the Mac plugged in. It will not sleep while this runs."
  awake "$PY" scripts/train_detector.py --model "$IN_USE" --data datasets/ppe5/data.yaml --epochs 10 --patience 4 \
      --name feedback --out-name ppe5_yolo26n_feedback --save-dir models/candidates || fail "training failed (see above)"
  CAND="models/candidates/ppe5_yolo26n_feedback.pt"
  step "3/5 Core ML version of the candidate (so both models are evaluated the same way)"
  "$PY" scripts/export_models.py --weights "$CAND" --backend coreml || fail "exporting the candidate failed"
  step "4/5 Both models on every test clip, end to end"
  OLD_EVAL="$("$PY" - "$IN_USE" <<'PYEOF'
import json, sys
from pathlib import Path
stem = Path(sys.argv[1]).stem
runs = sorted(Path("runs/evaluation").glob("*/metrics.json"), reverse=True)
ok = [r for r in runs if json.loads(r.read_text())["model"] == stem and json.loads(r.read_text())["clips"] >= 200]
print(ok[0] if ok else "")
PYEOF
)"
  if [ -z "$OLD_EVAL" ]; then
    awake "$PY" scripts/evaluate_system.py || fail "evaluating the model in use failed"
    OLD_EVAL="$(ls -td runs/evaluation/*/ | head -1)metrics.json"
  fi
  NEW_DIR="runs/evaluation/$(date +%Y%m%d_%H%M%S)_candidate"
  awake "$PY" scripts/evaluate_system.py --weights "$CAND" --out "$NEW_DIR" || fail "evaluating the candidate failed"
  step "5/5 The candidate against the model in use"
  awake "$PY" scripts/compare_models.py --new "$CAND" --checks feedback --evaluations "$OLD_EVAL" "$NEW_DIR/metrics.json"
  code=$?
  [ $code -eq 1 ] && fail "the comparison failed (see above)"
  echo "$OLD_EVAL $NEW_DIR/metrics.json" > runs/evaluation/.last_retrain
  step "Done"
  [ $code -eq 0 ] && echo "  It passed. To use it: bash scripts/mac_phase6.sh accept" \
                  || echo "  It did not pass every check; the model in use stays."

elif [ "$STAGE" = "accept" ]; then
  [ -f runs/evaluation/.last_retrain ] || fail "No retrained candidate yet. Run: bash scripts/mac_phase6.sh retrain"
  read -r OLD_EVAL NEW_EVAL < runs/evaluation/.last_retrain
  step "Putting the retrained model into use (only if it passes the comparison)"
  awake "$PY" scripts/compare_models.py --new models/candidates/ppe5_yolo26n_feedback.pt --checks feedback \
      --evaluations "$OLD_EVAL" "$NEW_EVAL" --accept || fail "not accepted (see above)"
  step "Making its Core ML and ONNX versions"
  "$PY" scripts/export_models.py || fail "exporting the new model failed (see above)"
  step "Done"

elif [ "$STAGE" = "docker" ]; then
  if ! command -v docker >/dev/null 2>&1; then
    step "Docker isn't installed on this Mac"
    echo "  The Compose setup is optional here: it has been checked on a Linux machine (docs/phase6_results.md),"
    echo "  and on a Mac, Docker can't use the Apple GPU, so the native setup stays the fast way."
    echo "  To try it anyway, install one of these, start it, and run this again:"
    echo "    OrbStack        https://orbstack.dev  (small and fast)"
    echo "    Docker Desktop  https://www.docker.com/products/docker-desktop/"
    exit 0
  fi
  docker info >/dev/null 2>&1 || fail "Docker is installed but not running. Start Docker Desktop / OrbStack, then run this again."
  step "docker compose up (the first time builds the image: 5-15 min), then checks through the dashboard"
  awake "$PY" scripts/check_compose.py "$@" || fail "the Docker Compose check failed (see above)"
  step "Done"

else
  [ -n "${1:-}" ] || fail "usage: bash scripts/mac_phase6.sh annotate data/clips/<clip>.mp4"
  step "Mark the violations in $1 (a window opens; the keys are listed in this window)"
  "$PY" scripts/annotate_clip.py "$@" || fail "the annotation tool stopped with an error (see above)"
  step "Done"
  echo "  Then: bash scripts/mac_phase6.sh evaluate --sets site   (your clips only)"
fi
sleep 0.3
