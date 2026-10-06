#!/usr/bin/env bash
# Phase 2 on the Mac: tracking people and deciding who is wearing a helmet and vest.
#
#   bash ~/Documents/ppe-monitor/scripts/mac_phase2.sh check   # ~15 min: tests, rule check, three event checks
#   bash ~/Documents/ppe-monitor/scripts/mac_phase2.sh demo    # watch it work on a test clip (press q to close)
#   bash ~/Documents/ppe-monitor/scripts/mac_phase2.sh demo datasets/event_clips_sway/005_violation.mp4
#
# `check` runs, in order:
#   1. the test suite (including walking, bending and crouching workers, with made-up detections);
#   2. the PPE rule check on the test split: how often one frame gets a person's helmet / vest wrong,
#      by person size and by posture (upright / bending / crouching / lying);
#   3. 86 short test clips made from test photos (slow pan, people still), run through the whole chain;
#   4. the same photos with the view swinging side to side, so everyone moves at walking speed;
#   5. those moving clips again, processing only every 3rd frame (5 fps), like a busy machine.
# The first run downloads the keypoint model (yolo26n-pose.pt, ~7 MB) into models/pretrained/.
# Output is also saved to logs/phase2_<stage>.log. Only one Phase 2 run at a time.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
STAGE="${1:-}"
case "$STAGE" in
  check|demo) shift ;;
  *) echo "usage: bash scripts/mac_phase2.sh check|demo [clip.mp4]"; exit 2 ;;
esac
mkdir -p logs

LOCK="$ROOT/logs/.phase2.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  OTHER="$(cat "$LOCK/pid" 2>/dev/null)"
  if [ -n "$OTHER" ] && kill -0 "$OTHER" 2>/dev/null; then
    echo "Phase 2 is already running in another window ($(cat "$LOCK/what" 2>/dev/null), process $OTHER)."
    echo "Let it finish, or stop it with Ctrl+C in its window, then run this again."
    exit 1
  fi
  rm -rf "$LOCK" && mkdir "$LOCK" || { echo "Could not create $LOCK"; exit 1; }
fi
echo $$ > "$LOCK/pid"
echo "$STAGE, started $(date '+%H:%M')" > "$LOCK/what"
trap 'rm -rf "$LOCK"' EXIT

LOG="$ROOT/logs/phase2_$STAGE.log"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

step() { printf '\n==== %s ====\n' "$*"; }
fail() { printf '\nSTOPPED: %s\nFull log: %s\n' "$*" "$LOG"; sleep 0.3; exit 1; }
echo "PPE monitor - Phase 2 ($STAGE), $(date '+%Y-%m-%d %H:%M:%S')"

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
ls models/*.pt >/dev/null 2>&1 || fail "No trained model in models/. Phase 1 comes first: bash scripts/mac_phase1.sh train"
[ -f datasets/ppe3/data.yaml ] || fail "No ppe3 dataset. Phase 1 comes first: bash scripts/mac_phase1.sh prepare"

if [ "$STAGE" = "check" ]; then
  step "1/5 Tests"
  "$PY" -m pytest -q || fail "some tests failed (see above)"

  step "2/5 PPE rules on people with a known status (test split)"
  "$PY" scripts/check_ppe_rules.py --split test || fail "the rule check failed (see above)"

  step "3/5 Whole chain on test clips: detector -> tracker -> PPE rules -> events"
  if [ ! -f datasets/event_clips/clips.json ]; then
    "$PY" scripts/make_event_clips.py || fail "making the test clips failed (see above)"
  fi
  "$PY" scripts/evaluate_events.py || fail "the event check failed (see above)"

  step "4/5 Moving people: the view swings side to side at walking speed"
  if [ ! -f datasets/event_clips_sway/clips.json ]; then
    "$PY" scripts/make_event_clips.py --motion sway || fail "making the moving clips failed (see above)"
  fi
  "$PY" scripts/evaluate_events.py --clips datasets/event_clips_sway || fail "the moving-clip check failed (see above)"

  step "5/5 Moving people, every 3rd frame only (5 fps)"
  "$PY" scripts/evaluate_events.py --clips datasets/event_clips_sway --stride 3 || fail "the 5 fps check failed (see above)"

  step "Done"
  M="$(ls -t models/*.pt | head -1)"; M="$(basename "${M%.pt}")"
  echo "  Rule check:        $ROOT/runs/ppe_rules/${M}_test.md"
  echo "  Still people:      $ROOT/runs/events/$M/event_clips/report.md"
  echo "  Moving people:     $ROOT/runs/events/$M/event_clips_sway/report.md"
  echo "  Moving, 5 fps:     $ROOT/runs/events/$M/event_clips_sway_every3/report.md"
else
  CLIP="${1:-datasets/event_clips/005_violation.mp4}"
  if [ ! -f "$CLIP" ]; then
    case "$CLIP" in
      *event_clips_sway*) "$PY" scripts/make_event_clips.py --motion sway || fail "making the moving clips failed (see above)" ;;
      *) "$PY" scripts/make_event_clips.py || fail "making the test clips failed (see above)" ;;
    esac
  fi
  step "Watching $CLIP (press q in the window to close)"
  "$PY" scripts/monitor.py --source "$CLIP" || fail "the monitor stopped with an error (see above)"
fi
sleep 0.3
