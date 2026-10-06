#!/usr/bin/env bash
# Phase 3 on the Mac: restricted zones and the rules engine.
#
#   bash ~/Documents/ppe-monitor/scripts/mac_phase3.sh check        # ~10 min: tests, rules, zone clips, cameras
#   bash ~/Documents/ppe-monitor/scripts/mac_phase3.sh demo         # watch the simulated gate camera with its zone
#   bash ~/Documents/ppe-monitor/scripts/mac_phase3.sh demo cam3    # ... or another simulated camera
#   bash ~/Documents/ppe-monitor/scripts/mac_phase3.sh draw cam1    # try the zone drawing tool on a camera
#
# `check` runs, in order:
#   1. the test suite (zones, rules, active hours, with made-up people);
#   2. the rules check: configs/rules.yaml and configs/zones.yaml, what applies to each camera, and a
#      picture of each camera's zones in runs/zones/;
#   3. 86 test clips in which people walk into a restricted zone and stay, or never reach it; every
#      zone event (and every PPE event) is scored;
#   4. the simulated cameras' clips (Phase 0): cam1's keep-out zone, and cam3's pit, whose rule is
#      only on 07:00-19:00, played as if at noon and as if just before 19:00.
# Output is also saved to logs/phase3_<stage>.log. Only one Phase 3 run at a time.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
STAGE="${1:-}"
case "$STAGE" in
  check|demo|draw) shift ;;
  *) echo "usage: bash scripts/mac_phase3.sh check|demo|draw [camera]"; exit 2 ;;
esac
mkdir -p logs

LOCK="$ROOT/logs/.phase3.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  OTHER="$(cat "$LOCK/pid" 2>/dev/null)"
  if [ -n "$OTHER" ] && kill -0 "$OTHER" 2>/dev/null; then
    echo "Phase 3 is already running in another window ($(cat "$LOCK/what" 2>/dev/null), process $OTHER)."
    echo "Let it finish, or stop it with Ctrl+C in its window, then run this again."
    exit 1
  fi
  rm -rf "$LOCK" && mkdir "$LOCK" || { echo "Could not create $LOCK"; exit 1; }
fi
echo $$ > "$LOCK/pid"
echo "$STAGE, started $(date '+%H:%M')" > "$LOCK/what"
trap 'rm -rf "$LOCK"' EXIT

LOG="$ROOT/logs/phase3_$STAGE.log"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

step() { printf '\n==== %s ====\n' "$*"; }
fail() { printf '\nSTOPPED: %s\nFull log: %s\n' "$*" "$LOG"; sleep 0.3; exit 1; }
echo "PPE monitor - Phase 3 ($STAGE), $(date '+%Y-%m-%d %H:%M:%S')"

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


[ -f data/clips/synthetic_cam1.mp4 ] || { "$PY" scripts/make_test_clips.py || fail "making the simulated camera clips failed"; }

events_summary() {  # $1 = folder with events.jsonl: one line per event kind
  "$PY" - "$1" <<'PYEOF'
import collections, json, pathlib, sys
f = pathlib.Path(sys.argv[1]) / "events.jsonl"
evs = [json.loads(l) for l in f.read_text().splitlines()] if f.is_file() else []
for e in evs:
    print(f"    {e['kind']:15s} rule {e['rule']:18s} {e['severity']:8s} track #{e['track_id']:<3d} "
          f"t={e['confirmed']:5.1f}s  {e['time']}" + (f"  zone {e['zone']}" if e['zone'] else ""))
n = collections.Counter(e["kind"] for e in evs)
print(f"    -> zone events: {n['zone_intrusion']}, PPE events: {n['no_helmet'] + n['no_vest']}")
PYEOF
}

if [ "$STAGE" = "check" ]; then
  step "1/4 Tests"
  "$PY" -m pytest -q || fail "some tests failed (see above)"

  step "2/4 Rules and zones"
  "$PY" scripts/check_rules.py || fail "configs/rules.yaml or configs/zones.yaml has a problem (see above)"

  step "3/4 Test clips: people walking into a restricted zone"
  if [ ! -f datasets/event_clips_zone/clips.json ]; then
    "$PY" scripts/make_event_clips.py --motion pan || fail "making the zone clips failed (see above)"
  fi
  "$PY" scripts/evaluate_events.py --clips datasets/event_clips_zone || fail "the zone clip check failed (see above)"

  step "4/4 Simulated cameras (cartoon clips: only the zones matter here)"
  OUT="$ROOT/runs/phase3"
  mkdir -p "$OUT"
  echo "  cam1, keep-out zone, rule always on (worker C walks in at about 8 s):"
  "$PY" scripts/monitor.py --source data/clips/synthetic_cam1.mp4 --as-camera cam1 --headless \
      --out "$OUT/cam1" --save "$OUT/cam1.mp4" >/dev/null || fail "the monitor failed on cam1"
  events_summary "$OUT/cam1"
  echo "  cam3, pit zone, rule on 07:00-19:00 Mon-Sat, played as if at noon on a Monday:"
  "$PY" scripts/monitor.py --source data/clips/synthetic_cam3.mp4 --as-camera cam3 --headless \
      --start-time "2026-09-21 12:00:00" --out "$OUT/cam3_noon" >/dev/null || fail "the monitor failed on cam3"
  events_summary "$OUT/cam3_noon"
  echo "  cam3 again, as if starting at 18:59:52: the rule switches off at 19:00, 8 s in, as worker C arrives:"
  "$PY" scripts/monitor.py --source data/clips/synthetic_cam3.mp4 --as-camera cam3 --headless \
      --start-time "2026-09-21 18:59:52" --out "$OUT/cam3_evening" >/dev/null || fail "the monitor failed on cam3"
  events_summary "$OUT/cam3_evening"

  step "Done"
  M="$(ls -t models/*.pt | head -1)"; M="$(basename "${M%.pt}")"
  echo "  Zone pictures:     $ROOT/runs/zones/"
  echo "  Zone clip report:  $ROOT/runs/events/$M/event_clips_zone/report.md"
  echo "  Simulated cameras: $OUT/ (events.jsonl and snapshots; cam1.mp4 is the annotated video)"
elif [ "$STAGE" = "demo" ]; then
  CAM="${1:-cam1}"
  CLIP="data/clips/synthetic_${CAM}.mp4"
  [ -f "$CLIP" ] || fail "no clip for $CAM ($CLIP). The simulated cameras are cam1, cam2 and cam3."
  step "Watching $CLIP as camera $CAM (press q in the window to close)"
  "$PY" scripts/monitor.py --source "$CLIP" --as-camera "$CAM" || fail "the monitor stopped with an error (see above)"
else
  CAM="${1:-cam1}"
  step "Zone drawing tool for $CAM: click corners in the window, Enter to finish a zone, s to save, q to quit"
  "$PY" scripts/draw_zones.py --camera "$CAM" || fail "the zone tool stopped with an error (see above)"
  "$PY" scripts/check_rules.py || fail "the rules or zones now have a problem (see above)"
fi
sleep 0.3
