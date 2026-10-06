#!/usr/bin/env bash
# The PPE monitor, day to day: one command for each thing you do with it. (The mac_phase*.sh
# scripts built and checked each phase; this one uses them.) Guide: docs/USER_GUIDE.md.
#
#   bash ~/Documents/ppe-monitor/scripts/ppe.sh app                # make the "PPE Monitor" app: double-click to start (--login: at log-in)
#   bash ~/Documents/ppe-monitor/scripts/ppe.sh start              # watch every camera: alerts to your phone + dashboard
#   bash ~/Documents/ppe-monitor/scripts/ppe.sh dashboard          # only the dashboard: look back at events
#   bash ~/Documents/ppe-monitor/scripts/ppe.sh photo PHOTO...      # who wears a helmet in a photo (-> PHOTO_checked.jpg)
#   bash ~/Documents/ppe-monitor/scripts/ppe.sh video VIDEO [CAM]  # check a video file (with camera CAM's zones and rules)
#   bash ~/Documents/ppe-monitor/scripts/ppe.sh camera CAM         # watch one camera in a window, no alerts
#   bash ~/Documents/ppe-monitor/scripts/ppe.sh zones CAM          # draw a camera's restricted zones
#   bash ~/Documents/ppe-monitor/scripts/ppe.sh rules              # check configs/rules.yaml and zones.yaml
#   bash ~/Documents/ppe-monitor/scripts/ppe.sh telegram           # connect alerts to your Telegram (once)
#   bash ~/Documents/ppe-monitor/scripts/ppe.sh check              # is everything working? (~3 min)
#   bash ~/Documents/ppe-monitor/scripts/ppe.sh evaluate           # the five headline numbers on every test clip (~7 min)
#   bash ~/Documents/ppe-monitor/scripts/ppe.sh feedback           # false alarms marked on the dashboard -> training data
#   bash ~/Documents/ppe-monitor/scripts/ppe.sh retrain            # learn from them (needs >= 10), compare, evaluate
#   bash ~/Documents/ppe-monitor/scripts/ppe.sh stop               # stop the database server (start starts it again)
#   bash ~/Documents/ppe-monitor/scripts/ppe.sh model              # which model is in use, and how it was accepted

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
CMD="${1:-help}"
[ $# -gt 0 ] && shift

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; }

case "$CMD" in
  app)       exec bash scripts/make_mac_app.sh "$@" ;;
  start)     exec bash scripts/mac_phase5.sh run "$@" ;;
  dashboard) exec bash scripts/mac_phase5.sh dashboard "$@" ;;
  telegram)  exec bash scripts/mac_phase5.sh telegram "$@" ;;
  check)     exec bash scripts/mac_phase6.sh check "$@" ;;
  evaluate)  exec bash scripts/mac_phase6.sh evaluate "$@" ;;
  feedback)  exec bash scripts/mac_phase6.sh feedback "$@" ;;
  retrain)   exec bash scripts/mac_phase6.sh retrain "$@" ;;
  photo|video|camera|zones|rules|stop|model) ;;
  help|-h|--help) usage; exit 0 ;;
  *) echo "Unknown command: $CMD"; echo; usage; exit 2 ;;
esac

PY=""
for c in "$(command -v conda 2>/dev/null)" /opt/anaconda3/bin/conda "$HOME/anaconda3/bin/conda" /opt/miniconda3/bin/conda \
         "$HOME/miniconda3/bin/conda" "$HOME/miniforge3/bin/conda"; do
  if [ -n "$c" ] && [ -x "$c" ]; then
    ENV_DIR="$("$c" env list 2>/dev/null | awk '$1=="ppe" {print $NF}')"
    if [ -n "$ENV_DIR" ] && [ -x "$ENV_DIR/bin/python" ]; then PY="$ENV_DIR/bin/python"; break; fi
  fi
done
[ -z "$PY" ] && [ -x .venv/bin/python ] && PY="$ROOT/.venv/bin/python"
[ -z "$PY" ] && command -v python3 >/dev/null 2>&1 && python3 -c "import ultralytics" 2>/dev/null && PY="$(command -v python3)"
[ -n "$PY" ] || { echo "No project environment found. Run scripts/mac_setup.sh first."; exit 1; }
export PATH="$(dirname "$PY"):$PATH"
export PYTHONUNBUFFERED=1

case "$CMD" in
  photo)
    [ $# -gt 0 ] || { echo "usage: bash scripts/ppe.sh photo PHOTO [PHOTO ...]   (or a folder of photos)"; exit 2; }
    exec "$PY" scripts/check_photo.py "$@" ;;
  video)
    [ $# -gt 0 ] || { echo "usage: bash scripts/ppe.sh video VIDEO [CAMERA_ID] [--headless]"; exit 2; }
    VIDEO="$1"; shift
    [ -f "$VIDEO" ] || { echo "No such file: $VIDEO"; exit 1; }
    ARGS=()
    if [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; then ARGS=(--as-camera "$1"); shift; fi
    NAME="$(basename "${VIDEO%.*}")"
    mkdir -p runs/videos
    echo "Annotated copy: runs/videos/${NAME}_checked.mp4   Events and their pictures: runs/monitor/"
    exec "$PY" scripts/monitor.py --source "$VIDEO" --save "runs/videos/${NAME}_checked.mp4" \
         ${ARGS[@]+"${ARGS[@]}"} "$@" ;;
  camera)
    [ $# -gt 0 ] || { echo "usage: bash scripts/ppe.sh camera CAMERA_ID   (from configs/cameras.yaml)"; exit 2; }
    CAM="$1"; shift
    exec "$PY" scripts/monitor.py --camera "$CAM" "$@" ;;
  zones)
    [ $# -gt 0 ] || { echo "usage: bash scripts/ppe.sh zones CAMERA_ID [--image photo.jpg | --video clip.mp4 --at 12]"; exit 2; }
    CAM="$1"; shift
    exec "$PY" scripts/draw_zones.py --camera "$CAM" "$@" ;;
  rules)
    exec "$PY" scripts/check_rules.py "$@" ;;
  stop)
    exec "$PY" scripts/db.py stop ;;
  model)
    exec "$PY" - <<'PYEOF'
import json, sys
from pathlib import Path
models = sorted(Path("models").glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
if not models:
    sys.exit("No model in models/.")
m = models[0]
card = m.with_suffix(".json")
info = json.loads(card.read_text()) if card.is_file() else {}
print(f"Model in use: {m.name} (the newest .pt in models/)")
if info:
    print(f"  trained on {info.get('data', '?')}, from {info.get('start_from', '?')}")
    t = info.get("test", {})
    if t.get("map50") is not None:
        print(f"  mAP@50 {t['map50']:.3f} on its {t.get('images', '?')} test photos")
    a = info.get("accepted")
    if a:
        print(f"  accepted {a['when']} in place of {a['replaced']}; report: {a['report']}/report.md")
        if a.get("failed_checks"):
            print(f"  failed check(s), accepted by a person: {'; '.join(a['failed_checks'])}")
            print(f"  reason: {a['override']}")
print("  model card: " + str(m.with_suffix(".md")))
print("Older models (kept, not used): " + (", ".join(p.name for p in models[1:]) or "none"))
PYEOF
    ;;
esac
