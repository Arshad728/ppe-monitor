#!/usr/bin/env bash
# Phase 5 on the Mac: store events, send alerts to your phone, and a dashboard.
#
#   bash ~/Documents/ppe-monitor/scripts/mac_phase5.sh setup      # once, ~5 min: libraries, PostgreSQL, the database
#   bash ~/Documents/ppe-monitor/scripts/mac_phase5.sh telegram   # once, ~2 min: connect the alerts to your Telegram
#   bash ~/Documents/ppe-monitor/scripts/mac_phase5.sh check      # ~5 min: tests, and the whole path end to end
#   bash ~/Documents/ppe-monitor/scripts/mac_phase5.sh run        # the cameras of configs/cameras.yaml -> database -> phone + dashboard
#   bash ~/Documents/ppe-monitor/scripts/mac_phase5.sh run --keep-running   # ... and start again whatever stops
#                                                                 #     (what the PPE Monitor app uses)
#   bash ~/Documents/ppe-monitor/scripts/mac_phase5.sh run clips  # ... or 4 of the Phase 2 moving-people clips
#   bash ~/Documents/ppe-monitor/scripts/mac_phase5.sh dashboard  # only the dashboard, to look back at events
#   bash ~/Documents/ppe-monitor/scripts/mac_phase5.sh telegram --speed   # how fast this network reaches Telegram
#
# `setup` installs SQLAlchemy, psycopg, FastAPI and Uvicorn (requirements-server.txt) and PostgreSQL
#   (from conda-forge, into the project's environment: nothing system-wide), then creates the
#   project's own database server in data/postgres. It listens on this Mac only (127.0.0.1:5433),
#   with a password made up and kept in configs/secrets.env (never in git).
# `telegram` asks for the token @BotFather gave you, finds your chat and sends a test alert.
# `check` runs the tests (also against PostgreSQL) and scripts/check_phase5.py: four clips as
#   cameras, a throw-away database, and a stand-in for Telegram that fails on purpose at first.
# `run` starts the alert worker and the dashboard (http://127.0.0.1:8080, opened in your browser),
#   then the cameras. Ctrl+C stops them all; the database server keeps running
#   (python scripts/db.py stop stops it). While it runs, the Mac is kept from sleeping (on power;
#   closing a laptop's lid still sleeps it). With --keep-running, a camera service, alert worker,
#   dashboard or simulated camera network that stops is started again (the camera service after
#   10 s, then 20, 40 ... up to 5 min if it keeps stopping), so a crash doesn't end the watching.
# Output is also saved to logs/phase5_<stage>.log. Only one Phase 5 run at a time.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
STAGE="${1:-}"
case "$STAGE" in
  setup|telegram|check|run|dashboard) shift ;;
  *) echo "usage: bash scripts/mac_phase5.sh setup|telegram|check|run [clips]|dashboard"; exit 2 ;;
esac
mkdir -p logs

LOCK="$ROOT/logs/.phase5.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  OTHER="$(cat "$LOCK/pid" 2>/dev/null)"
  if [ -n "$OTHER" ] && kill -0 "$OTHER" 2>/dev/null; then
    echo "Phase 5 is already running in another window ($(cat "$LOCK/what" 2>/dev/null), process $OTHER)."
    echo "Let it finish, or stop it with Ctrl+C in its window, then run this again."
    exit 1
  fi
  rm -rf "$LOCK" && mkdir "$LOCK" || { echo "Could not create $LOCK"; exit 1; }
fi
echo $$ > "$LOCK/pid"
echo "$STAGE, started $(date '+%H:%M')" > "$LOCK/what"
PIDS=""
cleanup() {
  for p in $PIDS; do kill "$p" 2>/dev/null; done
  for p in $PIDS; do wait "$p" 2>/dev/null; done
  rm -rf "$LOCK"
}
trap cleanup EXIT

LOG="$ROOT/logs/phase5_$STAGE.log"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1            # (the Telegram token is typed hidden, so it never reaches the log)

step() { printf '\n==== %s ====\n' "$*"; }
fail() { printf '\nSTOPPED: %s\nFull log: %s\n' "$*" "$LOG"; sleep 0.3; exit 1; }
awake() { if command -v caffeinate >/dev/null 2>&1; then caffeinate -i "$@"; else "$@"; fi; }
echo "PPE monitor - Phase 5 ($STAGE), $(date '+%Y-%m-%d %H:%M:%S')"

PY=""
CONDA=""
for c in "$(command -v conda 2>/dev/null)" /opt/anaconda3/bin/conda "$HOME/anaconda3/bin/conda" /opt/miniconda3/bin/conda \
         "$HOME/miniconda3/bin/conda" "$HOME/miniforge3/bin/conda"; do
  if [ -n "$c" ] && [ -x "$c" ]; then
    ENV_DIR="$("$c" env list 2>/dev/null | awk '$1=="ppe" {print $NF}')"
    if [ -n "$ENV_DIR" ] && [ -x "$ENV_DIR/bin/python" ]; then PY="$ENV_DIR/bin/python"; CONDA="$c"; break; fi
  fi
done
[ -z "$PY" ] && [ -x .venv/bin/python ] && PY="$ROOT/.venv/bin/python"
[ -n "$PY" ] || fail "No project environment found. Run scripts/mac_setup.sh first."
PY_BIN_DIR="$(dirname "$PY")"
export PATH="$PY_BIN_DIR:$PATH"
echo "Python: $PY"

have_server_libs() { "$PY" -c "import sqlalchemy, psycopg, fastapi, uvicorn, httpx" 2>/dev/null; }
external_db() { "$PY" - <<'PYEOF' 2>/dev/null
import sys; sys.path.insert(0, "src")
from ppe_monitor.backend.settings import ServerSettings
sys.exit(0 if ServerSettings.load().secrets.get("DATABASE_URL") else 1)
PYEOF
}
need_setup() {
  have_server_libs || fail "Phase 5 isn't set up yet. Run: bash scripts/mac_phase5.sh setup"
  if ! external_db; then
    [ -f data/postgres/PG_VERSION ] || fail "No database yet. Run: bash scripts/mac_phase5.sh setup"
    "$PY" scripts/db.py start || fail "the database server did not start (see logs/postgres.log)"
  fi
}

if [ "$STAGE" = "setup" ]; then
  step "1/3 Libraries: SQLAlchemy, psycopg, FastAPI, Uvicorn, httpx"
  "$PY" -m pip install -q -r requirements-server.txt || fail "installing the libraries failed (see above)"
  "$PY" -c "import sqlalchemy, psycopg, fastapi, uvicorn; print(f'  SQLAlchemy {sqlalchemy.__version__}, psycopg {psycopg.__version__}, FastAPI {fastapi.__version__}, Uvicorn {uvicorn.__version__}')" \
    || fail "the libraries are not usable (see above)"

  step "2/3 PostgreSQL (from conda-forge, into the project environment)"
  if external_db; then
    echo "  DATABASE_URL is set in configs/secrets.env: using that database, no local server needed."
  elif [ -x "$PY_BIN_DIR/pg_ctl" ] || command -v pg_ctl >/dev/null 2>&1; then
    echo "  already installed: $(pg_ctl --version)"
  elif [ -n "$CONDA" ]; then
    echo "  installing (a few minutes the first time) ..."
    "$CONDA" install -y -q -n ppe -c conda-forge --override-channels "postgresql=16" >/dev/null \
      || "$CONDA" install -y -q -n ppe -c conda-forge --override-channels postgresql >/dev/null \
      || fail "conda could not install PostgreSQL. Alternative: put DATABASE_URL=sqlite:///data/ppe.sqlite in configs/secrets.env and run setup again"
    echo "  installed: $("$PY_BIN_DIR/pg_ctl" --version)"
  else
    fail "No conda to install PostgreSQL with. Alternative: put DATABASE_URL=sqlite:///data/ppe.sqlite in configs/secrets.env and run setup again"
  fi

  step "3/3 The database"
  "$PY" scripts/db.py init || fail "creating the database failed (see above)"
  "$PY" scripts/db.py status

  step "Done"
  echo "  Next: connect your Telegram (you need the app on your phone):"
  echo "    1. In Telegram, open @BotFather, send /newbot, pick a name and a username for your bot."
  echo "    2. Open your new bot and press Start."
  echo "    3. bash scripts/mac_phase5.sh telegram"
  echo "  Then: bash scripts/mac_phase5.sh check"

elif [ "$STAGE" = "telegram" ]; then
  have_server_libs || fail "Run setup first: bash scripts/mac_phase5.sh setup"
  step "Connecting the alerts to Telegram"
  "$PY" scripts/telegram_setup.py "$@" || fail "Telegram is not connected yet (see above)"
  step "Done"
  echo "  Next: bash scripts/mac_phase5.sh check   (then run, to get real alerts)"

elif [ "$STAGE" = "check" ]; then
  need_setup
  [ -f datasets/event_clips_sway/clips.json ] || { "$PY" scripts/make_event_clips.py --motion sway || fail "making the moving clips failed"; }
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
  step "2/2 End to end: 4 clips as cameras -> database -> alert worker -> stand-in Telegram -> dashboard API (~2 min)"
  echo "  Keep the Mac plugged in, and leave it alone until this is done."
  awake "$PY" scripts/check_phase5.py || fail "the end-to-end check failed (see above)"
  step "Done"
  R="$(ls -td runs/phase5_check/*/ | head -1)"
  echo "  Report: $ROOT/${R}report.md"

else
  need_setup
  if [ "$STAGE" = "dashboard" ]; then
    step "Starting the dashboard (Ctrl+C to stop it)"
  else
    step "Starting the alert worker and the dashboard"
    "$PY" scripts/alert_worker.py > logs/alert_worker.log 2>&1 &
    WORKER=$!
    PIDS="$PIDS $WORKER"
    # keep the Mac awake (on power) for as long as this runs; -w: stops by itself when this script ends
    command -v caffeinate >/dev/null 2>&1 && { caffeinate -is -w $$ & PIDS="$PIDS $!"; }
  fi
  : > logs/dashboard.log
  "$PY" scripts/serve_dashboard.py > logs/dashboard.log 2>&1 &
  DASH=$!
  PIDS="$PIDS $DASH"
  PORT="$("$PY" -c "import sys; sys.path.insert(0, 'src'); from ppe_monitor.backend.settings import ServerSettings; print(ServerSettings.load().dashboard_port)")"
  URL="http://127.0.0.1:${PORT:-8080}"
  UP=""
  for i in $(seq 1 30); do                 # wait until the dashboard answers (up to 15 s)
    if curl -s -o /dev/null -m 1 "$URL/api/health"; then UP=1; break; fi
    for p in $PIDS; do kill -0 "$p" 2>/dev/null || { cat logs/alert_worker.log logs/dashboard.log; fail "the alert worker or the dashboard stopped (see above)"; }; done
    sleep 0.5
  done
  [ "$STAGE" = "run" ] && sed 's/^/  /' logs/alert_worker.log
  [ -n "$UP" ] || { cat logs/dashboard.log; fail "the dashboard did not answer at $URL (see above)"; }
  echo ""
  echo "  DASHBOARD: $URL   (opening it in your browser; if it doesn't open, copy this address into Safari or Chrome)"
  command -v open >/dev/null 2>&1 && open "$URL"
  echo "  Logs: logs/alert_worker.log, logs/dashboard.log"
  if [ "$STAGE" = "dashboard" ]; then
    wait                                  # until Ctrl+C
  elif [ "${1:-}" = "clips" ]; then
    [ -f datasets/event_clips_sway/clips.json ] || { "$PY" scripts/make_event_clips.py --motion sway || fail "making the moving clips failed"; }
    step "Four moving-people clips as four cameras (Ctrl+C to stop everything)"
    "$PY" scripts/run_cameras.py --files datasets/event_clips_sway --count 4 --db --live || true
  else
    # Which cameras are simulated (sim_source in configs/cameras.yaml)? Only those need the fake network.
    CAMS="$("$PY" -c 'import sys; sys.path.insert(0, "src"); from ppe_monitor.config import load_cameras; cams = load_cameras(); print(sum(c.is_simulated for c in cams), len(cams), ",".join(c.id for c in cams))')" \
      || fail "configs/cameras.yaml has a problem (see above)"
    read -r NSIM NCAM IDS <<< "$CAMS"
    [ "${NCAM:-0}" -gt 0 ] || fail "no enabled camera in configs/cameras.yaml"
    KEEP=""
    for a in "$@"; do [ "$a" = "--keep-running" ] && KEEP=1; done
    FAKE=""
    start_fake() {
      "$PY" scripts/fake_cameras.py --with-server >> logs/fake_cameras/phase5_run.log 2>&1 &
      FAKE=$!
      PIDS="$FAKE $PIDS"
    }
    if [ "$NSIM" -gt 0 ]; then
      [ -f data/clips/synthetic_cam1.mp4 ] || { "$PY" scripts/make_test_clips.py || fail "making the simulated camera clips failed"; }
      step "Starting the $NSIM simulated RTSP camera(s) (sim_source in configs/cameras.yaml)"
      mkdir -p logs/fake_cameras
      : > logs/fake_cameras/phase5_run.log
      start_fake
      sleep 6
      kill -0 "$FAKE" 2>/dev/null || { cat logs/fake_cameras/phase5_run.log; fail "the fake cameras did not start (is another copy already running on port 8554?)"; }
    fi
    step "Watching $NCAM camera(s): $IDS (Ctrl+C to stop everything)"
    [ -n "$KEEP" ] && echo "  Anything that stops is started again (--keep-running). Leave this window open; closing it stops the monitor."
    # The camera service runs in the background; this loop watches it and the other parts. Ctrl+C
    # reaches every part (they stop cleanly), and the trap ends the loop instead of restarting them.
    STOP=""
    trap 'STOP="Ctrl+C was pressed"' INT
    trap 'STOP="its Terminal window was closed"' HUP
    trap 'STOP="it was asked to stop"' TERM
    WAIT=10
    start_cameras() {
      "$PY" scripts/run_cameras.py --db --live &
      CAMPID=$!
      PIDS="$PIDS $CAMPID"
      STARTED=$(date +%s)
    }
    stamp() { date '+%H:%M:%S'; }
    start_cameras
    while [ -z "$STOP" ]; do
      sleep 2
      [ -n "$STOP" ] && break
      if ! kill -0 "$CAMPID" 2>/dev/null; then
        wait "$CAMPID" 2>/dev/null
        code=$?
        [ -n "$STOP" ] && break
        [ -n "$KEEP" ] || break
        [ $(( $(date +%s) - STARTED )) -gt 600 ] && WAIT=10           # it had run well for a while
        echo ""
        echo "$(stamp) The camera service stopped (exit code $code). Starting it again in $WAIT s ... (Ctrl+C to stop everything)"
        for _ in $(seq 1 "$WAIT"); do sleep 1; [ -n "$STOP" ] && break; done
        [ -n "$STOP" ] && break
        WAIT=$(( WAIT * 2 > 300 ? 300 : WAIT * 2 ))
        start_cameras
      fi
      if [ -n "$KEEP" ]; then
        if ! kill -0 "$WORKER" 2>/dev/null; then
          echo "$(stamp) The alert worker stopped (logs/alert_worker.log). Starting it again."
          "$PY" scripts/alert_worker.py >> logs/alert_worker.log 2>&1 &
          WORKER=$!
          PIDS="$PIDS $WORKER"
        fi
        if ! kill -0 "$DASH" 2>/dev/null; then
          echo "$(stamp) The dashboard stopped (logs/dashboard.log). Starting it again."
          "$PY" scripts/serve_dashboard.py >> logs/dashboard.log 2>&1 &
          DASH=$!
          PIDS="$PIDS $DASH"
        fi
        if [ -n "$FAKE" ] && ! kill -0 "$FAKE" 2>/dev/null; then
          echo "$(stamp) The simulated camera network stopped. Starting it again."
          start_fake
        fi
      fi
    done
    trap - INT TERM HUP
    [ -n "$STOP" ] && echo "" && echo "$(stamp) Stopping, because $STOP."
  fi
  step "Stopped"
  echo "  The database server is still running (python scripts/db.py stop to stop it)."
fi
sleep 0.3
