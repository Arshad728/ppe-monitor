#!/usr/bin/env bash
# Phase 4 on the Mac: several cameras at once, faster model formats, and what they cost in accuracy.
#
#   bash ~/Documents/ppe-monitor/scripts/mac_phase4.sh export      # ~5 min, once: Core ML and ONNX versions of the models
#   bash ~/Documents/ppe-monitor/scripts/mac_phase4.sh check       # ~15 min: tests, benchmark, a camera dropping out
#   bash ~/Documents/ppe-monitor/scripts/mac_phase4.sh run         # watch the 3 simulated cameras at once (q to close)
#   bash ~/Documents/ppe-monitor/scripts/mac_phase4.sh run clips   # ... or 4 of the Phase 2 moving-people clips
#
# `export` installs the export tools (requirements-export.txt: coremltools, ONNX, ONNX Runtime) and
#   exports the detector and the keypoint model four ways: Core ML FP16, Core ML with 8-bit weights,
#   ONNX FP32 and ONNX INT8 (calibrated on the validation photos). Files: models/exported/.
#
# `check` runs, in order:
#   1. the test suite (reader threads, reconnection, the multi-camera engine, with a stand-in model);
#   2. the benchmark: every format (PyTorch FP32 and FP16 on the GPU, Core ML on the Neural Engine,
#      ONNX on the CPU), its accuracy on the test photos, and how many frames a second it gives
#      each of 1, 2, 4 and 8 cameras. Keep the Mac plugged in and leave it alone while this runs;
#   3. four fake RTSP cameras, one of which is unplugged for 15 s and plugged back in: the others
#      must not stall, and it must come back by itself.
# Output is also saved to logs/phase4_<stage>.log. Only one Phase 4 run at a time.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
STAGE="${1:-}"
case "$STAGE" in
  export|check|run) shift ;;
  *) echo "usage: bash scripts/mac_phase4.sh export|check|run [clips]"; exit 2 ;;
esac
mkdir -p logs

LOCK="$ROOT/logs/.phase4.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  OTHER="$(cat "$LOCK/pid" 2>/dev/null)"
  if [ -n "$OTHER" ] && kill -0 "$OTHER" 2>/dev/null; then
    echo "Phase 4 is already running in another window ($(cat "$LOCK/what" 2>/dev/null), process $OTHER)."
    echo "Let it finish, or stop it with Ctrl+C in its window, then run this again."
    exit 1
  fi
  rm -rf "$LOCK" && mkdir "$LOCK" || { echo "Could not create $LOCK"; exit 1; }
fi
echo $$ > "$LOCK/pid"
echo "$STAGE, started $(date '+%H:%M')" > "$LOCK/what"
FAKE_PID=""
cleanup() {
  [ -n "$FAKE_PID" ] && kill "$FAKE_PID" 2>/dev/null && wait "$FAKE_PID" 2>/dev/null
  rm -rf "$LOCK"
}
trap cleanup EXIT

LOG="$ROOT/logs/phase4_$STAGE.log"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

step() { printf '\n==== %s ====\n' "$*"; }
fail() { printf '\nSTOPPED: %s\nFull log: %s\n' "$*" "$LOG"; sleep 0.3; exit 1; }
echo "PPE monitor - Phase 4 ($STAGE), $(date '+%Y-%m-%d %H:%M:%S')"

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

have_exports() { "$PY" -c "import coremltools, onnxruntime" 2>/dev/null && ls -d models/exported/*/coreml >/dev/null 2>&1; }

if [ "$STAGE" = "export" ]; then
  step "1/2 Export tools (coremltools, ONNX, ONNX Runtime)"
  "$PY" -m pip install -q -r requirements-export.txt || fail "installing the export tools failed (see above)"
  "$PY" - <<'PYEOF' || fail "the export tools are not usable (see above)"
import coremltools, numpy, onnx, onnxruntime
print(f"  coremltools {coremltools.__version__}, onnx {onnx.__version__}, onnxruntime {onnxruntime.__version__}, "
      f"numpy {numpy.__version__}")
print(f"  ONNX Runtime can use: {', '.join(onnxruntime.get_available_providers())}")
assert numpy.__version__ == "2.3.5", "numpy must stay 2.3.5 (requirements.txt): pip install numpy==2.3.5"
PYEOF

  step "2/2 Export the detector and the keypoint model"
  "$PY" scripts/export_models.py || fail "some exports failed (see above)"
  du -sh models/exported/*/* 2>/dev/null | sed 's/^/  /'

  step "Done"
  echo "  Next: bash scripts/mac_phase4.sh check"
elif [ "$STAGE" = "check" ]; then
  have_exports || echo "  (No Core ML / ONNX exports yet: only PyTorch will be measured. Run the export stage first.)"
  if [ ! -f datasets/event_clips_sway/clips.json ]; then
    "$PY" scripts/make_event_clips.py --motion sway || fail "making the moving clips failed (see above)"
  fi

  step "1/3 Tests"
  "$PY" -m pytest -q || fail "some tests failed (see above)"

  step "2/3 Benchmark: frames per second and accuracy, every format, 1 / 2 / 4 / 8 cameras (~12 min)"
  echo "  Keep the Mac plugged in, and leave it alone until this is done."
  "$PY" scripts/benchmark.py --streams 1,2,4,8 || fail "the benchmark failed (see above)"

  step "3/3 A camera drops out and comes back"
  "$PY" scripts/check_reconnect.py --cameras 4 --down 15 || fail "the reconnection check failed (see above)"

  step "Done"
  B="$(ls -td runs/benchmark/*/ | head -1)"
  R="$(ls -td runs/reconnect/*/ | head -1)"
  echo "  Benchmark:    $ROOT/${B}report.md"
  echo "  Reconnection: $ROOT/${R}report.md"
else
  if [ "${1:-}" = "clips" ]; then
    if [ ! -f datasets/event_clips_sway/clips.json ]; then
      "$PY" scripts/make_event_clips.py --motion sway || fail "making the moving clips failed (see above)"
    fi
    step "Four moving-people clips played as four cameras (press q in the window, or Ctrl+C here, to stop)"
    "$PY" scripts/run_cameras.py --files datasets/event_clips_sway --count 4 --show || fail "stopped with an error (see above)"
  else
    step "Starting the 3 simulated RTSP cameras (Phase 0)"
    mkdir -p logs/fake_cameras
    "$PY" scripts/fake_cameras.py --with-server > logs/fake_cameras/phase4_run.log 2>&1 &
    FAKE_PID=$!
    sleep 6
    kill -0 "$FAKE_PID" 2>/dev/null || { cat logs/fake_cameras/phase4_run.log; FAKE_PID=""; \
      fail "the fake cameras did not start (is another copy already running on port 8554?)"; }
    step "Watching cam1, cam2 and cam3 at once (press q in the window, or Ctrl+C here, to stop)"
    "$PY" scripts/run_cameras.py --show || fail "stopped with an error (see above)"
  fi
  L="$(ls -td runs/live/*/ | head -1)"
  echo "  Events and snapshots: $ROOT/$L"
fi
sleep 0.3
