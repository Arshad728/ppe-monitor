#!/usr/bin/env bash
# One-command Phase 0 setup and verification for a Mac (Apple Silicon).
#
#   bash ~/Documents/ppe-monitor/scripts/mac_setup.sh
#
# It is safe to run again: every step skips work that is already done.
#   1. checks the Mac (chip, macOS, Rosetta)
#   2. finds or creates a native arm64 Python 3.11 environment (conda env "ppe", or .venv)
#   3. makes sure ffmpeg is installed (Homebrew, or conda-forge if there's no Homebrew)
#   4. installs the pinned Python packages, including PyTorch for the Apple GPU
#   5. downloads MediaMTX, then runs: environment report, Phase 0 acceptance check,
#      the test suite, and the Apple GPU check
# Everything printed is also saved to logs/mac_setup.log.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
mkdir -p logs
LOG="$ROOT/logs/mac_setup.log"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

step() { printf '\n==== %s ====\n' "$*"; }
note() { printf '  %s\n' "$*"; }
fail() {
  printf '\nSETUP STOPPED: %s\n' "$*"
  printf 'Full log: %s\n' "$LOG"
  sleep 0.3
  exit 1
}

echo "PPE monitor - Mac setup, $(date '+%Y-%m-%d %H:%M:%S')"
echo "Project folder: $ROOT"

# ---------------------------------------------------------------- 1. the Mac
step "1/9 This Mac"
if [ "$(uname -s)" = "Darwin" ]; then
  note "$(sw_vers -productName) $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
  note "chip: $(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
  note "memory: $(( $(sysctl -n hw.memsize) / 1073741824 )) GB"
  if [ "$(sysctl -n sysctl.proc_translated 2>/dev/null)" = "1" ]; then
    fail "Terminal is running in Intel mode (Rosetta). In Finder: Applications > Utilities > Terminal > Get Info, untick 'Open using Rosetta', reopen Terminal and run this again."
  fi
  if [ "$(uname -m)" != "arm64" ]; then
    note "WARNING: this is not an Apple Silicon Mac; the Apple GPU check will not pass."
  fi
else
  note "$(uname -sm) - not macOS; continuing for testing purposes"
fi

# Homebrew (optional but preferred for ffmpeg)
BREW="$(command -v brew 2>/dev/null || true)"
[ -z "$BREW" ] && [ -x /opt/homebrew/bin/brew ] && BREW=/opt/homebrew/bin/brew
if [ -n "$BREW" ]; then
  eval "$("$BREW" shellenv)"
  note "Homebrew: $BREW"
else
  note "Homebrew: not installed"
fi

# ------------------------------------------------------- 2. Python environment
step "2/9 Python 3.11 environment"
PY=""
ACTIVATE=""
is_arm_or_linux() { # the given python is native (arm64 on a Mac)
  local arch
  arch="$("$1" -c 'import platform; print(platform.machine())' 2>/dev/null)"
  [ "$(uname -s)" != "Darwin" ] || [ "$arch" = "arm64" ]
}

# Option A: conda (Anaconda / Miniconda / Miniforge), only if it is the arm64 build
CONDA="$(command -v conda 2>/dev/null || true)"
if [ -z "$CONDA" ]; then
  for c in /opt/anaconda3/bin/conda "$HOME/anaconda3/bin/conda" /opt/miniconda3/bin/conda \
           "$HOME/miniconda3/bin/conda" "$HOME/miniforge3/bin/conda" /opt/homebrew/anaconda3/bin/conda \
           /opt/homebrew/Caskroom/miniforge/base/bin/conda /opt/homebrew/Caskroom/miniconda/base/bin/conda; do
    if [ -x "$c" ]; then CONDA="$c"; break; fi
  done
fi
if [ -n "$CONDA" ]; then
  CONDA_BASE="$("$CONDA" info --base 2>/dev/null)"
  if is_arm_or_linux "$CONDA_BASE/bin/python"; then
    ENV_DIR="$("$CONDA" env list 2>/dev/null | awk '$1=="ppe" {print $NF}')"
    if [ -z "$ENV_DIR" ]; then
      note "creating conda env 'ppe' with Python 3.11 (from conda-forge) ..."
      "$CONDA" create -y -q -n ppe -c conda-forge --override-channels python=3.11 >/dev/null \
        || note "conda could not create the env; trying another way"
      ENV_DIR="$("$CONDA" env list 2>/dev/null | awk '$1=="ppe" {print $NF}')"
    fi
    if [ -n "$ENV_DIR" ] && [ -x "$ENV_DIR/bin/python" ]; then
      PY="$ENV_DIR/bin/python"
      ACTIVATE="conda activate ppe"
    fi
  else
    note "found conda at $CONDA, but it is the Intel build (it would run under Rosetta and"
    note "could not use the Apple GPU), so it is not used for this project"
  fi
fi

# Option B: a .venv from Homebrew's Python 3.11, or any python3.11 on PATH
if [ -z "$PY" ]; then
  PY311=""
  if [ -n "$BREW" ]; then
    PY311="$("$BREW" --prefix)/bin/python3.11"
    if [ ! -x "$PY311" ]; then
      note "installing Python 3.11 with Homebrew ..."
      "$BREW" install -q python@3.11 || fail "brew install python@3.11 failed"
    fi
  elif command -v python3.11 >/dev/null 2>&1; then
    PY311="$(command -v python3.11)"
  fi
  [ -n "$PY311" ] && [ -x "$PY311" ] || fail "No suitable Python found. Install Homebrew (https://brew.sh) and run this script again."
  if [ ! -x .venv/bin/python ]; then
    note "creating .venv with $PY311 ..."
    "$PY311" -m venv .venv || fail "could not create .venv"
  fi
  PY="$ROOT/.venv/bin/python"
  ACTIVATE="source $ROOT/.venv/bin/activate"
fi
is_arm_or_linux "$PY" || fail "$PY is an Intel (x86_64) Python; the Apple GPU needs an arm64 Python."
note "using $PY ($("$PY" -c 'import platform; print(platform.python_version(), platform.machine())'))"
# Make the env's own programs (python, and ffmpeg if it comes from conda-forge) come first.
PY_BIN_DIR="$(dirname "$PY")"
export PATH="$PY_BIN_DIR:$PATH"

# ----------------------------------------------------------------- 3. ffmpeg
step "3/9 ffmpeg"
if ! command -v ffmpeg >/dev/null 2>&1; then
  if [ -n "$BREW" ]; then
    note "installing ffmpeg with Homebrew (a few minutes the first time) ..."
    "$BREW" install -q ffmpeg || fail "brew install ffmpeg failed"
  elif [ -n "$ACTIVATE" ] && [ "${ACTIVATE#conda}" != "$ACTIVATE" ]; then
    note "installing ffmpeg into the conda env from conda-forge ..."
    "$CONDA" install -y -q -n ppe -c conda-forge --override-channels ffmpeg >/dev/null || fail "conda install ffmpeg failed"
  else
    fail "ffmpeg is missing and there is no Homebrew or conda to install it with. Install Homebrew (https://brew.sh) and rerun."
  fi
fi
note "$(ffmpeg -version | head -1 | cut -c1-60)  ($(command -v ffmpeg))"

# -------------------------------------------------------- 4. Python packages
step "4/9 Python packages (pinned versions + PyTorch for the Apple GPU)"
"$PY" -m pip install -q --upgrade pip || fail "pip upgrade failed"
"$PY" -m pip install -q -r requirements-dev.txt -r requirements-gpu.txt || fail "pip install failed (see above)"
"$PY" -c 'import cv2, numpy, torch; print("  opencv", cv2.__version__, "| numpy", numpy.__version__, "| torch", torch.__version__)'

# ------------------------------------------------------------- 5. MediaMTX
step "5/9 MediaMTX RTSP server"
"$PY" scripts/get_mediamtx.py || fail "could not get MediaMTX (see above)"

# ------------------------------------------------------------ 6-9. checks
step "6/9 Environment report (saved to docs/ENVIRONMENT.md)"
"$PY" scripts/check_env.py --write; ENV_RC=$?

step "7/9 Phase 0 acceptance check"
"$PY" scripts/verify_phase0.py; P0_RC=$?

step "8/9 Test suite"
"$PY" -m pytest -q; TEST_RC=$?

step "9/9 Apple GPU check"
"$PY" scripts/check_gpu.py; GPU_RC=$?

verdict() { [ "$1" -eq 0 ] && echo "PASS" || echo "FAIL"; }
step "Summary"
note "environment report      $(verdict $ENV_RC)"
note "Phase 0 acceptance      $(verdict $P0_RC)"
note "test suite              $(verdict $TEST_RC)"
note "Apple GPU (PyTorch MPS) $(verdict $GPU_RC)"
echo
note "To work in this project from a new Terminal window:"
note "  cd \"$ROOT\" && $ACTIVATE"
note "Watch the fake cameras: python scripts/fake_cameras.py --with-server"
note "  then, in a second window: python scripts/view_stream.py --camera cam1"
note "Full log: $LOG"
sleep 0.3
[ $ENV_RC -eq 0 ] && [ $P0_RC -eq 0 ] && [ $TEST_RC -eq 0 ] && [ $GPU_RC -eq 0 ]
