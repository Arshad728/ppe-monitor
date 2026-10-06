"""The everyday launchers: ppe.sh, the Mac app builder, and `run --keep-running` restarting what stops."""

import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ["ppe.sh", "make_mac_app.sh", "mac_phase5.sh", "mac_phase6.sh", "mac_caps.sh", "push_to_github.sh"]


@pytest.mark.parametrize("name", SCRIPTS)
def test_the_scripts_parse(name):
    r = subprocess.run(["bash", "-n", str(ROOT / "scripts" / name)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.skipif(platform.system() == "Darwin", reason="on a Mac it would really make the app")
def test_the_app_is_made_only_on_a_mac():
    r = subprocess.run(["bash", str(ROOT / "scripts" / "make_mac_app.sh")], capture_output=True, text=True)
    assert r.returncode == 1 and "run it on the Mac" in r.stdout


def _harness(tmp: Path) -> Path:
    """mac_phase5.sh's watch-and-restart loop, as it is in the script, with stand-ins for the programs:
    a camera service that crashes after LIFE seconds (and stops cleanly on INT/TERM), and an alert worker
    that dies after 3 s."""
    (tmp / "cam.py").write_text(
        "import os, signal, sys, time\n"
        "stop = []\n"
        "signal.signal(signal.SIGINT, lambda *a: stop.append(1)); signal.signal(signal.SIGTERM, lambda *a: stop.append(1))\n"
        "t = time.time()\n"
        "while not stop:\n"
        "    time.sleep(0.1)\n"
        "    if time.time() - t > float(os.environ['LIFE']):\n"
        "        print('cam: crash', flush=True); sys.exit(1)\n"
        "print('cam: stopped cleanly', flush=True)\n")
    (tmp / "svc.py").write_text("import sys, time\ntime.sleep(float(sys.argv[1]))\n")
    s = (ROOT / "scripts" / "mac_phase5.sh").read_text()
    end = s.index('Stopping, because $STOP."', s.index("    trap - INT TERM HUP"))
    loop = s[s.index('    KEEP=""'):end + len('Stopping, because $STOP."')]
    py = sys.executable
    for old, new in (('"$PY" scripts/fake_cameras.py --with-server', f"{py} svc.py 1000"),
                     ('"$PY" scripts/run_cameras.py --db --live', f"{py} cam.py"),
                     ('"$PY" scripts/alert_worker.py', f"{py} svc.py 1000"),
                     ('"$PY" scripts/serve_dashboard.py', f"{py} svc.py 1000"),
                     ("WAIT=10", "WAIT=1")):
        assert old in loop, old
        loop = loop.replace(old, new)
    h = tmp / "harness.sh"
    h.write_text(f"""set -u
mkdir -p logs/fake_cameras
PIDS=""
cleanup() {{ for p in $PIDS; do kill "$p" 2>/dev/null; done; for p in $PIDS; do wait "$p" 2>/dev/null; done; echo "cleanup done"; }}
trap cleanup EXIT
step() {{ echo "== $*"; }}
NSIM=0; NCAM=1; IDS=cam1
{py} svc.py 3 > logs/alert_worker.log 2>&1 & WORKER=$!; PIDS="$PIDS $WORKER"
{py} svc.py 1000 > logs/dashboard.log 2>&1 & DASH=$!; PIDS="$PIDS $DASH"
{loop}
echo "Stopped"
""")
    return h


def _run(tmp: Path, args: list, life: str, stop_after: float | None) -> tuple:
    def fresh_signals():
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        os.setsid()
    p = subprocess.Popen(["bash", str(_harness(tmp)), *args], cwd=tmp, env=dict(os.environ, LIFE=life),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, preexec_fn=fresh_signals)
    if stop_after:
        time.sleep(stop_after)
        os.killpg(p.pid, signal.SIGINT)                  # what Ctrl+C in the Terminal does
    out, _ = p.communicate(timeout=30)
    return p.returncode, out


@pytest.mark.skipif(sys.platform == "win32", reason="bash and process groups")
def test_keep_running_starts_again_what_stops_and_ctrl_c_stops_everything(tmp_path):
    code, out = _run(tmp_path, ["--keep-running"], life="1.5", stop_after=7)
    assert out.count("The camera service stopped (exit code 1). Starting it again") >= 2
    assert "Starting it again in 1 s" in out and "Starting it again in 2 s" in out      # waits longer each time
    assert "The alert worker stopped" in out
    assert out.rstrip().endswith("cleanup done") and "Stopped" in out and code == 0


@pytest.mark.skipif(sys.platform == "win32", reason="bash and process groups")
def test_without_keep_running_a_stopped_camera_service_ends_the_run(tmp_path):
    code, out = _run(tmp_path, [], life="1", stop_after=None)
    assert "cam: crash" in out and "Starting it again" not in out and "Stopped" in out


@pytest.mark.skipif(sys.platform == "win32", reason="bash and process groups")
def test_ctrl_c_stops_the_camera_service_cleanly(tmp_path):
    code, out = _run(tmp_path, ["--keep-running"], life="100", stop_after=3)
    assert "cam: stopped cleanly" in out and "Starting it again" not in out and "cleanup done" in out
    assert "Stopping, because Ctrl+C was pressed" in out                  # says why it stopped


GH_STANDIN = """#!/usr/bin/env bash
# a stand-in for GitHub's CLI: signs in once, "creates" the repository as a local bare one
echo "gh $*" >> "$GH_LOG"
case "$1 $2" in
  "--version "*) echo "gh version 0.0 (stand-in)" ;;
  "auth status") [ -f "$GH_STATE/signed_in" ] ;;
  "auth login") touch "$GH_STATE/signed_in" ;;
  "auth setup-git") ;;
  "api user") case "$*" in *.login*) echo octo ;; *) echo 42 ;; esac ;;
  "repo view") case "$*" in *visibility*) echo PRIVATE ;; *url*) echo https://github.com/octo/ppe-monitor ;;
               *) [ -d "$GH_STATE/remote.git" ] ;; esac ;;
  "repo create") git init -q --bare "$GH_STATE/remote.git" && git remote add origin "$GH_STATE/remote.git" ;;
  *) echo "stand-in gh: unexpected $*" >&2; exit 1 ;;
esac
"""


@pytest.mark.skipif(sys.platform == "win32", reason="bash")
def test_push_to_github_signs_in_creates_a_private_repository_and_pushes_only_what_is_allowed(tmp_path):
    proj, state, bin_ = tmp_path / "proj", tmp_path / "state", tmp_path / "bin"
    for d in (proj / "scripts", proj / "configs", proj / "data" / "clips", state, bin_):
        d.mkdir(parents=True)
    (proj / "scripts" / "push_to_github.sh").write_text((ROOT / "scripts" / "push_to_github.sh").read_text())
    (proj / ".gitignore").write_text("configs/secrets.env\ndata/clips/*\n!data/clips/.gitkeep\n")
    (proj / "README.md").write_text("hello\n")
    (proj / "configs" / "secrets.env").write_text("TELEGRAM_BOT_TOKEN=secret\n")
    (proj / "data" / "clips" / "site.mp4").write_bytes(b"video")
    (proj / "data" / "clips" / ".gitkeep").write_text("")
    (bin_ / "gh").write_text(GH_STANDIN)
    (bin_ / "gh").chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}", GH_LOG=str(state / "log"), GH_STATE=str(state),
               HOME=str(tmp_path), GIT_CONFIG_NOSYSTEM="1")

    def run():
        return subprocess.run(["bash", "scripts/push_to_github.sh"], cwd=proj, env=env, capture_output=True, text=True)

    def git(*a):
        return subprocess.run(["git", "--git-dir", str(state / "remote.git"), *a], capture_output=True, text=True).stdout

    r = run()
    assert r.returncode == 0, r.stdout + r.stderr
    assert "First commit made" in r.stdout and "Made the private repository octo/ppe-monitor" in r.stdout
    log = (state / "log").read_text()
    assert "gh auth login" in log and "--private" in log
    pushed = git("ls-tree", "-r", "--name-only", "main").split()
    assert sorted(pushed) == [".gitignore", "README.md", "data/clips/.gitkeep", "scripts/push_to_github.sh"]
    assert "42+octo@users.noreply.github.com" in git("log", "-1", "--format=%ae", "main")

    r = run()                                                         # again, nothing changed
    assert r.returncode == 0 and "Already signed in" in r.stdout and "Nothing changed" in r.stdout
    (proj / "README.md").write_text("hello again\n")
    r = run()                                                         # a change is committed and pushed
    assert r.returncode == 0 and "Committed the changes" in r.stdout
    assert git("show", "main:README.md") == "hello again\n"

    (proj / ".gitignore").write_text("data/clips/*\n")                # secrets.env no longer ignored: refused
    r = run()
    assert r.returncode == 1 and "configs/secrets.env" in r.stdout and "must not go to GitHub" in r.stdout
    assert "secrets.env" not in git("ls-tree", "-r", "--name-only", "main")
