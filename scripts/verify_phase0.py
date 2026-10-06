#!/usr/bin/env python3
"""Phase 0 acceptance check, in one command.

    python scripts/verify_phase0.py

It (1) checks the environment, (2) creates any missing synthetic clips, (3) starts
MediaMTX and the simulated cameras, (4) reads every stream with OpenCV for a little longer
than the clips last (about 23 s), so each stream is also checked at the moment its clip
loops back to the start, and (5) shuts everything down again.

Phase 0 is done when this prints PASS: live RTSP streams from the fake cameras play back
correctly in OpenCV on your machine (book, Chapter 17).
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import _bootstrap  # noqa: F401

from ppe_monitor.config import PROJECT_ROOT, ConfigError, load_cameras
from ppe_monitor.env_check import collect, missing_required, recommended_device
from ppe_monitor.ingestion.rtsp import probe_stream
from ppe_monitor.sim.fake_cameras import FakeCameraError, FakeCameraNetwork
from ppe_monitor.sim.synthetic import clip_duration, clip_fps, make_synthetic_clip

REPORT_PATH = PROJECT_ROOT / "logs" / "phase0_report.txt"
# A local stream should never pause for half a second; this is stricter than view_stream's
# 1 s limit (meant for real cameras on real networks) so that a freeze at the clip's loop point
# - the bug found on the first Mac run - can't pass unnoticed.
MAX_GAP_S = 0.5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "cameras.yaml"))
    parser.add_argument("--duration", type=float,
                        help="seconds to read each stream (default: the longest clip + 3 s, so every "
                             "stream is checked across the moment its clip loops back to the start)")
    parser.add_argument("--no-server", action="store_true", help="use a MediaMTX that is already running")
    args = parser.parse_args()
    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    say(f"Phase 0 verification - {time.strftime('%Y-%m-%d %H:%M:%S')}")

    say("\n[1/4] Environment")
    checks = collect()
    missing = missing_required(checks)
    for c in checks:
        if c.required or c.status == "warn":
            say(f"  {c.status:<7} {c.name}: {c.value}" + (f"  ({c.note})" if c.note and c.status != "ok" else ""))
    say(f"  recommended compute device: {recommended_device(checks)}")
    if missing:
        say("\nFAIL - install the missing items above first (python scripts/check_env.py explains how).")
        return _finish(lines, 1)

    say("\n[2/4] Test clips")
    try:
        cameras = [c for c in load_cameras(args.config) if c.is_simulated]
    except ConfigError as exc:
        say(f"  config problem: {exc}")
        return _finish(lines, 1)
    if not cameras:
        say("  no simulated cameras (sim_source) in the config")
        return _finish(lines, 1)
    for seed, cam in enumerate(cameras):
        if not cam.sim_source.exists() and cam.sim_source.name.startswith("synthetic_"):
            make_synthetic_clip(cam.sim_source, label=cam.name.upper(), seed=seed)
            say(f"  {cam.id}: created {cam.sim_source.name}")
        else:
            say(f"  {cam.id}: {cam.sim_source.name}" + ("" if cam.sim_source.exists() else "  <- MISSING"))

    say("\n[3/4] Fake camera network")
    network = FakeCameraNetwork(cameras, start_server=not args.no_server)
    try:
        network.start()
    except FakeCameraError as exc:
        say(f"  could not start: {exc}")
        say("\nFAIL")
        return _finish(lines, 1)
    say(f"  {len(cameras)} stream(s) live on {cameras[0].url.rsplit('/', 1)[0]}/")

    duration = args.duration or max(10.0, max(clip_duration(c.sim_source) or 0 for c in cameras) + 3)
    say(f"\n[4/4] Reading every stream with OpenCV for {duration:g} s (long enough to cross each clip's loop point)")
    try:
        with ThreadPoolExecutor(max_workers=len(cameras)) as pool:
            reports = list(pool.map(
                lambda c: probe_stream(c.url, display_url=c.safe_url, transport=c.transport,
                                       duration_s=duration, expected_fps=clip_fps(c.sim_source),
                                       max_gap_s=MAX_GAP_S),
                cameras))
    finally:
        network.stop()

    for cam, report in zip(cameras, reports):
        say(f"\n  == {cam.id}: {cam.name}")
        for line in report.summary_lines():
            say("     " + line)

    passed = all(r.passed for r in reports)
    say("\n" + ("PASS - Phase 0 is done: every simulated camera plays back correctly in OpenCV."
                if passed else "FAIL - see the problems listed above and the logs in logs/fake_cameras/."))
    return _finish(lines, 0 if passed else 1)


def _finish(lines: list[str], code: int) -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n(report saved to {REPORT_PATH.relative_to(PROJECT_ROOT)})")
    return code


if __name__ == "__main__":
    sys.exit(main())
