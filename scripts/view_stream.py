#!/usr/bin/env python3
"""Open a camera stream with OpenCV, show it, and check that it plays back correctly.

    python scripts/view_stream.py --camera cam1                   # window; press q to close
    python scripts/view_stream.py --camera cam1 --headless        # no window, 10 s check
    python scripts/view_stream.py --all --headless                # check every camera
    python scripts/view_stream.py --url rtsp://localhost:8554/cam2 --transport udp

A stream passes when it opens, frames arrive at >= 90% of the expected rate, playback
never stalls for more than 1 s, and the frames are real pictures (not flat grey).
Exit code 0 means every checked stream passed.
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

import _bootstrap  # noqa: F401

from ppe_monitor.config import PROJECT_ROOT, Camera, ConfigError, get_camera, load_cameras
from ppe_monitor.ingestion.rtsp import probe_stream
from ppe_monitor.sim.synthetic import clip_fps


def check(cam: Camera, args, show: bool):
    expected = args.expected_fps
    if expected is None and cam.sim_source and cam.sim_source.is_file():
        expected = clip_fps(cam.sim_source)  # a simulated camera streams at its clip's rate
    if args.duration:
        duration = args.duration
    else:
        duration = None if show else 10.0  # a window stays open until you press q
    return probe_stream(
        cam.url,
        display_url=cam.safe_url,
        transport=args.transport or cam.transport,
        duration_s=duration,
        expected_fps=expected,
        show=show,
        window_title=f"{cam.id} - {cam.name}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--camera", metavar="ID", help="camera id from the config")
    target.add_argument("--url", help="any RTSP URL or video file")
    target.add_argument("--all", action="store_true", help="check every enabled camera (headless)")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "cameras.yaml"))
    parser.add_argument("--headless", action="store_true", help="don't open a window")
    parser.add_argument("--duration", type=float,
                        help="seconds to read (default: 10 headless, until q with a window)")
    parser.add_argument("--transport", choices=["tcp", "udp"], help="override the camera's RTSP transport")
    parser.add_argument("--expected-fps", type=float, help="frame rate the stream should deliver")
    args = parser.parse_args()

    try:
        if args.all:
            cameras = load_cameras(args.config)
        elif args.camera:
            cameras = [get_camera(args.camera, args.config)]
        else:
            cameras = [Camera(id="url", name=args.url, url=args.url, transport=args.transport or "tcp")]
    except ConfigError as exc:
        print(f"Config problem: {exc}")
        return 1

    show = not (args.headless or args.all)
    if show:
        print(f"Opening {cameras[0].safe_url} ... (press q in the window to stop)")
        reports = [check(cameras[0], args, show=True)]
    else:
        seconds = args.duration or 10
        print(f"Checking {len(cameras)} stream(s) for {seconds:g} s ...")
        with ThreadPoolExecutor(max_workers=len(cameras)) as pool:
            reports = list(pool.map(lambda c: check(c, args, show=False), cameras))

    for cam, report in zip(cameras, reports):
        print(f"\n== {cam.id}: {cam.name}")
        print("\n".join("   " + line for line in report.summary_lines()))

    failed = [c.id for c, r in zip(cameras, reports) if not r.passed]
    if len(reports) > 1:
        print(f"\n{len(reports) - len(failed)}/{len(reports)} streams passed" +
              (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
