#!/usr/bin/env python3
"""Run the fake CCTV network: every camera in configs/cameras.yaml that has a sim_source
becomes a live RTSP stream, looping its clip forever.

    python scripts/fake_cameras.py --with-server     # start MediaMTX too (most common)
    python scripts/fake_cameras.py                   # MediaMTX already running (e.g. Docker)
    python scripts/fake_cameras.py --camera cam1     # just one camera

Leave it running and, in a second terminal, watch a stream:
    python scripts/view_stream.py --camera cam1

Stop with Ctrl+C. ffmpeg and MediaMTX output goes to logs/fake_cameras/.
"""

import argparse
import sys
import time

import _bootstrap  # noqa: F401

from ppe_monitor.config import PROJECT_ROOT, ConfigError, load_cameras
from ppe_monitor.sim.fake_cameras import FakeCameraError, FakeCameraNetwork


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "cameras.yaml"))
    parser.add_argument("--with-server", action="store_true", help="also start MediaMTX (from tools/ or PATH)")
    parser.add_argument("--camera", action="append", metavar="ID", help="only these camera ids (repeatable)")
    parser.add_argument("--encoder", default="auto",
                        help="H.264 encoder for ffmpeg: auto (default), libx264, h264_videotoolbox, ...")
    parser.add_argument("--copy", action="store_true",
                        help="send clips without re-encoding (clips must already be H.264 with regular keyframes)")
    parser.add_argument("--duration", type=float, help="stop automatically after this many seconds")
    args = parser.parse_args()

    try:
        cameras = [c for c in load_cameras(args.config) if c.is_simulated]
    except ConfigError as exc:
        print(f"Config problem: {exc}")
        return 1
    if args.camera:
        unknown = set(args.camera) - {c.id for c in cameras}
        if unknown:
            print(f"Not simulated cameras in the config: {', '.join(sorted(unknown))}")
            return 1
        cameras = [c for c in cameras if c.id in args.camera]

    if not cameras:
        print("No simulated camera (sim_source) in the config: nothing to stream. Real cameras need no fake network.")
        return 0
    network = FakeCameraNetwork(cameras, start_server=args.with_server, encoder=args.encoder, copy=args.copy)
    print(f"Starting {len(cameras)} simulated camera(s)" + (" and MediaMTX" if args.with_server else "") + " ...")
    try:
        network.start()
    except FakeCameraError as exc:
        print(f"\nCould not start the fake camera network:\n{exc}")
        return 1

    print("\nLive streams:")
    for cam in cameras:
        print(f"  {cam.id:<8} {cam.url:<32} <- {cam.sim_source.name}")
    print("\nWatch one:  python scripts/view_stream.py --camera", cameras[0].id)
    print("Press Ctrl+C to stop.\n")

    started = time.monotonic()
    try:
        while args.duration is None or time.monotonic() - started < args.duration:
            for message in network.poll():
                print(time.strftime("%H:%M:%S"), message)
            if not network.server_running:
                print("MediaMTX has stopped; shutting down.")
                return 1
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping ...")
    finally:
        network.stop()
    print("Stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
