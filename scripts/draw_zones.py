#!/usr/bin/env python3
"""Draw a camera's restricted zones by clicking on a still frame (Phase 3).

    python scripts/draw_zones.py --camera cam1                   # a frame from the camera
    python scripts/draw_zones.py --camera gate --image gate.jpg   # draw on a saved picture from it
    python scripts/draw_zones.py --camera gate --video gate.mp4 --at 12
    python scripts/draw_zones.py --camera cam1 --show             # no window: save a picture of its zones

In the window:
    click              add a corner (go round the zone in order, along the ground)
    right-click or u   remove the last corner
    Enter              finish the zone; type its id and name in the Terminal
    d                  delete a zone (asks which, in the Terminal)
    s                  save          q or Esc   quit (asks if something isn't saved)

Draw the area people must not STAND in, on the ground: the system tests where each person's feet
are. Corners are saved as fractions of the frame (configs/zones.yaml), with the frame itself in
configs/zones/<camera>.jpg so you can see later what they were drawn on. Then add a rule for the
zone in configs/rules.yaml (who may not enter, when, how urgent) and check both with
python scripts/check_rules.py.

For a camera in configs/cameras.yaml the frame comes from its stream; a simulated camera whose
fake network isn't running falls back to its clip.
"""

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

import cv2

from ppe_monitor.config import PROJECT_ROOT, ConfigError, get_camera
from ppe_monitor.ingestion.rtsp import open_capture
from ppe_monitor.rules.zone_editor import ZoneEditor
from ppe_monitor.rules.zones import ZONES_FILE, CameraZones, load_zones, save_zones

REF_DIR = PROJECT_ROOT / "configs" / "zones"


def frame_from(path_or_url: str, at: float = 0.0, transport: str = "tcp"):
    cap = open_capture(path_or_url, transport=transport, open_timeout_s=8.0)
    if not cap.isOpened():
        return None
    if at > 0 and not path_or_url.lower().startswith(("rtsp://", "rtsps://")):
        cap.set(cv2.CAP_PROP_POS_MSEC, at * 1000)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def get_frame(args):
    """The picture to draw on, and where it came from."""
    if args.image:
        return cv2.imread(args.image), args.image
    if args.video:
        return frame_from(args.video, args.at), f"{args.video} at {args.at:.1f} s"
    try:
        cam = get_camera(args.camera)
    except ConfigError:
        cam = None
    if cam is not None:
        frame = frame_from(cam.url, transport=cam.transport)
        if frame is not None:
            return frame, cam.safe_url
        if cam.sim_source and Path(cam.sim_source).is_file():
            print(f"{cam.id}'s stream isn't running; using a frame of its clip {cam.sim_source}.")
            return frame_from(str(cam.sim_source), args.at), str(cam.sim_source)
        print(f"Could not read a frame from {cam.safe_url}. Save a picture from the camera and use --image.")
        return None, None
    ref = REF_DIR / f"{args.camera}.jpg"
    if ref.is_file():
        return cv2.imread(str(ref)), str(ref)
    print(f"Camera {args.camera} isn't in configs/cameras.yaml. Give a picture from it with --image or --video.")
    return None, None


def ask(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", required=True, help="camera id (configs/cameras.yaml)")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--image", help="draw on this picture instead of the camera's stream")
    src.add_argument("--video", help="draw on a frame of this video")
    parser.add_argument("--at", type=float, default=0.0, help="with --video (or a clip): seconds into it")
    parser.add_argument("--zones", default=str(ZONES_FILE))
    parser.add_argument("--show", action="store_true", help="no window: save a picture of the camera's zones")
    args = parser.parse_args()

    try:
        all_zones = load_zones(args.zones)
    except ConfigError as exc:
        print(f"{args.zones}: {exc}")
        return 1
    existing = all_zones.get(args.camera, CameraZones(args.camera))
    frame, origin = get_frame(args)
    if frame is None:
        return 1
    h, w = frame.shape[:2]
    if existing.size and abs(existing.size[0] / existing.size[1] - w / h) > 0.02:
        print(f"Warning: the zones of {args.camera} were drawn on a {existing.size[0]}x{existing.size[1]} frame and "
              f"this one is {w}x{h}, a different shape. Check that they still sit in the right place.")
    editor = ZoneEditor(args.camera, w, h, list(existing.zones))

    if args.show:
        out = PROJECT_ROOT / "runs" / "zones" / f"{args.camera}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), editor.render(frame, show_help=False))
        print(f"{args.camera}: {len(editor.zones)} zone(s) drawn on {origin} -> {out}")
        for z in editor.zones:
            print(f"  {z.id:16s} {z.name}  ({len(z.polygon)} corners, {100 * z.area:.1f}% of the frame)")
        return 0

    title = f"zones: {args.camera}"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            editor.add(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            editor.undo()

    cv2.setMouseCallback(title, on_mouse)
    print(f"Drawing zones for {args.camera} on {origin} ({w}x{h}). Click the corners in the window; "
          "press Enter to finish a zone.")
    show_help = True
    while True:
        cv2.imshow(title, editor.render(frame, show_help))
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10):                                   # Enter: finish the zone
            if len(editor.corners) < 3:
                print("A zone needs at least 3 corners.")
                continue
            problem = editor.problem()
            if problem:
                print(f"Not a valid zone: it {problem}. Remove corners with u or right-click.")
                continue
            while True:
                zid = ask(f"Zone id for camera {args.camera} (e.g. pit, crane-2; empty = discard): ").strip()
                if not zid:
                    editor.corners.clear()
                    break
                try:
                    zone = editor.finish(zid, ask("Name shown on alerts (e.g. Excavation pit): "))
                    print(f"  added {zone.id}. Press s to save.")
                    break
                except ValueError as exc:
                    print(f"  {exc}")
        elif key in (ord("u"), 8, 127):                        # u / Backspace / Delete: undo a corner
            editor.undo()
        elif key == ord("d"):
            zid = ask(f"Delete which zone? ({', '.join(z.id for z in editor.zones) or 'none'}): ")
            print("  deleted." if editor.delete(zid) else "  no such zone.")
        elif key == ord("h"):
            show_help = not show_help
        elif key == ord("s"):
            save(all_zones, editor, frame, args.zones)
        elif key in (ord("q"), 27) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            if editor.changed and ask("Save your changes? [Y/n] ").strip().lower() not in ("n", "no"):
                save(all_zones, editor, frame, args.zones)
            break
    cv2.destroyAllWindows()
    return 0


def save(all_zones, editor: ZoneEditor, frame, zones_path) -> None:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    ref = REF_DIR / f"{editor.camera}.jpg"
    cv2.imwrite(str(ref), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    all_zones[editor.camera] = editor.result(str(ref.relative_to(PROJECT_ROOT)))
    save_zones(all_zones, zones_path)
    editor.changed = False
    print(f"Saved {len(editor.zones)} zone(s) for {editor.camera} to {zones_path} (frame: {ref}). "
          "Next: a rule for each zone in configs/rules.yaml, then python scripts/check_rules.py")


if __name__ == "__main__":
    sys.exit(main())
