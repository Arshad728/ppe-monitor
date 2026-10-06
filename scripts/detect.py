#!/usr/bin/env python3
"""Run the detector on a camera, a video file or an image, and show what it finds.

    python scripts/detect.py --camera cam1                   # a (fake) camera from the config
    python scripts/detect.py --source my_clip.mp4 --save out.mp4
    python scripts/detect.py --source photo.jpg              # shows the image; saves photo_detected.jpg
    python scripts/detect.py --camera cam1 --headless --duration 20   # no window, just speed figures

Press q in the window to stop. Uses the newest model in models/ and its working threshold
unless --weights / --threshold are given.

For live streams a small reader thread keeps only the newest frame, so if the detector is
slower than the camera it skips frames instead of falling further and further behind real
time (the "latest-frame" idea from Chapter 16; Phase 4 builds the full version).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401

import cv2

from ppe_monitor.config import ConfigError, get_camera
from ppe_monitor.device import best_device
from ppe_monitor.ingestion.latest import LatestFrame
from ppe_monitor.ingestion.rtsp import open_capture
from ppe_monitor.vision.detector import MODELS_DIR, load_model

COLOURS = {0: (255, 255, 255), 1: (0, 215, 255), 2: (0, 140, 255)}  # person, helmet, vest (BGR)


def draw(frame, result, names):
    for c, (x1, y1, x2, y2), s in zip(result.boxes.cls.tolist(), result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
        colour = COLOURS.get(int(c), (0, 255, 0))
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, colour, 2)
        label = f"{names[int(c)]} {s:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), colour, -1)
        cv2.putText(frame, label, (p1[0] + 2, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", metavar="ID", help="camera id from configs/cameras.yaml")
    src.add_argument("--source", help="RTSP URL, video file or image")
    parser.add_argument("--weights", help="model file (default: newest in models/)")
    parser.add_argument("--threshold", type=float, help="confidence threshold (default: from the model card)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--headless", action="store_true", help="no window; print speed figures")
    parser.add_argument("--duration", type=float, help="stop after this many seconds")
    parser.add_argument("--save", help="write the annotated video/image to this file")
    args = parser.parse_args()

    weights = Path(args.weights) if args.weights else max(MODELS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, default=None)
    if not weights:
        print("No trained model in models/. Train one first: python scripts/train_detector.py")
        return 1
    threshold = args.threshold
    if threshold is None and weights.with_suffix(".json").is_file():
        threshold = json.loads(weights.with_suffix(".json").read_text())["val"]["threshold"]
    threshold = threshold if threshold is not None else 0.25
    device = best_device() if args.device == "auto" else args.device
    model = load_model(weights)
    names = model.names

    if args.camera:
        try:
            cam = get_camera(args.camera)
        except ConfigError as exc:
            print(exc)
            return 1
        source, transport, title = cam.url, cam.transport, f"{cam.id} - {cam.name}"
    else:
        source, transport, title = args.source, "tcp", Path(args.source).name

    if Path(source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        frame = cv2.imread(source)
        result = model.predict(frame, conf=threshold, imgsz=args.imgsz, device=device, verbose=False)[0]
        out = args.save or str(Path(source).with_name(Path(source).stem + "_detected.jpg"))
        cv2.imwrite(out, draw(frame, result, names))
        counts = {names[i]: result.boxes.cls.tolist().count(i) for i in names}
        print(f"{counts}  ->  {out}")
        if not args.headless:
            cv2.imshow(title, cv2.imread(out))
            cv2.waitKey(0)
        return 0

    cap = open_capture(source, transport=transport)
    if not cap.isOpened():
        print(f"Could not open {source}. For a fake camera, start the network first: "
              "python scripts/fake_cameras.py --with-server")
        return 1
    live = source.lower().startswith(("rtsp://", "rtsps://"))
    reader = LatestFrame(cap) if live else None
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    writer = None
    if not args.headless:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    print(f"Detecting on {source} with {weights.name} (threshold {threshold:.3f}, {device}). Press q to stop.")

    seq, frames, infer_total, skipped, start = 0, 0, 0.0, 0, time.monotonic()
    try:
        while args.duration is None or time.monotonic() - start < args.duration:
            if reader:
                ok, frame, new_seq, _ = reader.get(seq)
                skipped += max(0, new_seq - seq - 1)
                seq = new_seq
            else:
                ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            result = model.predict(frame, conf=threshold, imgsz=args.imgsz, device=device, verbose=False)[0]
            infer_total += time.perf_counter() - t0
            frames += 1
            frame = draw(frame, result, names)
            status = f"{1000 * infer_total / frames:.0f} ms/frame on {device}   skipped {skipped}   [q] quit"
            cv2.rectangle(frame, (0, frame.shape[0] - 30), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            cv2.putText(frame, status, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            if args.save:
                if writer is None:
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame.shape[1::-1])
                writer.write(frame)
            if not args.headless:
                cv2.imshow(title, frame)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if reader:
            reader.ok = False
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    elapsed = time.monotonic() - start
    if frames:
        print(f"\n{frames} frames in {elapsed:.1f} s: detector {1000 * infer_total / frames:.1f} ms/frame on {device} "
              f"(max ~{frames / infer_total:.0f} fps), end to end {frames / elapsed:.1f} fps"
              + (f", {skipped} stale frames skipped to stay live" if live else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
