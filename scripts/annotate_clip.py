#!/usr/bin/env python3
"""Mark the real violations in a test clip: the ground truth for the Phase 6 evaluation.

    python scripts/annotate_clip.py data/clips/site1_gate_day_01.mp4
    python scripts/annotate_clip.py data/clips/site1_gate_day_01.mp4 --show   # no window: print what is saved

For each violation (a person without a helmet, or standing in a restricted zone):
  1. go to the frame where it begins and press h (no helmet), v (no vest) or z (zone);
  2. drag a box around the person; move on a second or two and drag it again whenever they have
     moved (the box is followed in a straight line between the ones you draw);
  3. go to where it ends (the person leaves, or puts the helmet on) and press e.

In the window:
    space        play / pause                 , .       one frame back / on
    [ ]          one second back / on         Home or 0 back to the start
    h v z        begin an episode here        e         end it here
    mouse drag   the person's box, now        Tab       select the next episode
    n            sustained yes / no           x         delete the selected episode
    s            save                         q or Esc  quit (asks if something isn't saved)

An episode shorter than the rule's dwell plus half a second (3.5 s for PPE, 2.5 s for a zone) is
saved as "not sustained": an alert on it is neither expected nor wrong. `n` overrides that, for
instance for someone who is mostly hidden.

Mark EVERY violation in the clip: whatever isn't marked counts as compliant, so an alert there
counts as a false alarm. The truth is saved to data/ground_truth/<clip>.yaml (format:
src/ppe_monitor/evaluation/truth.py); the evaluation (scripts/evaluate_system.py) reads it.
"""

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

import cv2

from ppe_monitor.evaluation.annotator import HELP, KEY_KIND, Annotator, new_truth
from ppe_monitor.evaluation.truth import TRUTH_DIR, load_yaml, save_yaml

HEADER = ("# Ground truth for {clip} (Phase 6, book Chapter 23), made with scripts/annotate_clip.py.\n"
          "# Format: src/ppe_monitor/evaluation/truth.py. Times in seconds from the first frame; boxes are\n"
          "# fractions of the frame, [x1, y1, x2, y2], at the times given, followed in a straight line between.")


class Frames:
    """The clip's frames on demand (a few minutes of video don't fit in memory), shown at most
    1280 px wide and 820 px tall (boxes are saved as fractions, so the size doesn't matter)."""

    def __init__(self, path: Path, max_width: int = 1280, max_height: int = 820):
        self.cap = cv2.VideoCapture(str(path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 15.0
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.cap.isOpened() else 0
        self.max_width, self.max_height, self.pos, self.cache = max_width, max_height, -1, {}
        while self.n > 0 and self.get(self.n - 1) is None:     # the count in the header can be too high
            self.n -= 1

    def get(self, k: int):
        if k in self.cache:
            return self.cache[k]
        if k != self.pos + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, f = self.cap.read()
        self.pos = k if ok else -1
        if not ok:
            return None
        scale = min(1.0, self.max_width / f.shape[1], self.max_height / f.shape[0])
        if scale < 1.0:
            f = cv2.resize(f, (int(f.shape[1] * scale), int(f.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        if len(self.cache) > 120:
            self.cache.pop(next(iter(self.cache)))
        self.cache[k] = f
        return f


def ask(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip", help="the video file")
    parser.add_argument("--out", help="where to save (default: data/ground_truth/<clip>.yaml)")
    parser.add_argument("--private", action="store_true",
                        help="the clip shows real people: the evaluation never sends its pictures to Telegram")
    parser.add_argument("--show", action="store_true", help="no window: print the saved truth and stop")
    args = parser.parse_args()

    clip = Path(args.clip)
    out = Path(args.out) if args.out else TRUTH_DIR / f"{clip.stem}.yaml"
    src = Frames(clip)
    if not src.n:
        print(f"Can't read {clip}")
        return 1
    fps = src.fps
    truth = load_yaml(out, clip.parent) if out.is_file() else new_truth(clip, fps, src.n)
    if args.private:
        truth.private = True

    if args.show:
        print(f"{out}: {truth.clip}, {truth.duration:.1f} s, checked {', '.join(truth.checked)}")
        for i, e in enumerate(truth.episodes):
            print(f"  {i + 1}. {e.kind:15s} {e.start:6.2f}-{e.end:6.2f} s  "
                  f"{'sustained' if e.sustained else 'not sustained'}  {len(e.boxes)} box(es)  {e.note}")
        return 0

    a = Annotator(truth, src.n)
    win = f"annotate {clip.name}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    drag = {"from": None, "to": None}
    h, w = src.get(0).shape[:2]

    def on_mouse(event, x, y, flags, param):
        y -= a.top_h
        if event == cv2.EVENT_LBUTTONDOWN:
            drag["from"], drag["to"] = (x, y), (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drag["from"] is not None:
            drag["to"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drag["from"] is not None:
            (x0, y0), drag["from"] = drag["from"], None
            if a.selected is None:
                print("Begin an episode first (h, v or z), then draw the person's box.")
            else:
                a.set_box(x0 / w, y0 / h, x / w, y / h)

    cv2.setMouseCallback(win, on_mouse)
    print("\n".join(HELP))
    playing = False
    while True:
        img = a.render(src.get(a.frame), show_help=w >= 700)
        if drag["from"] is not None:
            (x0, y0), (x1, y1) = drag["from"], drag["to"]
            cv2.rectangle(img, (x0, y0 + a.top_h), (x1, y1 + a.top_h), (255, 255, 255), 1)
        cv2.imshow(win, img)
        key = cv2.waitKey(int(1000 / fps) if playing else 30) & 0xFF
        if playing:
            if a.frame >= src.n - 1:
                playing = False
            else:
                a.step(1)
        if key == 255:
            continue
        if key == ord(" "):
            playing = not playing
        elif key == ord(","):
            a.step(-1)
        elif key == ord("."):
            a.step(1)
        elif key == ord("["):
            a.step(-int(round(fps)))
        elif key == ord("]"):
            a.step(int(round(fps)))
        elif key in (ord("0"), 80):                 # 0, or Home on most keyboards
            a.seek(0)
        elif key in KEY_KIND:
            i = a.begin(KEY_KIND[key])
            print(f"Episode {i + 1}: {KEY_KIND[key]} from {a.t:.2f} s. Drag the person's box, then go to its end and press e.")
        elif key == ord("e"):
            if a.end():
                e = truth.episodes[a.selected if a.selected is not None else -1]
                print(f"  ended at {e.end:.2f} s ({e.end - e.start:.1f} s, "
                      f"{'sustained' if e.sustained else 'not sustained'}, {len(e.boxes)} box(es))")
            else:
                print("Nothing to end here (go past the episode's start first).")
        elif key == ord("n"):
            a.toggle_sustained()
        elif key == ord("x"):
            a.delete()
        elif key == 9:                               # Tab
            a.next()
        elif key == ord("s"):
            out.parent.mkdir(parents=True, exist_ok=True)
            save_yaml(truth, out, HEADER.format(clip=truth.clip))
            a.dirty = False
            print(f"Saved {len(truth.episodes)} episode(s) to {out}")
        elif key in (ord("q"), 27):
            if a.dirty and ask("Not saved. Save before quitting? [Y/n] ").strip().lower() != "n":
                out.parent.mkdir(parents=True, exist_ok=True)
                save_yaml(truth, out, HEADER.format(clip=truth.clip))
                print(f"Saved {len(truth.episodes)} episode(s) to {out}")
            break
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
