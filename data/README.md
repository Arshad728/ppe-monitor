# Data

Two kinds of data, kept apart on purpose:

| | Training data (`datasets/`) | Test clips (`data/clips/`) |
|---|---|---|
| What | Public labelled photos of workers | Your own videos of real scenes |
| Used for | Teaching the detector (Phase 1) | Measuring it honestly; the fake cameras stream them |
| In git | No (downloaded and rebuilt by `scripts/prepare_data.py`) | No (large, and they show real people) |

## Training data (Phase 1)

`python scripts/prepare_data.py` downloads both datasets into `datasets/raw/` and checks each
against a SHA-256. It merges them into `datasets/ppe3/` with the three classes person, helmet and
vest. `datasets/ppe3/report.md` records every decision it made. Why it works this way:
[decision 0003](../docs/decisions/0003-training-data-and-honest-test.md).

| Dataset | Licence | Attribution |
|---|---|---|
| Construction-PPE | AGPL-3.0 | Construction-PPE dataset by Ultralytics, <https://docs.ultralytics.com/datasets/detect/construction-ppe/> |
| Construction Safety | CC BY 4.0 | "construction safety" dataset, Roboflow Universe, part of the Roboflow 100 benchmark, <https://universe.roboflow.com/object-detection/construction-safety-gsnvb> |

To add another dataset, add an entry to `src/ppe_monitor/data/sources.py`. If it has label names
the config doesn't know, the build stops and lists them. Look at a few images of each, then add
the name to an alias list or to `ignore` in `configs/classes.yaml`.

# Test clips

`data/clips/` holds the video files the fake cameras stream. Nothing in it goes into git
(see `.gitignore`); `clips_catalog.csv` records what each clip is.

## Synthetic clips (created for you)

`python scripts/make_test_clips.py` renders `synthetic_cam1-3.mp4`: cartoon workers, a
restricted zone, and a clock and frame counter. They test the **video plumbing only**. Never
use them to train or measure the detector.

## Real clips (still to collect)

Collect **5–10 short clips (1–3 minutes each)** of real people wearing and not wearing helmets
and hi-vis vests. They become the **held-out test set**. The detector must never train on
them, because they're how it's measured honestly (book, Chapter 18).

Why they matter even though the public datasets have a test split: that split has almost no
distant, small people (3 of 409 people are smaller than 32×32 px), and a CCTV camera mostly sees
workers at that size. Only clips from real camera positions can measure that.

### What to vary

| Variation | Examples |
|---|---|
| PPE | helmet + vest, helmet only, vest only, neither, helmet carried in the hand or set down |
| Camera angle | high and looking down (typical CCTV), eye level, side on |
| Distance | people near the camera and far away (small in the frame) |
| Light | bright daylight, shade, backlit (person against a bright sky), dusk or indoor light |
| Clutter | people partly hidden behind railings, vehicles or each other |
| Look-alikes | orange or yellow clothing that isn't a vest, caps and hoods that aren't helmets |

### Where to get them

- **Record your own** on a phone, with the permission of everyone in the frame. A borrowed
  helmet and vest and one or two friends cover most rows of the table above.
- **Free stock video** sites (Pexels, Pixabay) have construction and warehouse footage. Search
  "construction worker", "hard hat", "warehouse". Check each clip's licence, and note it in the
  catalog. Keep only clips where the camera stays still, as a CCTV camera does.
  - Six Pexels clips are already here (`pexels_*.mp4`, test only), each with its ground truth
    and source link in `ground_truth/`. They are not in git; the links let anyone fetch them
    again. Results: `docs/phase6_results.md`.
- Avoid footage from real sites unless you have written permission. It shows real, identifiable
  people.

### Preparing a clip

iPhones record HEVC `.mov` files. Convert each clip to H.264 `.mp4` with no audio, like a
typical CCTV stream:

```bash
ffmpeg -i IMG_1234.MOV -an -c:v libx264 -pix_fmt yuv420p -vf "scale=-2:720" -r 15 data/clips/site1_gate_day_01.mp4
```

Name clips `place_camera_condition_##.mp4`, add a row to `clips_catalog.csv`, then point a
camera at the clip in `configs/cameras.yaml`:

```yaml
  - id: cam4
    name: Real test clip - gate, daylight
    url: rtsp://localhost:8554/cam4
    sim_source: data/clips/site1_gate_day_01.mp4
```

### Privacy

Clips with people in them are personal data. Keep them on your machine, out of git and
cloud sync folders, and delete any you don't need.
