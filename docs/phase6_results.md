# Phase 6: measured end to end, the feedback loop, and Docker Compose

**Status:** done, 29 Sep 2026: the last phase, and with it the project · **Book:** Chapter 23, with
Chapters 4 and 14 · **Why it is built this way:** [decision 0009](decisions/0009-evaluation-feedback-and-packaging.md)
· **Using the system:** [USER_GUIDE.md](USER_GUIDE.md)

```bash
bash ~/Documents/ppe-monitor/scripts/mac_phase6.sh check        # ~3 min: tests, and a short end-to-end run
bash ~/Documents/ppe-monitor/scripts/mac_phase6.sh evaluate     # ~7 min: every test clip -> the five headline numbers
bash ~/Documents/ppe-monitor/scripts/mac_phase6.sh evaluate --telegram 5   # ... and time 5 real alerts on the phone
bash ~/Documents/ppe-monitor/scripts/mac_phase6.sh evaluate --sets site     # ~2 min: real video only (8 clips)
bash ~/Documents/ppe-monitor/scripts/mac_phase6.sh feedback     # false alarms marked on the dashboard -> training data
bash ~/Documents/ppe-monitor/scripts/mac_phase6.sh retrain      # fine-tune on them, compare, evaluate (needs >= 10)
bash ~/Documents/ppe-monitor/scripts/mac_phase6.sh docker       # needs Docker: `docker compose up`, checked
bash ~/Documents/ppe-monitor/scripts/mac_phase6.sh annotate data/clips/<clip>.mp4   # mark the violations in a clip
```

## The five headline numbers (MacBook Air M4, 26 Sep 2026)

260 test clips (43.5 min of video), each played once as a camera, 8 at a time. They ran through
the whole system: Core ML at 10 frames a second per camera, PostgreSQL, the alert worker and
Telegram. Every alert was matched to the clip's ground truth. Report:
`runs/evaluation/20260926_133601/report.md`.

| Metric (Chapter 4) | Target | Measured | Met? |
|---|---|---|---|
| **mAP@50** (person, helmet, vest) | ≥ 0.85 | **0.883** on the 200 Phase 1 test photos; 0.885 on all 628 test photos | yes |
| **Throughput** | 8 cameras at 10 frames/s | **9.9 frames/s** per camera (slowest 9.1), 8 cameras at once, with the database and alerts on | yes |
| **Alert latency**, violation start to alert | ≤ 5 s, incl. the dwell | **4.0 s** median (90 % within 4.4 s) until the alert leaves the Mac (93 alerts). On the phone: **7.1 s** with the picture (6.9–8.0 s, 4 alerts), **5.0 s** without it (2 alerts, 28 Sep; below) | leaving the Mac: yes. On the phone: no, 5.0 s without the picture |
| **Missed violations** (event recall) | ≥ 90 % caught | **83 %** caught: 93 of 112 (95 % CI 75–89 %) | no |
| **False alarms** (event precision) | ≥ 90 % of alerts real | **89 %** real: 93 of 105 (95 % CI 81–93 %); 16.6 false alerts per camera-hour | just under |

**Without the picture** (`--text-only`, like `photo: false`; 16 still clips), the alert reached
the phone **5.0 s** after the violation began (2 alerts, slowest 5.2 s; 28 Sep): 3.7 s to the
event, then 1.5 s to send. The first run (26 Sep, 5.1 s) opened a fresh connection to Telegram
for each message, so this rerun kept one open, as the live worker does. That was expected to bring
it to about 4.4 s, and it didn't: sending still took 1.5 s. The likely cause is in the worker.
It sends the alerts it has taken off the queue in parallel, but waits for all of them before it
takes the next ones, so an alert that arrives meanwhile waits for the slowest send. Parallel sends
also open new connections beside the one kept warm. In that run even the stand-in's messages took
1.25 s from the event, against 0.27 s when nothing goes to the real Telegram. Not yet confirmed.
The fix to try: take new alerts off the queue while others are still sending. Reports:
`runs/evaluation/20260928_081354/report.md` (and `20260926_134605`).

### By clip set and by kind

| Set | Clips | Real violations | Caught | Alerts | Real | False | Not scored | Recall | Precision | Latency, median / 90 % |
|---|---|---|---|---|---|---|---|---|---|---|
| still (pan and zoom over test photos) | 86 | 28 | 24 | 60 | 24 | 7 | 29 | 86 % | 77 % | 4.0 / 7.0 s * |
| moving (the same photos at walking pace) | 86 | 47 | 41 | 55 | 41 | 5 | 9 | 87 % | 89 % | 4.0 / 4.4 s |
| zone (people walking into a zone) | 86 | 37 | 28 | 52 | 28 | 0 | 24 | 76 % | 100 % | 3.3 / 4.1 s |
| site (your two workshop clips) | 2 | 0 | 0 | 0 | 0 | 0 | 0 | – | – | – |
| **all** | 260 | 112 | 93 | 167 | 93 | 12 | 62 | **83 %** | **89 %** | **4.0 / 4.4 s** |

\* The still set's 90 % includes the 4 alerts sent through the real Telegram with their pictures.

| Kind | Real violations | Caught | Recall | False alarms | Precision |
|---|---|---|---|---|---|
| no helmet | 75 | 65 | 87 % (95 % CI 77–93 %) | 12 | 84 % |
| restricted zone | 37 | 28 | 76 % (95 % CI 60–87 %) | 0 | 100 % |

**Rules:** as `configs/rules.yaml` has them, so the vest rule stays off. On the zone clips, one
zone rule with a 2 s dwell. The rate limit was off, so every alert could be timed. No duplicate
alerts: each violation was alerted once.

### Where the time goes

- **Violation start to event: 3.6 s** median (90 % within 3.7 s). This is detection, tracking,
  the 1 s vote and the 3 s dwell. Zone events: 2.9 s, with a 2 s dwell.
- **Event to Telegram's servers: 0.38 s** median (90 % within 0.62 s). That covers the database,
  the queue (checked every 0.5 s) and the send, to the stand-in on the Mac.
- **Uploading the picture: about 3.2 s more.** Through the real Telegram, event to delivery took
  3.6 s median for 4 alerts sent at once, on this Mac's upload of about 1 Mbit/s.
- **Clip timing:** a clip's frames arrived at most 0.008 s late by its end, so a violation's
  "start on camera" is known to within a hundredth of a second.

### What went wrong, one by one

**The 12 false alarms** (all "no helmet"; the evidence images are in the run's `events/` folder):

| What | Alerts |
|---|---|
| Cyclists in **bicycle helmets** (one photo, 3 people, in both the still and the moving clip). The photo's labels call them helmets; the detector, trained on hard hats, doesn't | 6 |
| A photo **stored sideways** (the worker lies on his side in the frame) | 2 |
| A **small, distant worker** behind a closer one; his helmet isn't found at that size | 2 |
| A **close-up** with the helmet cut off by the frame's edge | 1 |
| A man at a crowd's edge labelled as wearing a helmet. **The label looks wrong**: no helmet is visible | 1 |

**The 19 missed violations:**
- **10 missing helmets.** Six were looked at: close-ups of faces (two of them children, not
  workers), people seen from behind in the dark, and a blurred person. These are unlike what a
  CCTV camera sees, but they are in the test set, so they count. None were small or cut off.
- **9 zone intrusions.** The same 9 as in Phase 3 (docs/phase3_results.md). Four are small,
  distant people the detector never finds. Three are groups boxed as one person. Two have their
  feet cut off by the frame's bottom edge, which the rules deliberately don't judge.

**Your workshop clips:** no alert at all, which is right. Nobody there is without a helmet for
long enough to expect one: the two bare-headed men are in view for under 3 s. The false helmet
alert that the fine-tuned model raised on clip 02 when every frame was analysed
(docs/finetune_results.md) didn't happen at 10 frames a second. That is 27 s of hand-held video:
too little to state a false-alarm rate for the site.

### How to read these numbers

- **The clips are mostly made from public photos, not real video.** People stand still or slide
  across the frame, and a dark bar sweeps past. They test detection, tracking and events end to
  end with known answers. They don't show real walking, turning or lighting changes. Only the
  "site" clips are real footage.
- **False alerts per camera-hour** come from clips where people are in view the whole time. A real
  camera also watches empty scenes, so its hourly rate is usually lower, but that can only be
  measured on real footage.
- **The intervals are wide** because the test set is small: recall could be anywhere from 75 % to
  89 %. Even a system that caught 95 % would need about 200 violations before the low end of
  its interval cleared Chapter 4's 90 %. This set has 112.

## Real video from other sites (28 Sep 2026)

The only real footage was 27 s of hand-held phone video, with no violation long enough to expect
an alert. So 10 free construction clips were taken from Pexels (Pexels licence: free to use, no
attribution needed). Four were dropped because the camera moves, which a CCTV camera doesn't.

- **Six fixed-camera clips were kept**: 1.2 min in all, converted to 1280 × 720 at 15 frames/s,
  like a CCTV substream.
- **Each was annotated by hand.** Every person near enough to judge was checked on crops of the
  original high-resolution file (`data/ground_truth/pexels_*.yaml`, each with its source link).
  - 6 real "no helmet" violations, in 3 clips.
  - In the other 3 clips everyone wears a helmet.
- **Test footage only:** they are listed as `test` in `data/clips/clips_catalog.csv` and never
  trained on. Unlike the workshop clips, they are public and may be shown.

**Run on the Mac** with your two workshop clips: 8 clips as 8 cameras at once, the whole system,
Core ML. Command: `mac_phase6.sh evaluate --sets site`, 1.7 min. Report:
`runs/evaluation/20260928_203352/report.md`.

| Clip (Pexels id) | Length | What it shows | Real violations | Caught | False alarms |
|---|---|---|---|---|---|
| brick_loading (32165834) | 24.6 s | two men loading bricks, no helmets; low camera | 2 | **2** | **1**: the red tractor at the left edge, taken for a person |
| bricklaying (11355903) | 16.1 s | two bricklayers in baseball caps | 2 | **1** | 0 |
| street_site (30157703) | 7.2 s | street works: a man in a checked shirt; a small man in a pink cap, bent over; a boy behind the barrier (not scored) | 2 | **1** | 0 |
| rebar_height (12098511) | 14.7 s | six people fixing rebar at height, seen from below, all in helmets | 0 | – | 0 |
| overhead_site (7448386) | 5.8 s | about 20 people seen from high above, like CCTV, all in helmets | 0 | – | 0 |
| stonewall_closeup (19832500) | 4.9 s | four people close up, all in helmets | 0 | – | 0 |
| your workshop (2 clips) | 27 s | hand-held, indoors | 0 | – | 0 |
| **all 8** | **1.7 min** | | **6** | **4 (67 %)** | **1** |

| Metric | Measured on real video | On the 260 clips (above) |
|---|---|---|
| Throughput | 10.0 frames/s per camera, 8 cameras | 9.9 |
| Missed violations (recall) | **4 of 6 caught, 67 %** (95 % CI 30–90 %) | 83 % |
| False alarms (precision) | **4 of 5 alerts real, 80 %** (95 % CI 38–96 %) | 89 % |
| False alerts per camera-hour | 36 (95 % CI 1–200) | 16.6 |
| Latency, violation start to alert leaving the Mac | 4.0 s median, 5.5 s slowest | 4.0 s |

**What the three mistakes show.** Each is a kind of scene the photo clips had little or none of.

- **A cap is taken for a helmet.** The two bricklayers both wear baseball caps. The man in the
  beige shirt was caught; the man in the blue shirt never was: the model saw his cap as a
  helmet. The training photos have few people in caps or hats labelled "no helmet". The fix is
  data: such photos in training.
  - This is the miss a site is most likely to meet, since caps are what people wear instead.
  - **Done on 29 Sep** ([caps_results.md](caps_results.md)): with 1,043 photos of people in hats in
    training, the model now in use alerts this man, and judges 2.5 % of people in hats on held-out
    photos "helmet worn" (27.4 % before). Real video: 5 of 6 caught.
- **A small worker is missed.** The man in the pink cap is about 90 pixels tall and bent over.
  Phase 3's missed zone intrusions were small people too.
  - A camera placed so that people are larger in the frame helps most.
  - A larger input to the model helps too, at a cost in speed.
- **A machine is taken for a person.** At 1.4 s the red tractor at the brick_loading clip's left
  edge was detected as a person, followed for 3 s, and raised a "no helmet" alert.
  - The same clips in the cloud (PyTorch, CPU) raised no alert on it. Core ML's slightly
    different numbers likely put it just over the threshold.
  - The feedback loop will meet false alarms like this one. So its instructions now say: when
    the "person" isn't a person, delete that label rather than draw a helmet.

**How to read these numbers.** Six violations are far too few to judge the system by: recall
could be anywhere from 30 % to 90 %. So could the false-alarm rate, which rests on one alarm in 1.7
minutes. What the clips do show is *which kinds* of mistake a real site brings, and they are not
the ones the photo clips found.
- There, the misses were close-ups and people seen from behind in the dark, and half the false
  alarms were bicycle helmets.
- Here, they are a cap, a small person and a machine.

Footage from your own fixed camera is still what can tell how the system does on your site.

## The feedback loop (false alarms -> the next model)

Built and tested; **not run for real yet.** Every false alarm marked on the dashboard so far came
from test clips, and test footage is never trained on.

- **`feedback export`** takes every event marked "False alarm" whose clean frame is kept. It keeps
  only those from real cameras, or from clips listed with split `train` in
  `data/clips/clips_catalog.csv`. Then it labels each frame:
  - what the model in use finds, at the rules' thresholds;
  - plus what the verdict says it missed. For a false "no helmet", the helmet on the person's
    head is looked for again down to confidence 0.05, and must be at least 15 % of the person's
    width.
  - Found: status **ready**. Not found: **needs a box** drawn by hand, or, if the "person" is
    not a person at all (the tractor above), that label deleted. Zone alarms: **for a person to
    look at**.
  - `datasets/feedback/review.html` shows every frame with its boxes; `skip.txt` leaves any out.
- **`retrain`** builds `datasets/ppe5`: ppe4 plus each ready image 5 times, in training only. It
  fine-tunes the model in use (at most 10 epochs, early stop after 4). Then it evaluates the
  candidate on every test clip, end to end.
  - It is accepted only if all four hold: mAP@50 no more than 0.01 lower on the Phase 1 test
    photos, and on all the extra test photos; no more false alarms; at most 2 more misses.
- **A dry run in the cloud**, on the evaluation's own false alarms, allowed with
  `--include-test` for trying the tools only:
  - 9 false alarms: 6 **ready**, with the missed helmet found again at confidence 0.06–0.19, and
    3 **need a box**.
  - One wrong lesson was caught and fixed: the small helmet of a man standing behind was first
    taken for the person's own. Since then a helmet must be at least 15 % of the person's width.
  - Ultralytics loaded the resulting dataset: 2,170 training images plus 6 × 5 feedback copies.
- **What it needs:** footage from your site that isn't test footage, played as cameras
  (`mac_phase5.sh run` on clips listed as `train`) or from real cameras. Then false alarms
  marked on the dashboard, at least 10 of them. Real clip 02's white helmet in haze is exactly
  the kind of lesson it is for.

## Hardening

- **Camera down:** a camera not sending for more than 60 s raises one alert ("📷 Camera down: Main
  gate, not sending since 14:03 (could not open the stream)"), and another when it is back.
  Cameras that drop together share one message.
- **Camera service silent:** no heartbeat for more than 60 s raises one alert ("⚠️ The camera
  service has stopped reporting: no camera is being watched"), since a monitor that has quietly
  stopped is worse than none. A deliberate stop raises nothing.
- Both limits are in `configs/server.yaml` (`alerts.health`). Unsent messages are retried.
- **Secrets can come from files** (`PPE_DB_PASSWORD_FILE`, `DASHBOARD_PASSWORD_FILE`), the way
  Docker hands them over. Docker's database address comes from `PPE_DB_HOST` / `PPE_DB_PORT`.
- Every service in Docker has a **healthcheck**, restarts if it stops, and keeps at most 30 MB of
  logs.

## Docker Compose

```bash
docker compose up -d --build      # first time: builds the image (2.5 min on the cloud machine)
docker compose logs secrets       # the dashboard's password
open http://127.0.0.1:8080        # user: safety
```

The services:
- `secrets` makes the passwords once, into a volume that only the services mount.
- `db` is PostgreSQL 16, not reachable from outside Docker; `db-init` creates the tables.
- `cameras`, `alerts` and `dashboard` each have a healthcheck.
- `demo-rtsp` (MediaMTX) and `demo-cameras` stream the three simulated cameras of
  `configs/cameras.yaml`.

`configs/secrets.env` (Telegram token and chat, `DASHBOARD_PASSWORD`, `TZ`) is passed to every
service if it exists. `docker-compose.gpu.yml` builds CUDA PyTorch for an NVIDIA GPU: written
but untested.

**Checked from a clean copy in the cloud** (`scripts/check_compose.py`, 26 Sep 2026; Linux, 2
cores, Docker 29.4). The copy had only the code, the settings and `models/`. No Python,
PostgreSQL or clips were installed; the check makes the clips itself.

| Check | Result | Detail |
|---|---|---|
| every service up; cameras, alerts and dashboard healthy | PASS | 24 s after `up` (image already built) |
| the dashboard asks for a password | PASS | HTTP 401 without it |
| the dashboard answers with it; the database is reachable | PASS | PostgreSQL |
| the three simulated cameras are live over RTSP | PASS | 3.4 frames/s each: 2 CPU cores, shared with three video encoders |
| violations become events in PostgreSQL | PASS | 2 no helmet, 1 restricted zone |
| their alerts are delivered with the picture | PASS | 3 of 3, 0.4–0.8 s after the event, to a stand-in Telegram |
| the dashboard serves the evidence image | PASS | 127 KB JPEG |

**Problems found on the way, and fixed:**
- **The fake cameras' ffmpeg crashed inside the image.** The static ffmpeg build shipped with the
  Python package crashes when it has to look up a host name. The publishers now send to
  127.0.0.1 instead of "localhost".
- **Ultralytics installed a library by itself at first start.** The tracker's `lap` was missing,
  so Ultralytics fetched it from the internet at run time. It is now pinned
  (`requirements-ml.txt`).
- **The check read the stand-in's log too soon**, and once counted 1 picture of 3. It now waits
  for them.

**On this Mac:** Docker isn't installed, and isn't needed. On a Mac, Docker runs Linux in a
virtual machine that can use neither the Apple GPU nor Core ML. The camera service would run on
the CPU there, where Phase 4 measured ONNX on the CPU at 2.5 frames/s per camera with 8 cameras
(10 with Core ML). `mac_phase6.sh docker` runs the same check if Docker Desktop or OrbStack is
installed.

## Checked in the cloud (26 Sep 2026)

- **Tests:** 233 pass, 1 skipped. 28 of them are new: ground truth, matching, the headline
  numbers and their intervals, the annotation tool, the feedback labels and dataset,
  camera-down and silent-service alerts, secrets from files, and a file played once as a camera.
- **The full evaluation** also ran on the cloud's 2 CPU cores, 2 cameras at a time. The cameras
  ran at 4.4 frames/s, not 10, so these numbers are not the reference. It caught 82 %, 91 % of
  alerts were real, and alerts took 4.3 s median. A Mac and a CPU machine agreeing this closely
  says the evaluation measures the system, not the machine.

## On the Mac (26 Sep 2026)

- **`check`:** 242 tests pass, including the database tests on PostgreSQL. The short run
  (10 clips) had 8 cameras at 10.0 frames/s, 3 of 3 violations caught, 0 false alarms, and alerts
  4.0 s after the violation began.
- **`evaluate --telegram 5`:** the numbers above, in 6.5 min.
  - 167 events, all stored, all alerted, none lost.
  - The loop used 96 ms of each 100 ms tick with 8 cameras: this Mac is at its limit there.
