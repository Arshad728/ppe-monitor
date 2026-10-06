# Worker Safety (PPE) Monitoring System

Watches live CCTV streams, notices when a worker isn't wearing a helmet or high-visibility vest
or has walked into a restricted zone, and alerts a safety officer within seconds with a
snapshot as evidence. The design and the reasoning behind it are in the project book,
[docs/book/Worker_Safety_PPE_Monitoring_System_Book.pdf](docs/book/Worker_Safety_PPE_Monitoring_System_Book.pdf).
Chapter numbers below refer to it. Each built phase has its own "As Built" notes in the book,
and the book is updated as the project moves on ([docs/book/README.md](docs/book/README.md)).

**The project is complete (29 Sep 2026): all six phases are built, checked and reviewed.**

Licence: the code is AGPL-3.0; the trained model is for non-commercial use only
([why](#licence-and-what-this-repository-leaves-out)).

## How to use it

**Guide: [docs/USER_GUIDE.md](docs/USER_GUIDE.md)** — everyday use, your own cameras, zones and
alerts, keeping it accurate, and the model on its own. One script covers the everyday tasks:

```bash
bash ~/Documents/ppe-monitor/scripts/ppe.sh app                # once: a "PPE Monitor" app to double-click (--login: start at log-in)
bash ~/Documents/ppe-monitor/scripts/ppe.sh start              # watch every camera: alerts to your phone + dashboard
bash ~/Documents/ppe-monitor/scripts/ppe.sh photo PHOTO.jpg    # who wears a helmet in a photo
bash ~/Documents/ppe-monitor/scripts/ppe.sh video CLIP.mp4     # check a video file
bash ~/Documents/ppe-monitor/scripts/ppe.sh help               # every command
```

## Build status

| Phase | What it delivers | Status |
|---|---|---|
| **0 — Setup and foundations** (Ch. 17) | Repository, pinned environment, fake CCTV network, stream checks, Apple GPU check | **Done — verified on the Mac (M4), 21 Sep 2026** |
| **1 — Data and baseline detector** (Ch. 18) | YOLO fine-tuned on person / helmet / vest, measured per class, failure cases listed | **Done: reviewed 23 Sep 2026** (mAP@50 0.869 on the public test split). **Replaced 25 Sep** by a model fine-tuned with more public data (0.883; [results](docs/finetune_results.md)), and **29 Sep** by one taught that caps and hats aren't helmets ([results](docs/caps_results.md)) |
| **2 — Tracking and PPE logic** (Ch. 19) | ByteTrack IDs, PPE-to-person matching, dwell time and cooldown; walking, bending and crouching workers | **Done: reviewed 24 Sep 2026** |
| **3 — Restricted zones and rules engine** (Ch. 20) | Zone drawing tool, point-in-polygon on each person's feet, rules as configuration (zone, severity, dwell, cooldown, active hours), one event shape | **Done: reviewed** |
| **4 — Multi-stream ingestion and optimisation** (Ch. 21) | Reader thread per camera with reconnection, one clock and batching across cameras, Core ML / ONNX exports, speed-against-accuracy benchmark | **Done, 25 Sep 2026:** 8 cameras at 10 fps on the Mac (keypoints on 1 frame in 2) |
| **5 — Alerts, storage and dashboard** (Ch. 22) | PostgreSQL, alert queue and worker, Telegram/email with the snapshot, FastAPI dashboard with the false-alarm button | **Done, 26 Sep 2026** (alert with the picture on the phone 3–4 s after the event) |
| **6 — Evaluation, hardening and packaging** (Ch. 23) | Ground truth and end-to-end metrics, the false-alarm → retraining loop, camera-down alerts, Docker Compose | **Done, 29 Sep 2026:** checked on the Mac and in Docker (26 Sep), on real video from other sites (28 Sep), and taught caps and hats (29 Sep) ([results](docs/phase6_results.md)) |

## The system at a glance

![How the system is built](docs/book/figures/diagram_as_built.png)

- **Camera service** (`scripts/run_cameras.py`): one reader thread per camera keeps only the
  newest frame. A detector call per tick covers every camera: YOLO26n, fine-tuned on public PPE
  data and on photos of people in caps and hats, as Core ML on the Mac, ONNX or PyTorch elsewhere. Then body keypoints, a tracker, the PPE
  and zone rules, and an event engine that raises one event per violation, after a 3 s dwell.
- **Storage:** events and their alert queue in PostgreSQL, and the evidence images as files. A
  writer thread stores them, so the video never waits.
- **Alert worker:** Telegram (or email) with the picture. It has a rate limit, retries, and alerts
  of its own when a camera or the camera service goes quiet.
- **Dashboard** (FastAPI and one page, behind a password): live cameras, counts, the event feed,
  and **Confirm / False alarm** on each event. False alarms become training data for the next
  model.

![The simulated main-gate camera](docs/demo_simulated_camera.gif)

*The simulated main-gate camera (a cartoon, from Phase 0), as the system sees it. A worker walks
into the hatched keep-out area, and a box turns amber, then red. Another takes his helmet off and
hangs it on the rail. The man in blue shows a false alarm: the model doesn't know the cartoon's
white helmet. That is one reason cartoons are never used to measure the system.*

## Measured: the five headline numbers

Every test clip played once as a camera, 8 at a time, through the whole system on a MacBook Air
M4: 266 clips, 44.7 minutes of video, 8 of them real video. Each alert was matched to the clip's
ground truth. Model in use: `ppe4caps_yolo26n_caps2` (29 Sep 2026, `runs/evaluation/20260929_211410_caps_candidate`).
Details, and every false alarm and miss: [docs/phase6_results.md](docs/phase6_results.md) (the first
measurement, 26 Sep) and [docs/caps_results.md](docs/caps_results.md).

| Metric (book, Chapter 4) | Target | Measured | Met? |
|---|---|---|---|
| **mAP@50** (person, helmet, vest) | ≥ 0.85 | **0.877** on held-out test photos (200; 0.886 on all 628) | yes |
| **Throughput** | 8 cameras at 10 frames/s | **9.9 frames/s** per camera with 8 cameras, the database and alerts on | yes |
| **Alert latency** (violation starts → alert) | ≤ 5 s, incl. the 3 s dwell | **4.0 s** median until the alert leaves the Mac (90 % within 4.5 s). On the phone: **7.1 s** with the picture (~1 Mbit/s upload), **5.0 s** without | leaving the Mac: yes; on the phone: not yet |
| **Missed violations** (event recall) | ≥ 90 % caught | **83 %** caught (98 / 118; 95 % CI 75–89 %) | no |
| **False alarms** (event precision) | ≥ 90 % real | **89 %** real (100 / 112; 95 % CI 82–94 %); 16.1 false alerts per camera-hour | just under |

- **Latency without the picture** (text and a dashboard link, `photo: false`): 5.0 s on the
  phone, measured again with the connection to Telegram kept open (28 Sep). Sending still takes
  1.5 s after the event. The likely cause: the alert worker waits for all the alerts it is sending
  before it takes new ones off the queue.
- **Real video from other sites** (28 Sep; [details](docs/phase6_results.md#real-video-from-other-sites-28-sep-2026)):
  6 fixed-camera construction clips from Pexels, annotated by hand, plus the 2 workshop clips.
  - 8 cameras at 10.0 frames/s.
  - First model: **4 of 6** real violations caught (67 %; 95 % CI 30–90 %), **4 of 5** alerts
    real (80 %): the false one was a tractor taken for a person. The misses: a man whose baseball
    cap was taken for a helmet, and a small worker (about 90 pixels tall).
  - Since the caps fine-tuning (29 Sep): **5 of 6** caught. The man in the cap is alerted; the
    small worker and the tractor remain.
  - Six violations are too few to judge by. The clips show what kinds of mistake a real site
    brings.

## Known limitations

- **Most test clips are made from photos, not real video.** 258 of the 260 clips pan or slide
  public test photos with known labels. The real video is short:
  - 27 seconds of hand-held phone video from your workshop, where no alert was raised and nobody
    needed one;
  - 1.2 minutes from six other sites (Pexels), with 6 real violations.

  Precision and recall on your site are not known yet. Measuring them needs footage from your
  own fixed camera, annotated with `scripts/annotate_clip.py`.
- **Caps and hats passed for helmets** until 29 Sep: the model then in use judged 27 % of people
  in hats "helmet worn". A fine-tuning with 1,043 photos of people in hats brought it to 2.5 %
  ([results](docs/caps_results.md)). The price, accepted by the user: on the Phase 1 test photos,
  5 small or half-hidden helmets of 350 are no longer found at the alert threshold, and 2 others
  now are.
- **Small people are missed.** A worker about 90 pixels tall was missed on real video, as were
  small, distant people in Phase 3. Place cameras so that people appear larger.
- **Machines can be taken for people.** A tractor at a frame's edge raised a "no helmet" alert.
- **Recall is below the 90 % target** (83 %).
  - Most missed helmets are close-ups of faces and people seen from behind in the dark.
  - Missed zone intrusions are small, distant people, groups boxed as one, and feet cut off by
    the frame's edge.
  - A CCTV camera placed with the whole zone in view avoids the last two.
- **Half the false alarms are bicycle helmets** in one public photo. The detector learnt hard
  hats; bicycle helmets and hoods are unfamiliar to it.
- **With a picture, alerts take about 7 s** on this Mac's connection (upload about 1 Mbit/s).
  `photo: false` in `configs/server.yaml` sends text with a link to the dashboard instead.
- **The retraining loop hasn't run for real.** It is built and tested, but every false alarm so
  far came from test clips, which are never trained on.
- **Vests:** the vest rule is off (hi-vis isn't required at the user's site). The fine-tuned
  models are weaker on vests than the first one ([details](docs/finetune_results.md)); the caps
  model judges wearers "no vest" a little more often (7.0 % -> 9.5 % on the Phase 1 test photos).
- **No GPU in Docker on a Mac.** Docker can't use the Apple GPU, so Compose is for Linux servers.
  The NVIDIA override (`docker-compose.gpu.yml`) is written but untested.
- **Licences:** Ultralytics is AGPL-3.0. Two of the training datasets are non-commercial or have
  no stated licence ([decision 0007](docs/decisions/0007-fine-tuning-with-extra-public-data.md)).
  This is a learning and portfolio project as it stands.

## Run it with Docker Compose

On any machine with Docker (Linux, or Docker Desktop / OrbStack on a Mac), from this folder with
its `models/`:

```bash
docker compose up -d --build      # first time ~3-10 min: builds the image
docker compose logs secrets       # the dashboard's password (user: safety)
open http://127.0.0.1:8080        # the dashboard: the three simulated cameras, live
docker compose ps                 # cameras, alerts and dashboard should say "healthy"
docker compose down               # stop; the database and the evidence images are kept
```

- **The services:**
  - PostgreSQL;
  - the camera service, the alert worker and the dashboard;
  - an RTSP server with the simulated cameras;
  - two one-off jobs: make the passwords, create the tables.
- **Real cameras:** remove `sim_source` from their entries in `configs/cameras.yaml`.
- **Telegram:** put `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `configs/secrets.env`, which
  every service reads. The alert worker then sends real alerts.
- **Checked from a clean copy** with `python scripts/check_compose.py`: every service healthy,
  the cameras live, violations stored and alerted, the password demanded
  ([results](docs/phase6_results.md#docker-compose)).

## Phase 6: measured end to end, learning from false alarms, packaged

```bash
bash ~/Documents/ppe-monitor/scripts/mac_phase6.sh check        # ~3 min: tests, and a short end-to-end run
bash ~/Documents/ppe-monitor/scripts/mac_phase6.sh evaluate     # ~7 min: every test clip -> the five numbers
bash ~/Documents/ppe-monitor/scripts/mac_phase6.sh evaluate --sets site   # ~2 min: the real video only
bash ~/Documents/ppe-monitor/scripts/mac_phase6.sh feedback     # dashboard false alarms -> training data
bash ~/Documents/ppe-monitor/scripts/mac_phase6.sh retrain      # fine-tune on them, compare, evaluate (needs >= 10)
bash ~/Documents/ppe-monitor/scripts/mac_phase6.sh annotate data/clips/<clip>.mp4   # mark a clip's violations
```

**How it works** ([decision 0009](docs/decisions/0009-evaluation-feedback-and-packaging.md)):
- **Ground truth:** each clip's violations, with when they begin and end and where the person is
  (`data/ground_truth/*.yaml`, or the labels of the generated clips). An alert that fits one in
  time and place is real. An alert on a person known to be compliant is a false alarm. A
  violation with no alert is a miss.
- **End to end:** the clips play as cameras through the real services. Latency runs from the
  moment a violation begins on camera to the moment the alert is delivered. The rates come with
  95 % confidence intervals.
- **Feedback:** false alarms marked on the dashboard become labelled training images. The helmet
  the model missed is found again at a lower confidence and becomes the lesson. They are used
  only from footage that isn't test footage. A retrained model is accepted only if it isn't worse
  on the test photos and raises no more false alarms end to end.
- **Hardening:** a camera down for a minute, or a silent camera service, raises its own alert.

### Caps and hats are not helmets (29 Sep 2026)

```bash
bash ~/Documents/ppe-monitor/scripts/mac_caps.sh run       # ~2-3 h: Open Images photos, fine-tune, compare
bash ~/Documents/ppe-monitor/scripts/mac_caps.sh accept    # put it into use, if it passed every check
```

A bricklayer in a baseball cap on real video was never alerted, and the model then in use judged
27 % of people in hats "helmet worn". 1,043 Open Images photos of people in hats (hats unlabelled,
so they are background to the detector; every hat the model half-believed was a helmet was
looked at by eye) were added to training, and the model fine-tuned gently. The first attempt was
rejected: it missed a navy helmet in your workshop clip. The second is in use: 2.5 % of people in
hats judged "helmet worn", no more false alarms end to end, no overfitting, one check failed
narrowly and accepted by you with the reason recorded
([results](docs/caps_results.md), [decision 0010](docs/decisions/0010-caps-and-hats-are-not-helmets.md),
[data](data/caps/README.md)).

## Phase 5: alerts to your phone, stored events, and a dashboard

```bash
bash ~/Documents/ppe-monitor/scripts/mac_phase5.sh setup      # once, ~5 min: libraries, PostgreSQL, the database
bash ~/Documents/ppe-monitor/scripts/mac_phase5.sh telegram   # once, ~2 min: connect the alerts to your Telegram
bash ~/Documents/ppe-monitor/scripts/mac_phase5.sh check      # ~5 min: tests, and the whole path end to end
bash ~/Documents/ppe-monitor/scripts/mac_phase5.sh run        # the 3 simulated cameras -> database -> phone + dashboard
bash ~/Documents/ppe-monitor/scripts/mac_phase5.sh run clips  # ... or 4 of the Phase 2 moving-people clips
bash ~/Documents/ppe-monitor/scripts/mac_phase5.sh dashboard  # only the dashboard, to look back at events
bash ~/Documents/ppe-monitor/scripts/mac_phase5.sh telegram --speed   # how fast this network reaches Telegram
```

**Before `telegram`:** in Telegram, open **@BotFather**, send `/newbot`, and pick a name and a
username. Then open your new bot and press **Start**. The script asks for the token BotFather gave
you (typing is hidden), finds your chat, saves both in `configs/secrets.env` (not in git), and
sends a test alert.

**How it works** ([decision 0008](docs/decisions/0008-storage-alerts-and-dashboard.md)):
- **Storing:** the camera service stores every event in PostgreSQL, with its evidence image and
  the clean frame (`data/events/`), and queues one alert per channel in the same transaction.
  A background thread does the writing, so **the video never waits** for the database.
- **Sending:** the **alert worker** takes alerts from the queue and sends them with the picture to
  Telegram (or email). Failures are retried after 2, 4, 8 ... s, and each alert's delay is
  recorded.
- **Rate limit:** at most 5 alerts per camera every 10 minutes. The rest go out as one summary.
- **The dashboard**, `http://127.0.0.1:8080`, shows:
  - live camera pictures;
  - counts, events per hour and per camera;
  - the event feed with its images;
  - **Confirm** / **False alarm** on each event. False alarms keep their images, for retraining
    in Phase 6.
- **The PostgreSQL server is the project's own** (`data/postgres`, 127.0.0.1:5433, from
  conda-forge). `python scripts/db.py status` / `start` / `stop` manage it. For SQLite instead,
  put `DATABASE_URL=sqlite:///data/ppe.sqlite` in `configs/secrets.env`.
- **Settings:** `configs/server.yaml`: channels, minimum severity, `photo: false` for text-only
  alerts, rate limit, retries, how long images are kept.

**On the Mac (26 Sep 2026):**
- `check`: every check passed, and 4 cameras stayed at 10 frames a second with storage on.
- **Real alerts reach the phone, with the picture, 3–4 s after the event.** That is 0.4 s to
  reach the worker, and the rest is uploading the pictures.
- The first run took 19–23 s: this network announces IPv6 but doesn't carry it. Telegram now
  uses IPv4 only (`ip: ipv4` in `configs/server.yaml`).

Details: [docs/phase5_results.md](docs/phase5_results.md).

## Phase 4: many cameras at once

```bash
bash ~/Documents/ppe-monitor/scripts/mac_phase4.sh export      # once, ~5 min: Core ML and ONNX versions of the models
bash ~/Documents/ppe-monitor/scripts/mac_phase4.sh check       # ~15 min: tests, benchmark, a camera dropping out
bash ~/Documents/ppe-monitor/scripts/mac_phase4.sh run         # the 3 simulated cameras in one window
```

**How it runs** (`python scripts/run_cameras.py`):

- **One reader thread per camera.** It keeps only the newest frame, so a slow or frozen camera
  never holds up the others.
- **Reconnection.** A camera that stops sending is closed and retried after 0.5, 1, 2, 4 and 8 s,
  then every 10 s, and is picked up again by itself.
- **One shared clock.** Ten times a second (`process_fps` in `configs/streams.yaml`), the newest
  frame of every camera goes through the detector in **one batch**. The frames with people in them
  then go through the keypoint model, also in one batch.
- **Keypoints on 1 frame in 2** (`pose: every: 2` in `configs/ppe.yaml`). Between those frames,
  each person's last keypoints are moved onto their new box. This halves the keypoint model's cost,
  with identical results on the test clips.
- **Per camera:** its own tracker, rules and events, exactly as in Phase 3.
- **Output:**
  - events and snapshots go to `runs/live/<time>/`;
  - every drop-out and return goes to `cameras.jsonl`;
  - every 10 s, a status table shows each camera's frame rate, lag, events and reconnections.

**Model formats** (`python scripts/export_models.py`). The book uses TensorRT, which needs an
NVIDIA GPU. On the Mac the same idea gives:

| Format | Numbers | Runs on |
|---|---|---|
| PyTorch | FP32 / FP16 | GPU (Metal) |
| Core ML | FP16, and 8-bit weights | Neural Engine |
| ONNX | FP32, and INT8 calibrated on the validation photos | CPU; the same file runs on a Linux server |

`python scripts/benchmark.py` measures every format twice:
- **accuracy:** mAP@50, and the Phase 2 rule errors;
- **speed:** frames per second, lag, CPU and memory with 1, 2, 4 and 8 cameras.

`python scripts/check_reconnect.py` unplugs one of four fake RTSP cameras for 15 s and checks that
nothing crashes, the others keep their frame rate, and it comes back by itself.

**On the Mac (M4, 25 Sep 2026, fine-tuned model, keypoints on 1 frame in 2)**, frames analysed per
second per camera (target 10). In brackets: 8 cameras with keypoints on every frame, the first run.

| Format | mAP@50 | 1 camera | 2 | 4 | 8 cameras |
|---|---|---|---|---|---|
| PyTorch FP32, GPU | 0.883 | 10 | 10 | 10 | 8.0 (6.1) |
| PyTorch FP16, GPU | 0.884 | 10 | 10 | 10 | 9.2 (7.1) |
| **Core ML FP16, Neural Engine** (the default) | 0.882 | 10 | 10 | 10 | **10.0** (7.7) |
| Core ML 8-bit weights | 0.883 | 10 | 10 | 10 | 9.9 (7.8) |
| ONNX FP32, CPU | 0.883 | 10 | 6.1 | 4.4 | 2.5 (2.0) |
| ONNX INT8, CPU | 0.859 | 10 | 10 | 6.0 | 2.9 (2.2) |

- **Cameras:** one Mac watches 8 cameras at the full 10 frames a second with Core ML.
- **Accuracy:** Core ML and PyTorch FP16 are within 0.001 mAP@50 of PyTorch FP32. ONNX INT8 of the
  fine-tuned model loses 0.024; it isn't used on the Mac, and needs more calibration photos before
  it is used on a server.
- **Drop-outs:** with a camera unplugged for 15 s, the other three kept their 10 frames a second
  (longest gap 0.12 s). It was analysed again 3.7 s after it came back.

Details, and the cloud reference run:
[docs/phase4_results.md](docs/phase4_results.md). Reasons for the design:
[decision 0006](docs/decisions/0006-multi-camera-and-model-formats.md).

## Fine-tuning with more public data (after the first real clips)

The first real clips ([docs/real_clips.md](docs/real_clips.md)) showed the model missing navy,
brown and white helmets indoors. It was fine-tuned with three more public datasets: CHV, GDUT-HWD
and part of SH17. They add helmets of every colour and indoor factory scenes.

```bash
bash ~/Documents/ppe-monitor/scripts/mac_retrain.sh data      # ~20-30 min: download, check the labels, build datasets/ppe4
bash ~/Documents/ppe-monitor/scripts/mac_retrain.sh train     # ~2-4 h: fine-tune, with early stopping
bash ~/Documents/ppe-monitor/scripts/mac_retrain.sh compare   # old against new, on photos neither trained on
bash ~/Documents/ppe-monitor/scripts/mac_retrain.sh accept    # use it, only if it passed
```

**Result on the Mac (25 Sep 2026): the candidate passed every check, and it is now the model in
use** (`models/ppe4_yolo26n_finetune.pt`, with its Core ML and ONNX versions).

| | Phase 1 model | Fine-tuned |
|---|---|---|
| Phase 1 test photos, mAP@50 | 0.869 | **0.883** |
| Extra datasets' test photos (428), helmet AP@50 | 0.666 | **0.918** |
| Blue helmets found | 58 % | **91 %** |
| Workshop clips: helmet found on navy / brown / white-in-haze helmets | 41 % / 0 % / 0 % | **96 % / 91 % / 91 %** |
| Workshop clips: helmet "found" on a bare head | 3 % | 3 % |

- **Overtraining.** Validation loss fell for all 30 epochs and every test set improved. The
  model card's train-against-validation gap is 0.085, just over the 0.08 line;
  [why that's acceptable here](docs/finetune_results.md#overfitting-check).
- **Vests are a trade-off.** The new model is more cautious about vests. On the Phase 2 moving
  clips, false "no vest" alerts on wearers rose from 6 to 11 of 81; the one-frame check is better
  on the test photos and worse on the validation photos. A lower threshold barely helps.
- **Hi-vis isn't required at this site**, so the `no_vest` rule is switched off in
  `configs/rules.yaml` (`enabled: false`). The event check still scores vests.
- **Still wrong:** one false helmet alert on the workshop clips, on a small person in heavy haze
  whose track jumped between people as the phone turned.

Details: [docs/finetune_results.md](docs/finetune_results.md). Why and how:
[decision 0007](docs/decisions/0007-fine-tuning-with-extra-public-data.md).

**Checking the labels.** Images where a person, helmet or vest is visible but not labelled are
left out. GDUT-HWD has no person boxes at all, so a large COCO model draws them.

**Guarding against overtraining:**
- it starts from the Phase 1 model, runs at most 30 epochs, and stops after 8 without improvement;
- the checkpoint kept is the best on validation;
- the model card shows training against validation loss per epoch.

## Phase 3: restricted zones and rules

```bash
bash ~/Documents/ppe-monitor/scripts/mac_phase3.sh check        # about 10 minutes: tests, rules, zone clips, cameras
bash ~/Documents/ppe-monitor/scripts/mac_phase3.sh demo         # the simulated gate camera with its zone
bash ~/Documents/ppe-monitor/scripts/mac_phase3.sh draw cam1    # try the zone drawing tool
```

**Zones** are drawn per camera by clicking on a frame (`scripts/draw_zones.py`). They are stored as
fractions of the frame in `configs/zones.yaml`, so a change of resolution doesn't move them. A
person is inside a zone when the point they stand on is: their ankles if the keypoint model sees
them, otherwise the bottom of their box. Feet out of view mean "can't tell", never an alert.

**Rules** are data in `configs/rules.yaml`, not code. Each rule has:
- a type: `no_helmet`, `no_vest` or `zone_intrusion`;
- the cameras it applies to, and optionally a zone ("helmets required in the pit");
- a severity, a dwell time and a cooldown;
- optional active hours, e.g. 07:00–19:00 Monday to Saturday, or 22:00–06:00 overnight.

`python scripts/check_rules.py` validates both files and shows what applies where. Every rule
goes through the same event engine as Phase 2, so **PPE and zone events come out in exactly the
same shape**: camera, kind, rule, zone, severity, track ID, times, box, frame and snapshot.

**On 86 test clips** in which people walk into a zone (made from the test photos):

| | people |
|---|---|
| Walked in and stayed: exactly one zone event | 28 / 37 |
| People with two zone events | 0 |
| False zone events on people who never went in | 0 / 88 |

The median time from the feet crossing the line to the event is 2.7 s (the rule's dwell is 2 s).
The 9 misses are:
- 4 small people the detector never finds;
- 3 people in groups boxed together;
- 2 people with their feet at the very edge of the frame.

None of them is the zone logic itself. Details are in
[docs/phase3_results.md](docs/phase3_results.md), and the reasons for the design are in
[decision 0005](docs/decisions/0005-zones-and-rules-engine.md).

## Phase 2: tracking and PPE logic

```bash
bash ~/Documents/ppe-monitor/scripts/mac_phase2.sh check   # about 15 minutes: tests, rule check, 3 event checks
bash ~/Documents/ppe-monitor/scripts/mac_phase2.sh demo    # watch it on a test clip (press q to close)
```

Every frame now goes through a chain:

1. **The detector** finds people, helmets and vests.
2. **ByteTrack** gives each person a track ID that stays the same from frame to frame, including
   while they walk, and when frames are skipped.
3. **Body keypoints** (a second small model) say where each person's head is and whether they are
   upright, bending, crouching or lying.
4. **The PPE rule** decides per person whether the helmet and vest are *worn*, *missing* or *unknown*:
   - a helmet counts when it sits in the head region of that person's box, or on the head the
     keypoints found (someone bent over);
   - a vest counts when it sits on the torso;
   - *unknown* means the person is too small or cut off to judge, or bent over with their head out
     of sight.
5. **The event engine** turns those per-frame answers into alerts:
   - it takes a 1-second majority vote;
   - a violation must last **3 seconds** before it counts;
   - it raises **one event per incident**, even if the tracker changes the person's ID while they walk;
   - the same person can't alert again within 60 seconds.

All settings are in `configs/ppe.yaml`. The results are in
[docs/phase2_results.md](docs/phase2_results.md); why walking, bending and crouching are handled
this way is in [decision 0004](docs/decisions/0004-posture-aware-ppe-rule.md).

**The rule's geometry** (checked on 346 test-set people whose status is known from the labels):
- on labelled boxes, it judges 0.7% of helmet wearers and no vest wearers as missing;
- it never judges a bare head or bare torso as wearing PPE.

**The whole chain on 86 test clips** made from test photos, with flicker and a passing occluder,
first with people standing still, then with everyone moving at walking pace:

| | still, helmet | still, vest | moving, helmet | moving, vest |
|---|---|---|---|---|
| People without it who got exactly one event | 25 / 28 | 35 / 45 | 42 / 47 | 55 / 68 |
| People who got duplicate events | 0 | 0 | 0 | 0 |
| False events on people wearing it | 4 / 119 | 6 / 81 | 5 / 119 | 6 / 81 |

**Bending over with a helmet on** no longer looks like "no helmet": with keypoints, 98% of worn
helmets on people bending over are found (92% before), and a worker who stays bent over *without* a
helmet is still caught. Processed at only 5 fps, groups walking together swap IDs and a few people
get two alerts, so such cameras need 10 fps or more. On the Mac the whole chain, keypoints included,
takes about 29 ms per frame (checked 2026-09-24).

- **Frame-to-frame flicker never caused an event.** Every false event came from a person the detector
  got wrong in most frames.
- **Most of the remaining errors** are crowds the detector can't separate, a few real detector misses
  (reflective strap vests at night), and test labels that disagree with the project's definitions.

**Try it on any video:**

```bash
python scripts/monitor.py --source my_video.mp4     # boxes: green ok, amber confirming, red violation, grey can't judge
python scripts/monitor.py --camera cam1 --headless  # a camera; events + snapshots -> runs/monitor/<camera>_<time>/
```

## Phase 1: data and a baseline detector

Two commands on the Mac, with a review in between (plus an optional third):

```bash
bash ~/Documents/ppe-monitor/scripts/mac_phase1.sh prepare   # about 20 minutes the first time
bash ~/Documents/ppe-monitor/scripts/mac_phase1.sh train     # about 3 hours on an M4 MacBook Air
bash ~/Documents/ppe-monitor/scripts/mac_phase1.sh review    # 1 minute: rebuild the mistake gallery
```

Run one at a time: the script refuses to start while another Phase 1 run is still going.

**`prepare`** installs Ultralytics YOLO and torchvision (`requirements-ml.txt`), builds the
training data and runs a one-epoch training check on the GPU. That check proves training works
and estimates how long the full run will take (`logs/train_quick.txt`).

**`train`** fine-tunes YOLO26n (COCO-pretrained, 2.6 M parameters) for up to 60 epochs, stopping
early once it stops improving. It then measures the best checkpoint and builds a gallery of its
mistakes. Keep the Mac plugged in with the lid open; the script stops it from sleeping.
Options pass through, e.g. `... train --epochs 40`.

### Results (23 Sep 2026)

YOLO26n trained on the Mac's GPU for 54 epochs (stopped early; best epoch 39). Test split, 200 images, one
working threshold (0.35) picked on validation:

| | AP@50 | Precision | Recall |
|---|---|---|---|
| person | 0.838 | 0.826 | 0.848 |
| helmet | 0.921 | 0.880 | 0.857 |
| vest | 0.849 | 0.852 | 0.789 |
| **mAP@50** | **0.869** | | |

The failure cases are written up in [docs/phase1_failure_cases.md](docs/phase1_failure_cases.md). The main ones:
- **The test labels miss many real people.** Precision is understated; it's about 0.93 once those are counted.
- **Real helmets and vests found with too little confidence.** Each one would be a false "no PPE" alarm, so
  Phase 2 must wait for a violation to persist.
- **Crowds.**
- **An inconsistent "vest" class.** Harnesses and life jackets are labelled "vest", while hi-vis jackets are
  often missed.
- **Look-alikes:** caps called helmets, and orange or blue clothing called vests.
- **Small helmets.** Recall is 0.46 on them. The public data can't measure distant workers; your own clips
  are needed for that.

### What it produces

| File | What's in it |
|---|---|
| `datasets/ppe3/report.md` | How every dataset label was treated, which images were moved to stop test-set leaks, counts per split and per object size |
| `models/ppe3_yolo26n_baseline.pt` | The trained model |
| `models/ppe3_yolo26n_baseline.md` | Model card: settings, time, per-class AP@50, precision and recall at the working threshold, recall by object size, data licences |
| `runs/review/<model>_ppe3_test/index.html` | Mistake gallery: every missed and false box on the test split, grouped by kind, most confident first |
| `runs/detect/baseline/results.png` | Ultralytics' training curves |

### The training data

Two public datasets are downloaded and checked against a SHA-256, then merged onto the project's
three classes through `configs/classes.yaml`:

| Source | Licence | Images |
|---|---|---|
| [Construction-PPE](https://docs.ultralytics.com/datasets/detect/construction-ppe/) (Ultralytics) | AGPL-3.0 | 1,416 |
| [Construction Safety](https://universe.roboflow.com/object-detection/construction-safety-gsnvb) (Roboflow 100) | CC BY 4.0 | 1,206 |

"No helmet" / "no vest" labels are dropped. Phase 2 decides what's missing by matching helmets
and vests to people. A label the config doesn't know stops the build rather than being
guessed. Before splitting, near-duplicate photos are found by image hash, and any group that
crosses splits goes to train: 56 images were moved and 15 photos found in both datasets were
kept once. Details: [decision 0003](docs/decisions/0003-training-data-and-honest-test.md).

Result: **2,170 train / 237 val / 200 test images**.

### How the model is measured

- **AP@50 per class** on val and test, computed by the project's own matcher
  (`ppe_monitor.vision.matching`) and cross-checked against Ultralytics' figure.
- **One working threshold**, the confidence with the best F1 on val. Precision and recall on test
  are reported at that threshold, so test never influences a setting.
- **Recall by object size** (COCO sizes at 640 px), because distant workers are the hard case for
  CCTV.

**Limit of the public test split:** it has almost no small (distant) people (3 of 409), so it
can't tell you how the model does on typical CCTV views. That needs your own labelled clips
([data/README.md](data/README.md)); `scripts/evaluate_detector.py --data <their data.yaml>`
measures any YOLO-format split the same way.

### Try the model

```bash
python scripts/fake_cameras.py --with-server           # terminal 1
python scripts/detect.py --camera cam1                 # terminal 2: boxes drawn live
python scripts/detect.py --source some_photo.jpg       # or a photo / video file
```

The synthetic clips have cartoon workers, so expect few or no detections on them. Their job is
the video plumbing. Real footage is the real test. For live streams, `detect.py` keeps only the
newest frame, so a slow detector skips frames instead of falling behind.

## What Phase 0 gives you

- **A fake CCTV network on your own machine.** MediaMTX acts as the RTSP server, and one
  ffmpeg process per camera loops a video file into it forever. To the rest of the system these
  look exactly like IP cameras, so every later phase can be built and tested without site access.
- **Synthetic test clips** with scripted moments: a helmet taken off and left on a railing
  (6–12 s), a worker with no vest, a person inside the restricted zone (8–17 s).
- **A stream viewer and checker.** It opens any RTSP stream with OpenCV and measures whether it
  plays back correctly: steady frame rate, no stalls, no grey frames.
- **An environment report** that pins every version (`docs/ENVIRONMENT.md`), so a mismatch shows
  up now instead of in Phase 4.
- **Apple GPU support.** A GPU check proves PyTorch runs on the Mac's GPU (MPS) and gives the
  same answers as the CPU, and measures the speed-up. The fake cameras encode video on the Mac's
  hardware video encoder instead of the CPU.
- **Two recorded decisions:** the class scheme ([0001](docs/decisions/0001-class-scheme.md)) and
  running on the Mac's GPU ([0002](docs/decisions/0002-apple-silicon-dev-machine.md)).
- **Automated tests**, including an end-to-end test (clip → ffmpeg → MediaMTX → OpenCV) and
  one that runs on the Apple GPU.

## Quick start on a Mac: one command

Open **Terminal** and run:

```bash
bash ~/Documents/ppe-monitor/scripts/mac_setup.sh
```

It's safe to run again and skips anything already done. It:

1. checks the Mac (chip, macOS version, and that Terminal isn't running in Intel/Rosetta mode);
2. finds or creates a native **arm64 Python 3.11** environment: a conda env called `ppe` if
   your conda is the Apple Silicon build, otherwise a `.venv` from Homebrew's Python. An Intel
   Python is never used, because it can't reach the Apple GPU;
3. installs ffmpeg (Homebrew, or conda-forge when there's no Homebrew);
4. installs the pinned packages, plus PyTorch for the GPU (`requirements-gpu.txt`);
5. downloads MediaMTX (checksum-verified), then runs the environment report, the Phase 0
   acceptance check, the test suite and the Apple GPU check, and prints a summary.

Everything is saved to `logs/mac_setup.log`. The first run takes 5–15 minutes, mostly
downloads. If macOS asks whether Terminal may access the Documents folder, click **Allow**.

Afterwards, work in the project from a new Terminal window with
`cd ~/Documents/ppe-monitor && conda activate ppe` (or `source .venv/bin/activate` if the setup
made a `.venv`; the summary tells you which).

### Or step by step (any OS)

```bash
# ffmpeg, then a Python 3.11 environment, e.g.
conda create -n ppe -c conda-forge python=3.11 -y && conda activate ppe
cd ppe-monitor
pip install -r requirements-dev.txt -r requirements-gpu.txt
python scripts/get_mediamtx.py        # MediaMTX into tools/, checksum-verified
python scripts/check_env.py --write   # environment report -> docs/ENVIRONMENT.md
python scripts/verify_phase0.py       # Phase 0 acceptance check
python scripts/check_gpu.py           # does PyTorch run on the GPU?
```

Phase 0 is done when the acceptance check ends with **`PASS`**: every simulated camera plays
back correctly in OpenCV on your machine (report: `logs/phase0_report.txt`). The GPU check
should also say **`PASS`** on the Mac (report: `logs/gpu_report.txt`).

### See it with your own eyes

Terminal 1, the camera network (leave it running):

```bash
python scripts/fake_cameras.py --with-server
```

Terminal 2, watch a camera (press `q` to close the window):

```bash
python scripts/view_stream.py --camera cam1
python scripts/view_stream.py --all --headless           # 10-second check of every camera
python scripts/view_stream.py --camera cam2 --transport udp   # Chapter 11's TCP vs UDP
```

Any RTSP player works too: in VLC, *File → Open Network* and enter `rtsp://localhost:8554/cam1`.

### Tests

```bash
python -m pytest            # 93 tests, about 40 seconds
```

The end-to-end tests run MediaMTX on a spare port, so they work even while your camera network
is running. They're skipped automatically if ffmpeg or MediaMTX isn't installed.

## Everyday commands

| Command | What it does |
|---|---|
| `bash scripts/ppe.sh app` / `start` / `dashboard` / `photo` / `video` / `camera` / `zones` / `rules` / `telegram` / `check` / `evaluate` / `feedback` / `retrain` / `model` / `stop` | Day-to-day use: [docs/USER_GUIDE.md](docs/USER_GUIDE.md) |
| `bash scripts/make_mac_app.sh [--login \| --no-login \| --remove]` | The "PPE Monitor" app: double-click to start the monitor (restarting any part that stops, the Mac kept awake), or to open the dashboard if it's running |
| `python scripts/check_photo.py PHOTO...` | Who wears a helmet and a vest in each photo, as the cameras would judge one frame → `PHOTO_checked.jpg` |
| `python scripts/verify_phase0.py` | One-shot acceptance check; starts and stops everything itself |
| `python scripts/fake_cameras.py --with-server` | Run the fake cameras and MediaMTX until Ctrl+C |
| `python scripts/view_stream.py --camera cam1` | Watch a camera; prints a playback report when you close it |
| `python scripts/make_test_clips.py [--force]` | (Re)create the synthetic clips |
| `python scripts/check_env.py --write` | Environment report → `docs/ENVIRONMENT.md` |
| `python scripts/get_mediamtx.py` | Download MediaMTX v1.21.0 into `tools/` |
| `python scripts/check_gpu.py` | Is PyTorch using the GPU, and how much faster than the CPU? → `logs/gpu_report.txt` |
| `bash scripts/mac_setup.sh` | Full Mac setup + all checks → `logs/mac_setup.log` |
| `bash scripts/mac_phase1.sh prepare` / `train` / `review` | Phase 1 on the Mac (see above) → `logs/phase1_*.log`; one run at a time |
| `python scripts/prepare_data.py` | Download, merge and de-leak the training data → `datasets/ppe3/` |
| `python scripts/train_detector.py [--quick]` | Train (or time one epoch) → `models/` + model card |
| `python scripts/evaluate_detector.py [--data ... --split ...]` | Per-class AP@50, precision/recall, recall by size → `runs/eval/` |
| `python scripts/review_mistakes.py` | Gallery of the model's mistakes, with why each object was missed → `runs/review/` |
| `python scripts/detect.py --camera cam1` | Run the detector on a camera, video or image |
| `bash scripts/mac_phase2.sh check` / `demo` | Phase 2 on the Mac → `logs/phase2_*.log` |
| `python scripts/monitor.py --source video.mp4` | Full chain: tracked people, PPE verdicts, events + snapshots → `runs/monitor/` |
| `python scripts/check_ppe_rules.py [--split val --sweep]` | The PPE rule against people whose status is known → `runs/ppe_rules/` |
| `python scripts/make_event_clips.py` | Build the 86 test clips from test photos → `datasets/event_clips/` |
| `python scripts/make_event_clips.py --motion sway` | The same clips with everyone moving at walking pace → `datasets/event_clips_sway/` |
| `python scripts/evaluate_events.py [--clips datasets/event_clips_sway] [--stride 3]` | Score the whole chain's events on those clips (`--stride 3`: every 3rd frame only) → `runs/events/` |
| `python scripts/simulate_events.py` | Dwell time vs false alerts with simulated detector errors |
| `bash scripts/mac_phase3.sh check` / `demo` / `draw` | Phase 3 on the Mac → `logs/phase3_*.log` |
| `python scripts/draw_zones.py --camera cam1` | Draw a camera's restricted zones by clicking → `configs/zones.yaml` |
| `python scripts/check_rules.py [--at "2026-09-24 20:30"]` | Validate rules and zones, show what applies where and when → `runs/zones/` |
| `python scripts/monitor.py --source clip.mp4 --as-camera cam3 --start-time "..."` | A recording with a camera's zones and rules, as if at that time |
| `python scripts/make_event_clips.py --motion pan` | Test clips of people walking into a zone → `datasets/event_clips_zone/` |
| `bash scripts/mac_phase4.sh export` / `check` / `run [clips]` | Phase 4 on the Mac → `logs/phase4_*.log` |
| `python scripts/run_cameras.py [--cameras cam1,cam3] [--backend coreml] [--show]` | Watch several cameras at once → `runs/live/` |
| `python scripts/run_cameras.py --files datasets/event_clips_sway --count 4` | Video files played as live cameras |
| `python scripts/export_models.py [--backend coreml]` | Core ML / ONNX versions of the detector and keypoint model → `models/exported/` |
| `python scripts/benchmark.py [--backends pytorch,coreml --streams 1,4]` | Speed against accuracy for every format and camera count → `runs/benchmark/` |
| `python scripts/check_reconnect.py [--cameras 8 --down 30]` | Unplug a fake camera and plug it back in: do the others carry on? → `runs/reconnect/` |
| `bash scripts/mac_retrain.sh data` / `train` / `compare` / `accept` | Fine-tuning with extra public datasets → `logs/retrain_*.log` |
| `python scripts/prepare_more_data.py [--sources chv gdut-hwd sh17]` | Download, check and merge the extra datasets → `datasets/ppe4/`, pictures in `runs/data_review/ppe4/` |
| `python scripts/compare_models.py [--accept]` | The model in use against the newest in `models/candidates/` → `runs/compare/` |
| `bash scripts/mac_phase5.sh setup` / `telegram` / `check` / `run [clips]` | Phase 5 on the Mac → `logs/phase5_*.log` |
| `python scripts/db.py init` / `start` / `stop` / `status` | The project's PostgreSQL server (`data/postgres`, 127.0.0.1:5433) |
| `python scripts/run_cameras.py --db --live` | Cameras, with events stored in the database and live pictures for the dashboard |
| `python scripts/alert_worker.py [--once]` | Send queued alerts (Telegram, email) → `logs/alert_worker.log` when run by `mac_phase5.sh run` |
| `python scripts/serve_dashboard.py [--host 0.0.0.0]` | The dashboard, http://127.0.0.1:8080 (another host needs `DASHBOARD_PASSWORD`) |
| `python scripts/telegram_setup.py [--test]` | Connect the alerts to your Telegram, send a test alert |
| `python scripts/check_phase5.py` | Cameras -> database -> alert worker -> stand-in Telegram -> dashboard API, checked → `runs/phase5_check/` |
| `bash scripts/mac_phase6.sh check` / `evaluate` / `feedback` / `retrain` / `accept` / `docker` / `annotate` | Phase 6 on the Mac → `logs/phase6_*.log` |
| `python scripts/evaluate_system.py [--sets site] [--telegram 5] [--text-only]` | Every test clip through the whole system: the five headline numbers → `runs/evaluation/` |
| `bash scripts/mac_caps.sh run` / `accept [--override "reason"]` | Photos of people in hats → `datasets/ppe4caps`, fine-tune, compare, accept → `logs/caps_*.log` |
| `python scripts/evaluate_system.py --rescore runs/evaluation/<time>` | Score an earlier run again without replaying it |
| `python scripts/annotate_clip.py data/clips/<clip>.mp4` | Mark a clip's real violations (window) → `data/ground_truth/<clip>.yaml` |
| `python scripts/feedback.py export` / `status` / `build` | Dashboard false alarms → `datasets/feedback/` (with `review.html`) → `datasets/ppe5/` |
| `python scripts/compare_models.py --checks feedback --evaluations OLD NEW` | Accept a retrained model only if it isn't worse, on photos and end to end |
| `docker compose up -d --build` | The whole system in Docker (see above) |
| `python scripts/check_compose.py [--keep]` | `docker compose up` from this folder, checked end to end → `runs/compose_check/` |


## How the pieces fit

```
 data/clips/synthetic_cam1.mp4 ──ffmpeg (loop, H.264, keyframe every 1 s)──┐
 data/clips/synthetic_cam2.mp4 ──ffmpeg ─────────────────────────────────┤──► MediaMTX ──► rtsp://localhost:8554/cam1..3
 data/clips/synthetic_cam3.mp4 ──ffmpeg ─────────────────────────────────┘    (127.0.0.1)         │
                                                                                                  ▼
                                                            OpenCV (ppe_monitor.ingestion.rtsp) → view / check
```

Three details from building and testing this, all relevant to later phases:

- **Low-latency opening.** With default settings, OpenCV/FFmpeg spent ~2.8 s analysing each
  stream and started **31 frames (~2 s) behind live**. The project opens streams with FFmpeg's
  low-latency options (`fflags nobuffer`, 0.5 s analysis): about 0.8 s to open, 4 frames of
  backlog, same steady 15 fps. If a real camera needs longer analysis, it automatically retries
  with the defaults.
- **Those options must not leak.** OpenCV reads them from a process-wide environment variable.
  If they're left set, every video *file* opened afterwards silently loses half its frames,
  which would quietly corrupt the evaluations in Phases 1 and 6. `open_capture` sets them only
  while a stream opens, and a regression test guards this.
- **No freeze when a clip loops.** The first run on the Mac showed every fake camera freezing for
  about 1 s each time its clip looped back to the start. ffmpeg decoded the clip on one thread
  per CPU core, which keeps a frame per core in flight, and that pipeline is flushed at every
  loop, so the gap grows with the number of cores (0.3 s on a 2-core machine, 1.1 s on a 10-core
  M4). The publishers now decode on one thread. The acceptance check reads each stream past its
  clip's loop point with a 0.5 s stall limit, and the end-to-end test does the same, so this
  can't come back unnoticed.

## Project layout

```
ppe-monitor/
├── configs/
│   ├── cameras.yaml        camera registry (URLs, transport; sim_source = fake camera clip)
│   ├── classes.yaml        detection classes + dataset label aliases (decision 0001)
│   ├── ppe.yaml            Phase 2 settings: thresholds, PPE geometry, keypoints, tracking, dwell/cooldown
│   ├── rules.yaml          Phase 3 rules: type, cameras, zone, severity, dwell, cooldown, active hours
│   ├── zones.yaml          restricted zones per camera (0-1 corners); zones/ = the frames they were drawn on
│   ├── streams.yaml        Phase 4: frames analysed per second, model format, reconnection waits
│   ├── server.yaml         Phase 5: database, image storage, alert channels, rate limit, dashboard;
│   │                       Phase 6: camera-down and silent-service alerts (alerts.health)
│   ├── secrets.env.example template for secrets.env (tokens, passwords; the real one is not in git)
│   └── mediamtx.yml        RTSP server config (localhost only)
├── src/ppe_monitor/
│   ├── config.py           loads and validates cameras.yaml
│   ├── env_check.py        environment report
│   ├── device.py           picks cuda / mps (Apple GPU) / cpu, GPU check and benchmark
│   ├── executables.py      finds ffmpeg / ffprobe / MediaMTX, picks a working H.264 encoder
│   ├── ingestion/rtsp.py   open streams (low latency), RTSP status check, playback probe
│   ├── ingestion/reader.py one thread per camera: newest frame only, reconnection with backoff
│   ├── sim/                synthetic clips + the fake camera network
│   ├── data/               dataset download, label mapping, near-duplicate check, merged build, false alarms
│   │                       as training data (feedback.py),
│   │                       known helmet/vest status from absence labels (compliance.py); extra
│   │                       datasets, label check and the ppe4 build (extra_sources.py, audit.py, build_ext.py);
│   │                       photos of people in hats from Open Images (caps.py)
│   ├── vision/             detector, box matching + AP, evaluation, mistake review, person tracking,
│   │                       body keypoints and posture (pose.py), model formats and exports (backends.py)
│   ├── rules/              PPE rule (ppe.py), zones (zones.py), rules engine (engine.py), event engine
│   │                       (events.py), zone editor, rule check, event scoring
│   ├── pipeline.py         one camera's chain: detect -> track -> PPE rule -> events
│   ├── service.py          many cameras: one clock, batched models, per-camera chains, event sink, status
│   ├── draw.py             overlays and event snapshots
│   ├── backend/            Phase 5: settings, tables, event store, DB writer thread, alert channels
│   │                       and worker (+ camera-down alerts), the API, the local PostgreSQL (localpg.py),
│   │                       a stand-in Telegram for checks (standin.py)
│   ├── evaluation/         Phase 6: ground truth, matching alerts to it and the numbers, the annotation tool
│   └── dashboard/static/   the dashboard page (plain HTML, CSS, JavaScript)
├── scripts/                command-line tools listed above; ppe.sh for everyday use, make_mac_app.sh for the app
├── assets/                 the app's icon
├── tests/                  pytest suite (unit + end-to-end)
├── data/                   clip guide (README.md), clips/ (not in git), clips_catalog.csv,
│                           ground_truth/ (each test clip's real violations), caps/ (the Open Images photos
│                           of people in hats: list, checksums, labels, credits)
├── docs/decisions/         why things are the way they are
├── Dockerfile              one image for the camera service, alert worker and dashboard (CPU; GPU: override)
├── docker-compose.yml      the whole system: database, cameras, alerts, dashboard, demo cameras
├── docker-compose.gpu.yml  the camera service on an NVIDIA GPU (untested)
├── datasets/               downloaded + merged training data (not in git)
├── models/                 trained models + model cards; exported/ = Core ML / ONNX versions (not in git)
├── runs/                   training runs, evaluations, mistake galleries (not in git)
├── tools/                  MediaMTX binary (downloaded, not in git)
└── logs/                   ffmpeg/MediaMTX logs, phase reports (not in git)
```

## Adding a real camera later

Add an entry to `configs/cameras.yaml` without `sim_source`, and keep the password out of the file:

```yaml
  - id: gate
    name: Main gate
    url: rtsp://admin:${CAM_GATE_PASSWORD}@192.168.1.64:554/Streaming/Channels/101
```

```bash
export CAM_GATE_PASSWORD='...'
python scripts/view_stream.py --camera gate
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ffmpeg not found` | `brew install ffmpeg` (or `conda install -c conda-forge ffmpeg`) |
| `get_mediamtx.py`: `CERTIFICATE_VERIFY_FAILED` | Python from python.org: run *Install Certificates.command* in Applications → Python 3.11. Conda Python doesn't need this. |
| MediaMTX download blocked | Download `mediamtx_v1.21.0_darwin_arm64.tar.gz` from the [releases page](https://github.com/bluenviron/mediamtx/releases/tag/v1.21.0) and put the `mediamtx` file in `tools/mediamtx/` |
| macOS: "mediamtx cannot be opened because the developer cannot be verified" | Only happens if you downloaded it with a browser: `xattr -d com.apple.quarantine tools/mediamtx/mediamtx` |
| `Something is already listening on localhost:8554` | A MediaMTX is already running: `lsof -i :8554` to find it, or run `fake_cameras.py` without `--with-server` to reuse it |
| `nothing is answering at localhost:8554` | Start the network first: `python scripts/fake_cameras.py --with-server` |
| `server is running, but nothing is publishing to /camX` | That camera's ffmpeg isn't running; check `logs/fake_cameras/camX.log` |
| fps below expected | Close heavy apps; on battery, turn off Low Power Mode. `python scripts/check_env.py` shows which video encoder the fake cameras use |
| GPU check: "Intel (x86_64) build running under Rosetta" | Your Python is the Intel version. Run `scripts/mac_setup.sh`: it skips Intel Pythons and makes a native one |
| GPU check: "MPS built False" | PyTorch was installed from somewhere without Apple GPU support: `pip install --force-reinstall -r requirements-gpu.txt` |
| `pip` can't find torch 2.14 | Expected on macOS 13 or older; pip installs 2.11.0 instead, which also supports the Apple GPU |
| No window appears | You have `opencv-python-headless` installed; `pip uninstall opencv-python-headless && pip install -r requirements.txt`, or use `--headless` |
| Your own clip joins slowly with `--copy` | Its keyframes are far apart; leave out `--copy` so it's re-encoded with one every second |
| `prepare_data.py`: "labels that configs/classes.yaml doesn't know" | A dataset has a new label. Look at a few images with it, then add it to an alias list or `ignore` in `classes.yaml` |
| `prepare_data.py`: checksum mismatch | The download was cut off or the file changed upstream. Delete it from `datasets/raw/` and run again |
| Training: `NotImplementedError ... MPS` | An operation isn't on the Apple GPU yet; the scripts already enable the CPU fallback. If it still fails, report the message, and use `--device cpu` meanwhile |
| Training much slower than `logs/train_quick.txt` predicted | The Mac is hot or on battery: plug it in, keep the lid open, close heavy apps. `--epochs 30` halves the time |
| `backend coreml: not exported yet` | `bash scripts/mac_phase4.sh export` (or `python scripts/export_models.py --backend coreml`) |
| Export: `numpy` version conflict | The Core ML converter needs numpy 2.3.5 or older: `pip install numpy==2.3.5` |
| `run_cameras.py`: a camera stays `reconnecting` | It isn't reachable. The status line shows why; check it with `python scripts/view_stream.py --camera <id>`. The others carry on meanwhile |
| Status table: every camera below the target fps | The machine can't keep up. Lower `process_fps` in `configs/streams.yaml`, watch fewer cameras, or pick a faster `backend` (see the benchmark report) |

## Next steps

The project is complete: all six phases are built, checked and reviewed. What would move the
numbers most from here:
- **Fixed-camera footage from the site**, 1–3 minutes per clip, high up and in landscape. Mark its
  violations with `bash scripts/mac_phase6.sh annotate <clip>`. That gives real precision and
  recall, and false alerts per hour on real scenes.
- **More of it, not for testing**, played as cameras (`clips_catalog.csv` split `train`). Mark the
  false alarms on the dashboard, then `bash scripts/mac_phase6.sh feedback` and `retrain`. That
  closes the loop with lessons from the site itself.
- **Machines taken for people.** The tractor at the edge of a Pexels clip is the remaining false
  alarm on real video. On your own cameras, mark such alerts "False alarm" on the dashboard: the
  feedback loop turns them into training images (delete the "person" box on them).
- **Text-only alerts** (`photo: false`), with the alert worker taking new alerts while others are
  still sending, to bring alerts on the phone under 5 s.

## Licence, and what this repository leaves out

**The code is AGPL-3.0** ([LICENSE](LICENSE)), the same licence as Ultralytics, which it is built
on. Anyone who runs a changed version as a service for other people must offer them its source.

**The model in `models/ppe4caps_yolo26n_caps2.pt` is for non-commercial use only** (learning,
research, a portfolio). Part of its training data is SH17, licensed CC BY-NC-SA 4.0
(non-commercial), and GDUT-HWD, which states no licence. The rest is Construction-PPE (AGPL-3.0),
Construction Safety (CC BY 4.0), CHV (free to use, with citation) and Open Images V7 (boxes CC BY
4.0, photos CC BY 2.0). Commercial use would need a model trained only on data that allows it,
and an Ultralytics Enterprise licence or a differently licensed detector. Details: decisions
[0003](docs/decisions/0003-training-data-and-honest-test.md),
[0007](docs/decisions/0007-fine-tuning-with-extra-public-data.md) and
[0010](docs/decisions/0010-caps-and-hats-are-not-helmets.md).

**Not in the repository** (git-ignored; each machine has or makes its own):

| What | Why | To get it |
|---|---|---|
| `configs/secrets.env` | Telegram token and passwords | `bash scripts/ppe.sh telegram`, and section 3 of the guide |
| `data/clips/` (the catalogue is kept) | large, and real clips show real people | your own cameras; the public clips from the links in `clips_catalog.csv` |
| `datasets/`, `runs/` | large, and rebuilt from their sources | `prepare_data.py`, `prepare_more_data.py`, `prepare_caps_data.py` (each checks its downloads) |
| older models, `models/candidates/`, `models/exported/` | large | Core ML and ONNX versions: `bash scripts/mac_phase4.sh export`. Optional: without them the monitor runs the model in PyTorch on the Apple GPU |
| `models/pretrained/` | public models | downloaded on first use (the keypoint model, and the large COCO model for audits) |
| `data/postgres/`, `data/events/`, `logs/` | this machine's events, evidence pictures and logs | made as it runs |

On a new Mac: clone the repository into `~/Documents/ppe-monitor`, run
`bash ~/Documents/ppe-monitor/scripts/mac_setup.sh`, then follow [the guide](docs/USER_GUIDE.md).
