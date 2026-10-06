# 0009 — Measuring end to end, learning from false alarms, and packaging

**Status:** accepted: Phase 6 done, 29 Sep 2026 (checked on the Mac and, for Docker Compose, from a clean copy in the cloud, 26 Sep; on real video from six other sites, 28 Sep) · **Book:** Chapters 4, 14 and 23

## Context

- **The book's Phase 6** (Chapter 23):
  - annotate the real violations in the test clips, with their start and end;
  - measure event precision and recall, false alerts per camera-hour and alert latency;
  - use the dashboard's false alarms as training data, retrain, and measure again;
  - Docker Compose with secrets in env files and a password on the dashboard;
  - a README with an architecture diagram, real numbers, a demo and known limitations.
- **Done when:** `docker compose up` brings up the whole stack on a clean machine, and the README
  states real, measured numbers for all five headline metrics of Chapter 4.
- **What there is to measure with:**
  - Real footage: 27 s of hand-held phone video from one workshop (2 clips). Both are test clips,
    kept private.
  - 258 generated clips (Phase 2 and 3), made from labelled public test photos, with every
    labelled person's status known.
  - No false alarms yet from footage that may be trained on: every dashboard event so far came
    from test clips.
- **This Mac has no Docker.** Docker on a Mac runs Linux in a virtual machine, which can use
  neither the Apple GPU nor Core ML.

## Decision

1. **Ground truth as violation episodes** (`src/ppe_monitor/evaluation/truth.py`).
   - Each episode has a kind, a start and an end, and where the person is: a box at a few
     moments, followed in a straight line between them.
   - An episode is one of three sorts:
     - **sustained:** an alert is expected; no alert counts as a miss;
     - **not sustained:** real, but shorter than the dwell or mostly hidden; an alert on it
       counts neither way;
     - **compliant:** a person known to wear the item, or never in the zone; an alert on them is
       a false alarm.
   - **Two sources:**
     - Hand-annotated YAML for real footage (`data/ground_truth/`). `scripts/annotate_clip.py`
       is a window to mark episodes and draw boxes.
     - The labels of the generated clips.
   - **Complete or incomplete truth:**
     - Hand-annotated truth is "complete": everyone near enough to judge was checked, so an alert
       on nobody known is a false alarm.
     - Generated truth is not: the source photos leave some people unlabelled, and an alert on
       one of them can't be judged. Example: a woman without a helmet in a photo whose labels
       only cover the two men.
2. **End to end means the real code paths, timed against the clip's own clock**
   (`scripts/evaluate_system.py`).
   - Each clip plays once as a camera, at its own speed, 8 at a time.
   - Every part is the real one: the camera service, the database writer, PostgreSQL (a
     throw-away `ppe_eval` database), the alert worker, and Telegram's HTTP API.
   - The API is a stand-in on the same machine, reached through the real HTTP client. With
     `--telegram N`, N alerts from public photos also go to the real Telegram, to time real
     delivery.
   - A frame's time on camera is known exactly: the reader records when each clip's first frame
     arrived, and files are paced at their own frame rate. So latency is measured from the
     moment a violation begins on camera to the moment the alert is delivered.
   - **Matching:** an alert matches an episode of its kind when the event was confirmed between
     1 s before the episode began and 4 s after it ended, and its person box overlaps the
     episode's (IoU ≥ 0.3, or its centre inside).
   - **The numbers:**
     - precision: real alerts ÷ scored alerts (duplicates count as real, and are listed);
     - recall: caught ÷ sustained episodes;
     - false alerts per camera-hour of footage;
     - latency: median, 90 % and worst.
   - **Error bars:** Wilson intervals for the rates and exact Poisson intervals for the hourly
     rate. The test set is small, and a single number would hide how uncertain it is.
   - **Rules:** the site's own `configs/rules.yaml`, so the vest rule stays off. On the zone
     clips, one zone rule with a 2 s dwell.
   - **The rate limit is off** for the evaluation: it would hold back alerts that must be timed.
3. **False alarms become training data only through a person, and never from test footage**
   (`src/ppe_monitor/data/feedback.py`, `scripts/feedback.py`).
   - **Labels:** each false alarm's clean frame is labelled with what the model in use finds at
     the rules' thresholds. The verdict adds the lesson.
     - A false "no helmet" means the person wore one. So the helmet on their head is looked for
       again, down to confidence 0.05, and must be at least 15 % of the person's width (not the
       small helmet of someone behind). If found, it becomes a label.
     - If none is found, the image waits for a box drawn by hand.
     - Zone false alarms always go to a person: only a person can tell a wrong zone from a
       non-person.
   - **Review:** `review.html` shows every image with its boxes; `skip.txt` leaves any out.
   - **Test footage is never used:** only real cameras, or clips listed with split `train` in
     `data/clips/clips_catalog.csv`, may be trained on. Otherwise the model would be tested on
     what it was taught.
   - **Retraining:**
     - `datasets/ppe5` = ppe4, plus each ready false alarm 5 times, in training only. Validation
       and test stay exactly as they were.
     - It starts from the model in use: at most 10 epochs, early stop after 4 without
       improvement.
     - **Accepted only if all four hold:**
       - mAP@50 is no more than 0.01 lower on the Phase 1 test photos;
       - mAP@50 is no more than 0.01 lower on all the extra test photos;
       - no more false alarms on the test clips, end to end;
       - at most 2 more missed violations there.
   - **At least 10 false alarms are needed** before a retrain is worth its hours.
4. **The system watches itself** (`backend/worker.py`).
   - A camera down for more than 60 s raises one alert, and another when it is back. Cameras that
     go down together share one message.
   - A camera service with no heartbeat for more than 60 s raises one alert, since a silent
     safety system is worse than none.
   - A deliberate stop raises nothing: the service says so as it ends.
   - The alerts go to the channels that take "high" alerts, and unsent ones are tried again.
5. **Docker Compose: one image, one container per service** (`Dockerfile`, `docker-compose.yml`).
   - **The image:** Python 3.11 slim, CPU PyTorch 2.14, OpenCV without its GUI, the project's
     code and default settings.
   - **The services:**
     - `secrets` (once): makes the database and dashboard passwords into a volume that only the
       services mount;
     - `db`: PostgreSQL 16, not reachable from outside Docker;
     - `db-init` (once): creates the tables;
     - `cameras`, `alerts`, `dashboard`: each with a healthcheck;
     - `demo-rtsp` and `demo-cameras`: MediaMTX, and the simulated cameras streamed into it.
   - **Secrets:** `configs/secrets.env` is passed to every service if it exists. Values there win
     over the generated passwords.
   - **The dashboard** is published on 127.0.0.1:8080 only and always needs its password.
   - The project's `models/`, `data/` and `runs/` are mounted, so both setups share them.
   - **A GPU override** (`docker-compose.gpu.yml`) builds CUDA PyTorch and hands an NVIDIA GPU to
     the camera service. It is untested: no NVIDIA GPU was available.
   - **The check** (`scripts/check_compose.py`) runs the stack as a separate Compose project, with
     a stand-in Telegram, then checks through the dashboard's API:
     - every service healthy;
     - the password demanded;
     - the three simulated cameras live;
     - violations stored as events;
     - alerts delivered with the picture;
     - the evidence image served.

## Consequences

- **The five numbers are measured, and two miss their targets** ([results](../phase6_results.md)).
  - Met: mAP@50 0.883; 8 cameras at 9.9 frames/s with the database and alerts on; 4.0 s from a
    violation's start to the alert leaving the Mac.
  - Missed: recall is 83 % (target 90 %). Precision is 89 %, just under 90 %.
  - Alerts on the phone take 7.1 s with the picture on this Mac's ~1 Mbit/s upload, and 5.0 s
    without it. The 5.0 s was measured with the connection kept open, where 4.4 s had been
    expected. The likely reason: the worker waits for every alert it is sending before it takes
    new ones off the queue.
  - The misses and false alarms were all looked at. Half the false alarms are bicycle helmets in
    one public photo. The missed helmets are mostly close-ups and people seen from behind in the
    dark. The missed zone intrusions are the same 9 as in Phase 3: small, grouped or cut-off
    people.
- **The test set, not the system, limits what can be said.**
  - 258 of the 260 clips are made from photos, and the intervals are wide (recall 75–89 %).
  - The false-alarm rate per camera-hour on a real site is unknown until fixed-camera footage
    from the site is annotated. `annotate_clip.py` and the `site` set are ready for it.
- **Real video from six other sites** (Pexels, fixed cameras, annotated by hand, test only; 28
  Sep). The whole system, 8 cameras at 10 frames/s:
  - 4 of 6 violations caught; 4 of 5 alerts real. Far too few to judge by (recall 30–90 %).
  - The mistakes are of kinds the photo clips hardly had: a baseball cap taken for a helmet, a
    worker 90 pixels tall, a tractor taken for a person. The caps were fixed on 29 Sep by a
    fine-tuning with photos of people in hats (decision 0010).
  - So the photo clips' numbers can't simply be carried over to a site: the kinds of mistake
    change with the scenes.
  - The feedback loop's instructions now cover a false alarm on something that isn't a person
    at all: delete that person label rather than draw a helmet.
- **The retraining loop is ready but idle.** It needs at least 10 false alarms from footage that
  isn't test footage. The dry run on test footage showed why a person must review the automatic
  labels: the first version took a bystander's helmet for the person's own.
- **Two ways to run the system:**
  - natively on the Mac (Core ML, 8 cameras): `mac_phase5.sh run`;
  - in Docker on a Linux server (CPU by default).
  They share `configs/`, `models/` and `data/`. The Compose check needs Docker on the machine
  running it; it passed in the cloud from a clean copy.
- **Containers run as root.** That avoids permission problems with the project folders mounted
  into them. Only the dashboard is published, on 127.0.0.1, with a password.
- **Evaluation runs are kept** in `runs/evaluation/<time>/`: the report, `metrics.json`, every
  alert's verdict, the evidence images, and what's needed to score the run again
  (`--rescore`) after a change to the matching.
