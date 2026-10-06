# Phase 2: tracking people and deciding who wears what

**Book:** Chapter 19, using Chapters 9 and 10 · **Model:** `ppe3_yolo26n_baseline` (Phase 1) ·
**Settings:** `configs/ppe.yaml`

**Reproduce everything below:** `bash scripts/mac_phase2.sh check`.

**Update after the first review:** workers who walk, bend over or crouch are now handled (body
keypoints, a tracker that expects people to move, events that follow them). What changed and how it
was checked is in [Walking, bending and crouching](#walking-bending-and-crouching); the reasons are
in [decision 0004](decisions/0004-posture-aware-ppe-rule.md).

## What was built

```
frame ─► detector ─► person tracker ─► PPE rule per person ─► event engine ─► event + snapshot
    │    (Phase 1)   (ByteTrack,       (helmet on the head?    (vote, dwell,     events.jsonl
    │                 person IDs)       vest on the torso?)     cooldown)        snapshots/*.jpg
    └──► body keypoints ───────────────► (where the head is,
         (YOLO26n-pose)                   posture)
```

| Part | File | What it does |
|---|---|---|
| **PPE rule** | `rules/ppe.py` | Decides per person, per frame, whether each item is **worn**, **missing** or **unknown**. A helmet counts if its centre is in the head region: from 10% of the person's height above the box down to 35% into it. A vest counts if its centre is in the torso region (15–85%) and at least half of it lies inside the person box. Each helmet and vest goes to at most one person, closest match first. The answer is *unknown* when the person is shorter than 64 px at the detector's input (48 px for the vest) or the box touches the frame edge, and unknown never raises an alarm. |
| **Keypoints** | `vision/pose.py` | A second small model (YOLO26n-pose, COCO-pretrained, nothing to train) finds 17 body points per person: where the head really is, and the posture (upright / bending / crouching / lying). The PPE rule uses them to find the helmet on a head that is not at the top of the box. |
| **Tracker** | `vision/tracking.py` | ByteTrack on person boxes only. It has repairs found on the test clips below, and for people who move; see "Tracking" and "Walking, bending and crouching" further down. |
| **Event engine** | `rules/events.py` | Decides when a violation becomes an alert. See the four steps below. |
| **Chain** | `pipeline.py` | Runs the steps above in order for each frame. Used by `scripts/monitor.py` (live camera or video file) and `scripts/evaluate_events.py` (scoring). |

How the event engine turns per-frame answers into one alert:

1. **Vote.** Over the last 1 s, more than 60% "missing" votes means the person is violating now.
2. **Dwell.** The violation must last 3 s before it becomes an event.
3. **One event per episode.** No new event until the person has been compliant for 2 s, and not
   within 60 s of the last one.
4. **Same place, same incident.** Each event remembers where its incident is, and follows the person
   as they move. A person who reappears there under a new track ID joins the incident instead of
   starting a new alert.

## How well each part works

### 1. The geometry, on labelled boxes

The datasets also label absences: "no helmet" on bare heads, "none" / "no vest" on torsos. Those
labels give 346 people in the test split whose status is known. The rule was run on their labelled
boxes (`scripts/check_ppe_rules.py`).

| Test split, labelled boxes | helmet | vest |
|---|---|---|
| People wearing it / not wearing it | 271 / 52 | 210 / 90 |
| Wearing it, but judged missing | 0.7% | 0.0% |
| Not wearing it, but judged worn | 0.0% | 0.0% |

**The geometry is sound.** The book's tighter regions (top fifth, middle third) were worse on
validation: 8.8% and 13.6% of wearers judged missing. The regions used here were set from where
5,900 labelled helmets and vests sit in the training and validation images.

### 2. One frame of the detector

The same people, judged from the detector's boxes in a single image. That is roughly what one video
frame gives:

| Test split, one frame | helmet | vest |
|---|---|---|
| Wearing it, but judged missing (**pushes towards a false alert**) | 4.5% (5.3% without keypoints) | 9.5% |
| Not wearing it, but judged worn (**hides a violation that frame**) | 2.3% | 8.6% |

The keypoints (see [Walking, bending and crouching](#walking-bending-and-crouching)) find a few more
worn helmets whose centre falls just outside the head region of an imperfect person box. The vest
rule doesn't use them.

**Size matters.** For people 128–256 px tall at the detector's input, 19% of helmet wearers and 22%
of vest wearers were judged missing (27 and 23 people). For people over 256 px it was 3.0% and 8.0%.
CCTV mostly sees the smaller sizes. The public images have almost no one below 128 px, so how the
rule does at typical CCTV sizes has to be measured on your clips.

**Thresholds.** The helmet and vest thresholds (0.25 each) were chosen on the validation split. They
lowered false "missing" answers compared with 0.35: helmet 7.8% → 5.2%, vest 7.0% → 4.1%. The cost was
more vests "seen" on people without one: 2.0% → 3.9%.

### 3. The event logic, with simulated detector errors

Real detector errors are not independent from frame to frame. A worker who turns away hides their
vest for a second or two. So errors were simulated in bursts, and the real event engine was run over
them: `scripts/simulate_events.py`, 1 simulated hour per row. The rows use the test-split vest error
rate, about 10% of frames.

| Errors come in bursts of | Dwell 2 s: false alerts / hour / worker | Dwell 3 s | Dwell 5 s | Reported after (dwell 3 s, median / 90%) |
|---|---|---|---|---|
| single frames | 0 | 0 | 0 | 3.6 s / 3.7 s |
| 0.5 s | 6 | 0 | 0 | 3.6 s / 6.2 s |
| 1 s | 25 | 12 | 0 | 3.6 s / 5.6 s |
| 2 s | 32 | 20 | 9 | 3.6 s / 5.2 s |

**What this means:**
- **Independent flicker is fully removed**, and so are bursts up to half a second.
- **Longer bursts get through.** A longer dwell stops them, at the cost of a slower alert.
- **Dwell is set to 3 s**, the top of the book's 2–3 s range. How long real misses last is the
  number your clips must provide. If false alerts turn out too frequent, raise `dwell` in
  `configs/ppe.yaml`.

With a pessimistic 25% error rate, dwell 3 s gives 0 false alerts per hour for single-frame errors,
4 for 0.5 s bursts and 27 for 1 s bursts.

### 4. The whole chain, on test clips with known answers

There are no real site videos yet, so `scripts/make_event_clips.py` turns 86 test photos into
10-second clips. 43 photos have a person without a helmet or vest; 43 have only compliant people.
Each clip adds:
- a slow pan and zoom, so detections flicker from frame to frame;
- camera noise and video compression;
- a dark bar sweeping across the frame from 3 s to 7 s, like someone walking past close to the
  camera, which briefly hides everyone.

The people don't move, so these clips test flicker and occlusion, not walking or putting PPE on and
off. Moving people have their own clips: see [Walking, bending and crouching](#walking-bending-and-crouching).

| 86 clips, 176 people | helmet | vest |
|---|---|---|
| People without it: each should get exactly one event | 28 | 45 |
| … exactly one event | **25 (89%)** | **35 (78%)** |
| … more than one event | **0** | **0** |
| … no event | 3 | 10 |
| People wearing it: should get none | 119 | 81 |
| … false events | 4 | 6 |

Median time to the event: 3.5 s. With keypoints on or off, these numbers are the same.

**Momentary misdetections produce no events** (the book's "done when"). Of the 195 compliant
person-items with a judgement:
- 45 were judged "missing" in some frames, up to half of them, and **none** got an event;
- every false event came from a person the detector got wrong in **more than half** of all frames
  (10 of 14 such people). That is a detector problem, which smoothing can't and shouldn't hide.

**All 23 wrong outcomes were looked at one by one:**

| Cause | Count | Examples |
|---|---|---|
| Person not found as a separate person (crowd, umbrellas, a classroom) | 7 | people under umbrellas merged into one box |
| **Real detector misses** | 4 | reflective strap vests at night (2), a red vest, a bending worker's vest |
| The test label disagrees with our definitions | 5 | cycling helmets and a plain yellow jacket labelled as PPE (the chain rightly says no), a harness labelled as a vest |
| The test label is wrong | 3 | a worker in a hi-vis vest labelled "no vest"; labels drawn around groups of people |
| Look-alike | 1 | an orange top taken for a vest about half the time |
| Orange workwear the detector is unsure about | 1 | judged vest 40% of the time |
| Person lying sideways (rotated photo) | 1 | the helmet is found in only a few frames, and the keypoint model finds no body to place the head |
| Head cut off by the frame edge, so "unknown" | 1 | working as designed |

So about 14 of the 23 are real limitations of the chain, 8 are label or definition problems, and 1 is
working as designed.

### Tracking

On the clips, a sweeping bar hides every person for about half a second.

**Plain ByteTrack failed here:**
- it gave people new IDs, and sometimes handed an ID to a neighbour;
- one person detected twice (full body plus upper body) was followed as two people.

**Three fixes:**
1. **Merge duplicate person boxes.** A clearly smaller box, mostly inside another, with the head in
   the same place, is the same person.
2. **Re-link.** A new track that starts within half a person-height of a track lost in the last 2 s
   takes its ID back.
3. **Slow down lost tracks.** A lost track's speed is damped each frame, so its predicted box does
   not coast onto a neighbour.

Then two changes on the event side:
- the association threshold was tightened to `match: 0.7`;
- events use **incidents** (step 4 above) instead of bare track IDs.

| On the 86 clips | Plain ByteTrack, events by track ID | First Phase 2 build | Now |
|---|---|---|---|
| ID switches (176 people, everyone briefly hidden) | 139 | 47 | 42 |
| People given more than one event for one violation | 15 | **0** | **0** |
| Share of frames where each person's track was found | 88% (mean) | 87% (mean) | 87% (mean) |

"Now" includes the changes for moving people described below.

**Tracking caveat:** the fixes and `match: 0.7` were developed while looking at these test clips.
Treat the tracking numbers as development results. The detector and the PPE thresholds were not tuned
on test data; they were tuned on the validation split only.

## Walking, bending and crouching

Added after the first review. The first build assumed people stand upright and barely move between
frames. Real workers walk, bend over to pick things up, and crouch to work near the ground.

### What changed

| Problem | Change | Where |
|---|---|---|
| A worker who bends over has their head halfway down the box, so a worn helmet falls outside the "head region". A few seconds of bending would raise a false alert. | **Body keypoints** (YOLO26n-pose, COCO-pretrained, nothing to train) say where the head really is. A helmet also counts when its centre is within 1.25 neck-lengths of the head keypoints (and not more than 1.5 below the head, so a helmet carried at the waist still doesn't count). | `vision/pose.py`, `rules/ppe.py` |
| From a high camera, someone bent over with their back to it shows no head at all. | Bent, crouching or lying, **head out of sight** (no nose, eye or ear keypoint) and no helmet found: **unknown**, judged once the head is back in view. If the head can be seen, a missing helmet is missing in any posture, so a worker who stays bent over without one is still caught. | `rules/ppe.py` |
| A walker's boxes barely overlap from one processed frame to the next when frames are skipped (a slow machine, many cameras), so each frame started a new track. | Boxes are **widened in proportion to the time between frames** before they are compared (almost nothing at 15 fps, about a quarter of a box width on each side at 5 fps). | `vision/tracking.py` |
| A walker hidden by a pillar reappears further on, under a new ID. | A new track takes back the ID of a track lost nearby, where "nearby" is measured from **where that person would be now** (at the speed they had), and grows at walking pace with the time they were lost. Heights may change 0.5–2× (bending, crouching). | `vision/tracking.py` |
| ByteTrack sometimes revived an old, lost track on a different person. | A revived track whose ID is already in use goes through the same re-link check instead of getting a fresh ID. | `vision/tracking.py` |
| An event's incident stayed where the person was when their track was lost; a walker who got a new ID was then far from it 3 s later and raised a second alert. | An incident **follows** its person with the same widened-box test, and a **new track that appears in a lost owner's place takes the incident over**. | `rules/events.py` |
| At a low frame rate, the 1 s vote window held too few votes to decide. | The window stretches, up to 3 s, until it holds 3 votes. | `rules/events.py` |

Why these and not others: [decision 0004](decisions/0004-posture-aware-ppe-rule.md).

### Bending, crouching, lying: the rule on labelled images

The public data has few people who aren't upright. This table uses the labelled training and
validation images (the rule's settings were set there; the test split has just 7 bending and 4
crouching people with a known status, all wearing helmets, and all judged correctly both ways):

| Training + validation, labelled boxes | box rule alone | with keypoints |
|---|---|---|
| Worn helmets found: bending over (51 people) | 92% | **98%** |
| … crouching (49) | 98% | 98% |
| … lying (11) | 0% | 55% |
| … upright (2,628) | 99.7% | 99.8% |
| Bare-headed people bending or crouching (17): judged "missing" | 17 | **17** |

`scripts/check_ppe_rules.py` prints the same split by posture for the detector's boxes.

### Scenarios with made-up detections

`tests/test_motion.py` and `tests/test_pose.py` run the whole chain (tracker → rule → events) on
detections and keypoints made by hand, so they pin down the logic independently of the detector. All
pass:

| Scenario | Expected and got |
|---|---|
| A worker without a helmet walks across the view, at 15 and 3 fps | exactly one event, one ID |
| The same worker with a helmet | no event |
| A worker without a helmet walks behind a pillar for 1 s (15 and 3 fps) | one ID, one event |
| Two workers cross, one without a helmet | one event, on the right one |
| Three workers walk shoulder to shoulder, the middle one without a helmet (15 and 5 fps) | one event, three IDs |
| Bends over for 6 s with the helmet on | no event (the box rule alone raises one) |
| Bends over away from the camera, helmet not detected | no event |
| Stays bent over without a helmet, head in view | one event, while still bent |
| Crouches for 3 s | same ID, no event |
| A helmet held at the waist | still "missing" |

### Moving people, on the test clips

`scripts/make_event_clips.py --motion sway` makes a second set of the 86 clips. The frame is 1.25×
as wide as the photo, and the photo slides from side to side in it, 10% of the frame width either way
every 3 s. Everyone moves across the frame at walking pace, back and forth, and stays in view. The
same dark bar sweeps across from 3 s to 7 s. Because everyone stays in view, more people can be
scored than in the still clips.

| 86 moving clips, 176 people | helmet, 15 fps | vest, 15 fps | helmet, 5 fps | vest, 5 fps |
|---|---|---|---|---|
| People without it: each should get exactly one event | 47 | 68 | 47 | 68 |
| … exactly one event | **42 (89%)** | **55 (81%)** | 38 (81%) | 48 (71%) |
| … more than one event | **0** | **0** | 1 | 4 |
| … no event | 5 | 13 | 8 | 16 |
| People wearing it: should get none | 119 | 81 | 119 | 81 |
| … false events | 5 | 6 | 5 | 3 |

"5 fps" processes only every 3rd frame, as a machine that can't keep up would.

- **At 15 fps, moving people do as well as still ones:** no duplicate alerts, and the same share
  caught (89% / 81% here, 89% / 78% on the still clips). Of the 13 wrong outcomes that only the moving
  set has, 11 involve people the still clips' zoom left mostly out of view. The other 2 are new false
  events on people in full view.
- **Before the last two changes** (revived tracks, incident takeover), the same moving clips gave 15
  people duplicate alerts at 15 fps. Re-linking revived tracks removed 10; the last 5 were each one
  large person split in two by the passing bar, which the incident takeover handles.
- **At 5 fps, crowds walking together swap IDs.** 473 ID switches, against 113 at 15 fps. Almost
  two thirds come from two clips with 7 and 11 people side by side, where each person moves half a
  box width between processed frames. Most alerts still come out right, because incidents follow
  people, but 5 people got two alerts (at 7.5 fps: 284 switches, 2 people). A camera with crowds
  should be processed at 10 fps or more.
  If Phase 4 can't manage that, the tracker needs appearance features (re-identification).
- **Keypoints change little here:** with them off, the moving clips at 15 fps give the same catches
  and one more false helmet event (6 instead of 5). On the still clips the results are identical with and
  without them. The photos are mostly of people standing.
- **Scoring change:** an event now counts for the person its track followed during the violation,
  not the person under its box at the moment of the alert. At that moment the box can sit on a
  half-hidden sliver of the person. Re-scored this way, the first Phase 2 build gives the same
  still-clip numbers as before.

### Cost

The keypoint model is a second network on each frame that has people in it. **On the Mac (Apple
GPU) the whole chain now takes 27–29 ms per frame, up from 13 ms**: one camera at 15 fps uses less
than half of the Mac's time. On the cloud CPU used for development it went from about 100 to about
180 ms. Phase 4 decides how to share the time between cameras.

### On the Mac (2026-09-24)

`mac_phase2.sh check` on the MacBook Air M4: all 115 tests pass, and the rule check matches the
cloud exactly. The clips are made on the Mac, so their video compression differs slightly, and so do
a few detections:

| | still clips | moving, 15 fps | moving, 5 fps |
|---|---|---|---|
| No helmet: exactly one event | 24/28 | 42/47 | 39/47 |
| No vest: exactly one event | 35/45 | 55/68 | 45/68 |
| People with more than one event | 0 | **0** | 8 (cloud: 5) |
| False events, helmet / vest | 4 / 6 | 5 / 5 | 5 / 5 |
| ID switches | 38 | 100 | 495 |

The one still-clip difference is a helmet missed on one person (the first Mac run had it too). At
5 fps the duplicates are no longer only in the two crowd clips: groups of three to five people
walking close together also get them. That confirms the advice above: 10 fps or more when people
walk in groups.

### What this doesn't show

- **The moving clips are still photos sliding sideways.** Everyone moves together; nobody turns,
  changes speed, passes behind anyone, or puts PPE on or off. That is gentler than a real site in
  some ways and harsher in others (the direction reverses every 1.5 s).
- **Postures are rare in the public data** (about 3% of people), so the bending and crouching
  numbers rest on about 100 people.
- **The keypoint model has never seen your cameras.** From high up, keypoints are less reliable.

Your clips should include people walking past each other, behind machinery, bending down and
crouching, with and without a helmet.

## Settings chosen (configs/ppe.yaml)

| Setting | Value | Why |
|---|---|---|
| helmet / vest threshold | 0.25 | fewer false "missing" per frame (see 2) |
| person threshold / new track | 0.40 / 0.50 | fewer tracks started on false people |
| head region | top −10% … 35% | covers ~98% of worn helmets; excludes a helmet carried at the waist |
| minimum person height | 64 px (helmet), 48 px (vest) | a helmet on a smaller person is under 10 px; **to be checked on your clips** |
| vote window / ratio | 1 s / 60% | |
| dwell | 3 s | book: 2–3 s; see the simulation in 3 |
| clear / cooldown | 2 s / 60 s | one alert per incident, at most one per person per minute |
| lost track kept | 2 s | |
| keypoints | on (`pose:`) | helmets on bent, crouching and lying people; posture |
| helmet near the head keypoints | 1.25 neck-lengths, ≤ 1.5 below | 98% of worn helmets on people bending over (box rule: 92%) |
| bent, head out of sight | unknown | judged once the head is back in view |
| re-link distance | 0.5 + 0.8 person-heights per second lost | a brisk walk is about 0.8 heights a second |
| box widening | 1.8 × seconds between frames − 0.1 | almost none at 15 fps, a quarter of a box at 5 fps |
| incident takeover | new tracks up to 1 s older than the owner's loss | the same person under a new ID |

## What Phase 2 can't know yet (your clips will tell)

1. **How long real detector misses last** (burst length). This decides the dwell time.
2. **How the rule does on small people.** CCTV people are mostly 64–256 px at the detector's input;
   the public data barely covers that.
3. **Real walking and real occluders.** The moving test clips slide a still photo sideways (see
   above).
4. **Bending, crouching or lying workers at your cameras' height.** The keypoint rule was checked
   on about 100 such people in the public data.
5. **Your camera's crowds.** Gates and queues; below 10 fps, crowds walking together swap IDs.

## Try it

```bash
bash ~/Documents/ppe-monitor/scripts/mac_phase2.sh demo      # a test clip, with the live overlay
bash ~/Documents/ppe-monitor/scripts/mac_phase2.sh demo datasets/event_clips_sway/005_violation.mp4   # people moving
conda activate ppe && cd ~/Documents/ppe-monitor
python scripts/monitor.py --source some_video.mp4             # any video; events -> runs/monitor/
python scripts/monitor.py --camera cam1                       # a (fake) camera
```

**Box colours:**
- green: helmet and vest worn;
- amber: a missing item is being confirmed;
- red: confirmed violation;
- grey: too small or cut off to judge.

A person who isn't upright gets their posture in the label (bending / crouching / lying), and a dot
marks where the keypoints put their head.

Each confirmed event is printed and saved, with a snapshot, to
`runs/monitor/<camera>_<time>/events.jsonl`.
