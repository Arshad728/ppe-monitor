# Phase 3: restricted zones and the rules engine

**Book:** Chapter 20, using Chapter 10 · **Model:** `ppe3_yolo26n_baseline` (Phase 1) ·
**Settings:** `configs/rules.yaml`, `configs/zones.yaml`, `configs/ppe.yaml` ·
**Why it is built this way:** [decision 0005](decisions/0005-zones-and-rules-engine.md)

**Reproduce everything below:** `bash scripts/mac_phase3.sh check`.

## What was built

```
frame ─► detector ─► tracker ─► PPE verdict ──┐
    └──► keypoints ─► where each person stands ┴─► rules of this camera ─► event engine ─► events
                                                  (configs/rules.yaml:        (vote, dwell,     one shape,
                                                   type, zone, severity,       cooldown,         PPE or zone
                                                   dwell, hours)               incidents)
```

| Part | File | What it does |
|---|---|---|
| **Zones** | `rules/zones.py`, `configs/zones.yaml` | Polygons per camera, stored as fractions (0–1) of the frame. They are checked on load: 3+ corners, all within 0–1, edges that don't cross. Point-in-polygon is ray casting. |
| **Where a person stands** | `rules/zones.py` | The visible ankle keypoints, or else the bottom-centre of the box. If the box reaches the bottom of the frame and no ankle shows, it is "can't tell", which never alerts. |
| **Rules engine** | `rules/engine.py`, `configs/rules.yaml` | Rules are data: type (`no_helmet`, `no_vest`, `zone_intrusion`), cameras, zone, severity, dwell, cooldown, active hours. A PPE rule with a zone applies only inside it. Every rule gets one vote per person per frame. |
| **Event engine** | `rules/events.py` | The Phase 2 state machine, now once per rule, with each rule's dwell and cooldown. A rule outside its hours counts nothing, and whoever was being confirmed starts again from zero. |
| **Drawing tool** | `scripts/draw_zones.py` | Click the corners on a frame from the camera, press Enter, and type the zone's id and name in the Terminal. It saves the corners and the frame. |
| **Rules check** | `scripts/check_rules.py` | Validates both files, lists what applies to each camera and whether it is on now, and draws each camera's zones into `runs/zones/`. |
| **Monitor** | `scripts/monitor.py` | Draws the zones and each person's standing point. `--as-camera` applies a camera's zones to a video file; `--start-time` says when a recording was made, so active hours work. |

### Adding a zone to a camera

1. `python scripts/draw_zones.py --camera cam3`: click round the area **on the ground**, press
   Enter, then type an id (`pit`) and a name (`Excavation edge`). Press `s` to save and `q` to quit.
2. Add a rule to `configs/rules.yaml`:

   ```yaml
   - id: pit-working-hours
     type: zone_intrusion
     cameras: [cam3]
     zone: pit
     severity: critical
     dwell: 2.0
     active: {hours: "07:00-19:00", days: mon-sat}
   ```
3. `python scripts/check_rules.py`: it says what's wrong if anything is, and shows the zone in
   `runs/zones/cam3.jpg`.

No code changes. The three simulated cameras come with a zone each, one per kind of schedule:

| Camera | Zone | Rule | On |
|---|---|---|---|
| cam1 | Hatched area, main gate | nobody inside, high | always |
| cam2 | Loading bay | nobody inside, medium | 19:00–07:00 (past midnight) |
| cam3 | Excavation edge | nobody inside, critical | 07:00–19:00, Monday–Saturday |

### One event shape (the book's "done when")

PPE and zone events are the same object, and `events.jsonl` stores exactly its fields. Two events
from the simulated gate camera:

```json
{"kind": "no_helmet", "camera": "cam1", "track_id": 3, "started": 4.2, "confirmed": 7.667,
 "box": [0.3045, 0.4008, 0.3777, 0.6728], "share": 1.0, "frame_index": 115,
 "event_id": "cam1-no_helmet-3-7666", "rule": "no_helmet", "zone": null, "severity": "high",
 "time": "2026-09-24T14:57:02+00:00", "snapshot": "snapshots/cam1-no_helmet-3-7666.jpg"}
{"kind": "zone_intrusion", "camera": "cam1", "track_id": 1, "started": 8.6, "confirmed": 10.6,
 "box": [0.2479, 0.5703, 0.3277, 0.9291], "share": 1.0, "frame_index": 159,
 "event_id": "cam1-gate-keep-out-1-10600", "rule": "gate-keep-out", "zone": "keep-out", "severity": "high",
 "time": "2026-09-24T14:57:05+00:00", "snapshot": "snapshots/cam1-gate-keep-out-1-10600.jpg"}
```

- `started` and `confirmed` are seconds on the stream's clock.
- `time` is the wall-clock time of confirmation.
- `frame_index` is the frame the snapshot shows.

## How well it works

### 1. The rules engine, with made-up people (no video)

The book asks for the engine to be tested with hand-written tracks. `tests/test_rules_engine.py`
(28 tests) and `tests/test_zones.py` (22 tests) do that, through the whole chain from tracker to
events:

| Scenario | Expected and got |
|---|---|
| Walks into the zone and stays | one event, 2 s after entering (plus the 0.5 s vote) |
| Crosses a corner of the zone in 1 s | no event |
| Stands on the edge, feet flickering in and out | no event |
| Feet out of view (box at the bottom of the frame) inside the zone | no zone event |
| The same person on another camera | no event there |
| Leaves and comes back, rule with a 10 s cooldown | a second event only after the cooldown |
| Inside at 19:00:30, rule on 07:00–19:00 | no event |
| Inside from 18:59, same rule | one event, stamped 18:59:0x |
| Already inside when the rule switches on at 07:00 | one event, counted from 07:00 |
| Walks in at 18:59:59 (the rule goes off before the dwell ends) | no event |
| No helmet, rule "helmet required in the pit": outside the pit, then inside | no event outside; one inside |
| No helmet, inside a no-entry zone | two events (helmet, zone), same fields |
| Hidden for 0.5 s inside the zone, back under a new track ID | still one event |
| Hours "22:00–06:00", "06:00–12:00, 13:00–18:00", "00:00–24:00", Mon–Sat | on and off at the right moments |
| Twelve kinds of mistake in the rules file, and three in the zones file | refused with a message that says what to fix |
| A concave zone, a ray through a corner, a bow-tie polygon, pixel coordinates | inside/outside right; the last two refused |

All 165 tests pass (Phase 0–3), plus one that only runs on the Mac.

### 2. People walking into a zone, on test clips

`scripts/make_event_clips.py --motion pan` makes 86 clips from the same test photos as Phase 2.
- **Movement:** the frame is 1.4× as wide as the photo, and the photo slides steadily from the
  left edge to the right over 10 s.
- **Zone:** it covers the right part of the frame. People on the right of the photo walk into it at
  different times and stay; people on the left never reach it.
- **Occluder:** the same dark bar sweeps across from 3 s to 7 s.
- **Truth:** each person's labelled box gives where their feet are in every frame.
- **Rule:** nobody inside for 2 s.

| 86 clips, 195 people (cloud CPU) | people |
|---|---|
| **Walked into the zone and stayed long enough to be confirmed** | 37 |
| … exactly one event | **28 (76%)** |
| … more than one event | **0** |
| … no event | 9 |
| **Never inside** | 88 |
| … false zone events | **0** |

- **Time from the feet crossing into the zone to the event:** median 2.7 s, 90% within 3.4 s. That
  is the 2 s dwell plus the half-second vote, plus a little for tracks that start late.
- **Not scored:** 70 people, who entered too late in the clip for a 2 s dwell (after about 6.5 s),
  or whose feet were out of view. Also not scored: 3 zone events on people nobody labelled.
- **The PPE rules ran at the same time:** 40 of 47 people without a helmet and 56 of 68 without a
  vest were caught exactly once, with no duplicates. These clips have both PPE and zone rules on
  every person.

**The 9 misses, looked at one by one:**

| Cause | Count |
|---|---|
| A small, distant person the detector never finds | 4 |
| Two or three people close together, boxed as one: the standing point falls between them, or the event goes to the neighbour | 3 |
| Feet at the very bottom of the frame: the detector's box touches the edge, so "can't tell" (by design) | 2 |

**None of the misses are the zone logic itself.** Four are the detector's known weakness on small
people (Phase 1). Three are merged boxes in groups (Phase 2). Two are the deliberate rule that
feet cut off by the frame edge can't be judged. A real camera for zone rules should see the whole
zone, with some margin below it.

### 3. The simulated cameras (Phase 0's cartoon clips)

In the Phase 0 clip, worker C walks into the hatched area at about 8 s. The detector finds the
cartoon people well enough to track them. It does not find their cartoon helmets and vests
reliably, so the PPE events on these clips mean nothing.

| Run | Zone events |
|---|---|
| cam1, keep-out zone, always on | 1: worker C, began 8.6 s, confirmed 10.6 s |
| cam3, pit zone (07:00–19:00), played as if at noon on a Monday | 1: worker C, confirmed 10.5 s |
| cam3, played as if starting at 18:59:52 (the rule goes off at 19:00, as worker C arrives) | 0 |

### 4. Phase 2 is unchanged

The event engine now runs per rule, and PPE events carry more fields. Re-scoring the Phase 2
clips with the same saved detections gives exactly the Phase 2 results:
- still clips: 25/28 and 35/45;
- moving clips at 15 fps: 42/47 and 55/68;
- moving clips at 5 fps: 38/47 and 48/68;
- the same duplicates and false events.

### 5. On the Mac (2026-09-24)

`mac_phase3.sh check` on the MacBook Air M4 (Apple GPU):
- **Tests:** all 166 pass (165 plus the Mac-only GPU test).
- **Rules check:** no problems. At 21:19 it correctly showed the loading-bay rule on (19:00–07:00)
  and the pit rule off (07:00–19:00).
- **Simulated cameras:** the same as in the cloud: one zone event on cam1 (confirmed 10.6 s); one
  on cam3 played as noon; none on cam3 played from 18:59:52.

| Zone clips | cloud CPU | Mac |
|---|---|---|
| Walked in and stayed: exactly one zone event | 28 / 37 | 27 / 37 |
| Two zone events for one person | 0 | 0 |
| False zone events (88 people never inside) | 0 | 0 |
| Feet entering → event, median / 90% | 2.7 s / 3.4 s | 2.6 s / 3.3 s |
| No helmet: exactly one event | 40 / 47 | 39 / 47 |
| No vest: exactly one event | 56 / 68 | 56 / 68 |
| False PPE events (helmet / vest) | 5 / 7 | 5 / 6 |

- **The Mac's one extra zone miss** is a person in clip 037. The clips are made on the Mac, and the
  GPU's detections differ slightly from the cloud's, as in Phase 2. The other 9 misses are the same
  people.
- **Speed:** 27 ms per frame for the whole chain with zones and rules, the same as Phase 2. The
  zones and rules cost nothing measurable.
- **The live demo window** ran at 54 ms per frame (about 18 fps), because drawing and showing the
  picture take time too. Phase 4 will keep the display separate from the processing.

## What Phase 3 can't know yet (your clips will tell)

1. **Where your zones are, and whether the cameras see people's feet there.** A zone test needs
   the feet in view; a camera that cuts people off at the knees can't enforce a zone.
2. **Perspective.** A zone drawn on the ground works for people standing on that ground. Someone on
   a ladder or a platform above the zone has their feet somewhere else in the image.
3. **People on the zone's edge.** The vote ignores feet flickering across the line. If your sites
   need a margin ("within 1 m of the edge"), it can be added as a zone setting.
4. **Small people and groups.** The same limits as Phase 2 apply, since a zone can only judge people
   the detector finds and separates.

## Try it

```bash
bash ~/Documents/ppe-monitor/scripts/mac_phase3.sh demo         # the gate camera: worker C walks into the zone
bash ~/Documents/ppe-monitor/scripts/mac_phase3.sh draw cam1    # try the drawing tool (q to quit without saving)
conda activate ppe && cd ~/Documents/ppe-monitor
python scripts/check_rules.py                                   # what applies where, and zone pictures
python scripts/check_rules.py --at "2026-09-27 21:00"           # ... on a Sunday night
python scripts/monitor.py --source clip.mp4 --as-camera cam3 --start-time "2026-09-24 18:59:30"
```

- **Zone colours:** red outline when a rule is on, grey with "(off now)" outside its hours, red tint
  while someone stands inside.
- **People:** the ring at each person's feet is the point tested against the zones, and the label
  says "in <zone>" when it is inside one.
