# 0005 — Restricted zones and a rules engine driven by configuration

**Status:** accepted (Phase 3 review) · **Book:** Chapters 10 and 20

## Context

Phase 3 adds the second kind of violation: a person inside an area they must stay out of. The
book asks for:

- zones drawn per camera and stored as fractions of the frame, not pixels;
- an intrusion test on each tracked person's feet, with a dwell time;
- rules that are configuration, not code: type, camera, zone, severity, dwell, cooldown, active hours;
- PPE events and zone events in the same shape.

## Decision

1. **Zones** live in `configs/zones.yaml`, per camera.
   - Corners are 0–1 fractions of the frame's width and height.
   - The frame each set of zones was drawn on is kept in `configs/zones/<camera>.jpg`, so a moved
     camera can be spotted later.
   - Zones are checked on load: at least 3 corners, all within 0–1 (a message says "not pixels" if
     they aren't), edges that don't cross, some area.
2. **Where a person stands:**
   - the average of their visible ankle keypoints when the keypoint model found any;
   - otherwise the bottom-centre of their box.
   - If the box reaches the bottom of the frame and no ankle is visible, the answer is
     "can't tell". It never raises an alarm and never clears one.
   - Ankles come first because they stay right when the box is wrong, for example an outstretched
     arm or two people merged into one box.
3. **Point-in-polygon** is our own ray casting (ten lines, tested), not Shapely. That keeps one
   less dependency on the Mac, and it handles concave zones.
4. **Rules** live in `configs/rules.yaml` and are checked on load, with messages that say what to
   fix. There are three types:
   - `no_helmet` and `no_vest`. With a `zone:`, a PPE rule applies only inside that zone, e.g. "helmets
     on the scaffold"; outside it the person counts as compliant.
   - `zone_intrusion`: standing inside the zone is the violation.
   - Each rule also has cameras, severity, dwell, cooldown and optional active hours. Hours can
     be several ranges, can run past midnight, and can be limited to certain days. They use the
     machine's local time unless a `timezone:` is set.
5. **One event engine for every rule.** The Phase 2 state machine (vote, dwell, one event per
   episode, cooldown, incidents that survive ID changes) now runs once per rule, with that rule's
   dwell and cooldown.
   - A rule outside its active hours counts nothing.
   - Anyone already being confirmed starts again from zero when the rule comes back on.
   - A rule that switches off during someone's dwell raises nothing.
6. **One event shape:**
   - kind, rule, camera, zone, severity, track ID;
   - start and confirmation time on the stream's clock, and the wall-clock time;
   - the person's box, the share of violating votes, the frame index and the snapshot file.
   - `scripts/monitor.py` writes exactly this to `events.jsonl`.
7. **No rules file = the Phase 2 behaviour:** no helmet and no vest, on every camera.

## Consequences

- Adding a camera's zone and rule is an edit of two YAML files plus a check
  (`scripts/check_rules.py`), with no code change.
- **The zone test uses feet, which the camera must see.** A camera that only sees people from
  the waist up (feet cut off by the frame edge) can't enforce a zone. Its zones stay silent and
  the check says "can't tell".
- **Zones are flat polygons on the image.** Drawn on the ground they work for people standing on
  that ground. A zone for a raised platform or a stair would need its own rule type.
- **A person standing on the zone's edge** flickers in and out; the vote needs more than 60%
  "inside" before it counts, so a person half on the line doesn't alert. A margin setting can be
  added if your cameras need one.
- **Active hours use wall-clock time.** For recorded video, `monitor.py --start-time` says when
  it was recorded.
