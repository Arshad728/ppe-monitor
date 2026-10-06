# 0004 — Walking, bending and crouching workers: body keypoints and motion-aware tracking

**Status:** accepted (Phase 2 review, 2026-09-24); the thresholds are still to be checked on your clips · **Book:** Chapters 9, 10, 19

## Context

The first Phase 2 build assumed people stand upright and move slowly between frames:

- **The helmet rule looks for the helmet at the top of the person box.** A worker who bends over
  to pick something up has their head halfway down the box or at one side. The worn helmet then
  falls outside the "head region", every frame says "no helmet", and a few seconds of bending is
  enough for a false alert.
- **The tracker compares boxes by overlap.** A walking person's boxes barely overlap from one
  processed frame to the next when frames are skipped (a slow machine, several cameras). Each frame
  then starts a new track, and each new track can start a new alert.
- **Crouching halves the box height**, which the "same person?" checks treated as a different person.

## Decision

1. **Body keypoints.** A second small model, YOLO26n-pose (COCO-pretrained, 17 keypoints, nothing to
   train), runs on every frame that has a person in it. From the keypoints: where the head is, the
   neck length (the unit for distances), and the posture: upright, bending, crouching or lying.
   It can be switched off in `configs/ppe.yaml` (`pose: enabled: false`), and the box rule is used alone.
2. **Helmet rule.** A helmet also counts as worn when its centre is within 1.25 neck-lengths of the
   head keypoints, and not more than 1.5 neck-lengths below the head along the body. That keeps a
   helmet carried at the waist from counting.
3. **Bent over, head out of sight.** A bent, crouching or lying person with no helmet found is
   *unknown*, not *missing*, only when no nose, eye or ear keypoint is visible (their back is to a
   high camera). When the head can be seen, a missing helmet is missing in any posture, so a worker
   who stays bent over without a helmet is still caught.
4. **Tracking people who move.**
   - Boxes are widened before they are compared, in proportion to the time between processed
     frames ("buffered IoU"): nothing at 15 fps, about half a box width at 3 fps.
   - A new track takes back the ID of a track lost nearby. "Nearby" is measured from where that
     person would be now, at the speed they had. It grows with the time the track was lost, at
     walking pace (0.5 + 0.8 person-heights per second).
   - The height may change 0.5–2× between sightings (bending, crouching, standing up).
   - ByteTrack sometimes revives an old lost track on someone else. If that track's ID is already
     in use, it goes through the same re-link check instead of getting a fresh ID.
5. **Events follow a moving person.**
   - An incident follows its person with the same widened-box test. A jump onto a neighbour never
     passes that test between consecutive frames.
   - If its person goes out of sight and a new track appears in their place, that track takes the
     incident over. A walker who gets a new track ID then joins their existing incident instead of
     raising a second alert.
   - When frames are skipped, the 1-second vote window stretches, up to 3 s, so it still holds
     3 votes.

## Evidence (details in `docs/phase2_results.md`)

| Labelled training + validation images, labelled boxes | box rule | with keypoints |
|---|---|---|
| Worn helmets found: people bending over (51) | 92% | **98%** |
| … crouching (49) | 98% | 98% |
| … lying (11) | 0% | 55% |
| … upright (2,628) | 99.7% | 99.8% |
| Bare-headed people bending or crouching (17) still judged "missing" | 17 | **17** |

On the 86 test clips with everyone moving at walking pace (15 fps): 42 of 47 people without a
helmet and 55 of 68 without a vest got exactly one event; nobody got two; 5 and 6 false events on
119 and 81 wearers. Before the last two tracking changes, 15 people got two alerts.

`tests/test_motion.py` covers each of these scenarios with hand-made detections:

- walking with and without a helmet, at 15 fps and 3 fps;
- a group walking shoulder to shoulder, at 15 fps and 5 fps;
- walking behind a pillar;
- two workers crossing;
- bending with the helmet on;
- bending away from the camera;
- staying bent over without a helmet;
- crouching.

## Consequences

- **More work per frame.** The keypoint model is a second network on each frame with people in
  it. It roughly doubles the model time; Phase 4 can run it less often if needed.
- **Unknown while bent away.** A worker without a helmet who is bent away from the camera is not
  flagged until their head is in view again.
- **The public data barely covers these postures.** People bending, crouching or lying are about 3%
  of it, and the test split has no bare-headed ones. The thresholds (1.25, −1.5, the posture
  angles) must be checked on your clips.
- **Crowds at a low frame rate.** At 5 fps, people walking side by side swap IDs, and a few get two
  alerts. Crowded cameras need 10 fps or more, or appearance-based re-identification (Phase 4).
- **The moving test clips slide still photos sideways.** Nobody turns, bends or steps behind anything
  there. Real walking needs real clips.
