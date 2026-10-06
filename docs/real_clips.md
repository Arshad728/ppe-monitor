# Real clips: first check

**25 Sep 2026** · Two clips recorded by the user in a fabrication workshop · **Model:**
`ppe3_yolo26n_baseline` (Phase 1) · **Chain:** the full Phase 3 chain (detector, keypoints,
tracker, PPE rule, events), with the default rules (no helmet and no vest, on every camera).

> **Update, same day: after fine-tuning.** A model fine-tuned with three more public datasets
> finds a helmet on the head in 96 % of the navy-helmet frames (was 41 %), 91 % of the brown and
> 91 % of the white (both 0 %), and still almost never on a bare head (3 %). The navy-helmet false
> alarm below is gone; one new false helmet alert remains, in heavy haze. See
> [After fine-tuning](#after-fine-tuning) and [finetune_results.md](finetune_results.md).

These are **test clips**: they measure the system, and are never used to train it.

## The clips

| Clip | Length | Frame | Camera | What's in it |
|---|---|---|---|---|
| `workshop_handheld_day_01` | 14.7 s | 478 × 850, portrait | phone, hand-held, eye level, moving | Workers with navy and yellow helmets, mostly seen from behind against bright skylights. Two men without helmets in the last 2–3 s. |
| `workshop_handheld_day_02` | 12.5 s | 720 × 1280, portrait | the same | Workers with yellow, navy, white and brown helmets, in haze. |

- Both clips were converted to H.264 at 15 frames a second, with no sound, the way a CCTV
  stream arrives. The originals are unchanged in `data/clips/originals/`.
- Nobody in either clip wears hi-vis.

## What the system raised

| Clip | Alert, and when | Person | Right? |
|---|---|---|---|
| 01 | no vest, 3.5 s | navy helmet, grey shirt | yes, by the rules (nobody here wears hi-vis) |
| 01 | no vest, 4.4 s | navy helmet, dark shirt | yes, by the rules |
| 02 | no vest, 6.9 s | yellow helmet, check shirt | yes, by the rules |
| 02 | no vest, 7.9 s | navy helmet, dark clothes | yes, by the rules |
| 02 | **no helmet, 9.0 s** | **navy helmet, dark clothes** | **no: a false alarm** |

- **The one helmet alert is false.** The man wears a navy helmet from start to finish
  (`false_alarm_track4_zoom.jpg`), but he was judged as wearing one in only 9 of his 87 frames.
- **A second false helmet alert came close.** Another man with a navy helmet (clip 01, #3) was
  judged "no helmet" in 42 of 60 frames, but never for long enough in a row to raise an alert.
- **The two men who really had no helmet** (clip 01, 12–15 s) were judged "no helmet" in 31 of
  32 frames. They were in view for only about 2 s, under the 3 s an alert needs, so there was
  rightly no alert.

## Helmets, by colour

I checked every tracked person by eye, by cropping their head in each frame. The table covers
every frame of each person I checked.

| Truth | People | Frames | Judged "worn" | A helmet found on the head, score ≥ 0.25 (the setting in use) | … score ≥ 0.10 |
|---|---|---|---|---|---|
| yellow helmet | 2 | 200 | 98 % | 98 % | 100 % |
| navy helmet | 5 | 251 | **39 %** | 41 % | 77 % |
| white helmet, in haze | 1 | 11 | **0 %** | 0 % | 0 % |
| brown helmet | 1 | 23 | **0 %** | 0 % | 0 % |
| no helmet | 2 | 32 | 3 % | 3 % | 31 % |

- **Yellow helmets are found almost every time. Navy, brown, and white in haze are mostly missed.**
- **Many navy helmets are seen, but with low confidence** (0.10–0.25), under the 0.25 the rule
  uses.
- **Why.** The Phase 1 training photos are mostly outdoors, in daylight, from the front, with
  yellow, white and orange helmets. Dark helmets seen from behind against bright skylights are
  rare in them. Phase 1 already listed white and blue helmets as failure case 6
  ([phase1_failure_cases.md](phase1_failure_cases.md)); these clips show it in a real workshop.
- **Lowering the threshold to 0.10 is not a fix on its own.** Navy helmets would be found in
  77 % of frames instead of 41 %. But a bare head would get a helmet in 31 % of frames, mostly
  the yellow helmet of the man standing just behind the seated one.

## Other things these clips show

- **A hand-held camera confuses the tracker.** The tracker expects a fixed camera. When the
  phone turns, IDs jump from one person to another: in clip 01, #2 passes from the navy-helmet
  man to a yellow-helmet man. A CCTV camera doesn't move, so this affects these test clips, not
  the system. The next clips should be filmed from a fixed spot.
- **There is no hi-vis at this site.** The vest rule raised an alert for every worker.
  **Decided 25 Sep:** hi-vis is not required here, so the `no_vest` rule is now switched off in
  `configs/rules.yaml` (`enabled: false`). The event check still scores vests, so model results
  stay comparable.
- **People further away are not found.** Only the 6 nearest people in each clip were tracked,
  about 210–890 px tall, out of the dozens further back on the workshop floor.

## What would fix the helmet misses

**Tried first: fine-tuning on more public data** with helmets of every colour and indoor factory
scenes (CHV, GDUT-HWD, SH17; [decision 0007](decisions/0007-fine-tuning-with-extra-public-data.md)).
It worked for the colours: see the next section. What public data can't teach is this site's haze
and distances. For that:

1. **Train on pictures like these.** Take 200–500 frames from this workshop, from other clips
   and not these two, and box every helmet whatever its colour. Fine-tune again, then measure these
   two clips again.
2. **Record the next test clips the way a CCTV camera sees.** Fix the phone high on a stand, in
   landscape, for 1–3 minutes in the same place. Include people without helmets on purpose, with
   their permission.

## After fine-tuning

The same clips, people and frames, with the fine-tuned candidate
(`models/candidates/ppe4_yolo26n_finetune.pt`). Full results: [finetune_results.md](finetune_results.md).

| Truth | Frames | A helmet found on the head (≥ 0.25): Phase 1 model | Fine-tuned |
|---|---|---|---|
| yellow helmet | 200 | 98 % | **100 %** |
| navy helmet | 251 | 41 % | **96 %** |
| brown helmet | 23 | 0 % | **91 %** |
| white helmet, in haze | 11 | 0 % | **91 %** |
| no helmet | 32 | 3 % | 3 % |

| Clip | Phase 1 model | Fine-tuned |
|---|---|---|
| 01 | 2 no-vest alerts | 2 no-vest alerts |
| 02 | 2 no-vest, 1 false no-helmet (navy helmet) | 5 no-vest, 1 false no-helmet (white helmet, in haze) |

- **The navy-helmet false alarm is gone:** that man is judged "worn" in all 132 of his frames.
- **The new false alarm** is on a man about 155 px tall in the thickest haze, at 12.5 s in clip
  02. His track had jumped between several people as the phone turned, so the 3 s of "missing"
  were not all his.
- **More no-vest alerts** because more people are tracked for long enough (489 person-frames in
  clip 02, against 304). Hi-vis isn't required at this site, so the vest rule is now off and none
  of these would be raised.

## Files (on the Mac)

`runs/real_clips/<clip>/` holds:
- `annotated.mp4`: the clip with every person's box and verdict;
- `events.json` and `snapshots/`;
- `heads_judged_missing.jpg`: each person's head in the frames where no helmet was found.

`runs/real_clips/helmet_scores_by_person.txt` has the numbers behind the colour table.
