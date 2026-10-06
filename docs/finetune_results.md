# Fine-tuning results: the Phase 1 model plus three public datasets

**25 Sep 2026** · MacBook Air M4 (Apple GPU) · **Model:** `models/ppe4_yolo26n_finetune.pt`,
**in use since 25 Sep 2026** (`mac_retrain.sh accept`, 22:13; Core ML and ONNX versions made) · **Why and how:**
[decision 0007](decisions/0007-fine-tuning-with-extra-public-data.md) · **Commands:**
`bash scripts/mac_retrain.sh data | train | compare | accept`

## In short

- **It passed every check** against the model in use, on photos neither model trained on. It is
  better on the Phase 1 test photos too (mAP@50 0.869 → 0.883).
- **On the workshop clips it now finds the helmets the old model missed.** A helmet is found on
  the wearer's head in 96 % of the navy-helmet frames (was 41 %), 91 % of the brown (was 0 %) and
  91 % of the white-in-haze (was 0 %). Bare heads are still almost never given a helmet (3 %, as
  before).
- **The overfitting check flagged it, by a hair.** The model scores 0.085 higher on training
  images than on validation ones; the line is 0.08. Every other sign is healthy: validation loss
  fell for all 30 epochs and scores rose on every test set. [What that means](#overfitting-check).
- **Moving clips: helmets about the same, vests a little worse.** False "no vest" events on
  wearers went from 6 to 11 of 81 on the Phase 2 moving clips, although the one-frame check on the
  same photos improved. [Details](#5-moving-clips-the-phase-2-event-check).
- **One false helmet alert remains** on the workshop clips. The old model's false alarm (a navy
  helmet) is gone; the new one is on a man in a white helmet, small and in heavy haze, on a track
  that had jumped between people as the phone turned.
- **Put into use the same evening.** It fixes the failure that matters at your site (helmet
  colours). Vests are the trade-off, and they don't matter there: hi-vis isn't required, so the
  vest rule is now off.

## 1. The data

`mac_retrain.sh data` took 20 min on the Mac. Images where something visible had no label were
left out (decision 0007, step 2):

| Dataset | Images | Kept | Left out, and why |
|---|---|---|---|
| CHV | 1,330 | 1,122 | 139 an unlabelled person, 58 an unlabelled vest, 11 an unlabelled helmet |
| GDUT-HWD | 3,174 | 1,747 (9,164 people drawn by yolo26l) | 879 an unlabelled vest, 487 a head with no person found around it, 61 an unlabelled helmet |
| SH17 (part) | 1,228 | 1,164 | 59 an unlabelled vest, 5 photos that couldn't be downloaded |

Then 208 extra images that looked like a Phase 1 validation or test photo were left out (176 of
them CHV, which shares photos with the Phase 1 sources). The result, `datasets/ppe4`:

| Split | Phase 1 (ppe3) | CHV | GDUT-HWD | SH17 | Total | person / helmet / vest boxes |
|---|---|---|---|---|---|---|
| train | 2,170 | 934 | 1,002 | 945 | **5,051** | 14,316 / 10,045 / 3,944 |
| val | 237 | 5 | 100 | 112 | 454 | 1,312 / 885 / 359 |
| test | 200 | 7 | 320 | 101 | 628 | 2,587 / 1,714 / 311 |

The ppe3 splits are copied unchanged, so the Phase 1 test photos are still a fair test for both
models.

## 2. Training

| | |
|---|---|
| Start | the Phase 1 model (`ppe3_yolo26n_baseline.pt`), not from scratch |
| Epochs | 30 of at most 30; early stop after 8 without improvement never triggered |
| Kept | epoch 24, the best on validation (mAP@50-95 0.539, mAP@50 0.866) |
| Time | 193 min on the Apple GPU |

The validation score rose until about epoch 23 and then stayed flat (0.537–0.539), so more
epochs would not have helped.

### Overfitting check

From the model card (`models/candidates/ppe4_yolo26n_finetune.md`):

| Sign | Result | Healthy? |
|---|---|---|
| Validation loss after the best epoch | kept falling: 2.235 at epoch 24, 2.187 at epoch 30 (2.573 at epoch 1) | yes: it never turned upwards |
| Validation score after the best epoch | flat (0.537–0.539) | yes |
| mAP@50 on 300 training images against validation | 0.942 against 0.857: a gap of **0.085** | just over the 0.08 line |
| Scores on test photos neither model saw | up on every test set (section 3) | yes |

**What the flag means here.** A gap between training and validation scores always exists; the
0.08 line is where I decided it would be worth a look. At 0.085 it is just over. Three things
explain most of it:

- **The last 10 epochs are trained without mosaic** (Ultralytics' default). Training images get
  easier, and the training loss drops sharply at epoch 21 (2.153 → 1.955) while the validation
  loss doesn't move. The model has seen each training image about 30 times by the end.
- **The validation set is harder than the training set on average.** It has proportionally more
  GDUT-HWD and SH17 images: small, distant people and indoor scenes.
- **What memorising looks like is missing.** A model that memorises gets *worse* on new photos.
  This one scores higher (mAP@50) on every test set.

So the kept model is not overtrained in the sense that matters: it did not trade new photos for
old ones. The flag stays in the model card as a reminder for the next fine-tuning, which should be
checked the same way.

## 3. Old against new, on photos neither trained on

`mac_retrain.sh compare` (`runs/compare/20260925_183929/report.md`):

| Check | Result | |
|---|---|---|
| Not worse on the Phase 1 test photos (mAP@50 at most 0.01 lower) | **PASS** | 0.869 → 0.883 |
| Finds helmets better on the extra datasets' test photos (helmet AP@50, 428 photos) | **PASS** | 0.666 → 0.918 |
| Finds more blue helmets | **PASS** | 128/220 → 201/220 |

| Test photos | mAP@50 old → new | person | helmet | vest |
|---|---|---|---|---|
| Phase 1 (ppe3, 200) | 0.869 → **0.883** | 0.838 → 0.839 | 0.921 → 0.934 | 0.849 → 0.878 |
| All extra datasets (428) | 0.649 → **0.799** | 0.764 → 0.907 | 0.666 → 0.918 | 0.519 → 0.573 |
| GDUT-HWD (320) | 0.714 → 0.915 | 0.764 → 0.909 | 0.664 → 0.922 | (no vest labels) |
| SH17 (101) | 0.707 → 0.780 | 0.782 → 0.874 | 0.749 → 0.843 | 0.589 → 0.622 |
| CHV (7) | 0.689 → 0.862 | | | too few to judge |

- GDUT-HWD's people were drawn by a COCO model, so its person figures mean less than its helmet
  figures.
- **Vests on the extra datasets stay weak** (0.573). SH17's vests are often partly hidden or
  seen side-on, and there are few of them (311 in all the test photos).

**Helmets found, by colour** (at the rule's threshold 0.25, on the test photos that record the
colour):

| Colour | Helmets | Old | New |
|---|---|---|---|
| blue | 220 | 58 % | **91 %** |
| red | 457 | 53 % | **89 %** |
| white | 302 | 59 % | **89 %** |
| yellow | 333 | 63 % | **88 %** |

**The PPE rule on one frame** (Phase 2 check, Phase 1 test photos):

| | Old | New |
|---|---|---|
| Helmet: wearer judged missing | 4.5 % | **2.7 %** |
| Helmet: bare head judged worn | 2.3 % | 2.4 % |
| Vest: wearer judged missing | 9.5 % | **7.0 %** |
| Vest: no vest judged worn | 8.6 % | **5.0 %** |

Fewer false "missing" judgements and fewer missed vests, for the same rate of bare heads judged
worn.

## 4. The workshop clips

The same two clips as [real_clips.md](real_clips.md), the same people, checked frame by frame
against what each person really wears. "Found" = a helmet detection with score ≥ 0.25 (the
setting in use) on the person's head.

| Truth | People | Frames | Old model | New model |
|---|---|---|---|---|
| yellow helmet | 2 | 200 | 98 % | **100 %** |
| navy helmet | 5 | 251 | 41 % | **96 %** |
| brown helmet | 1 | 23 | 0 % | **91 %** |
| white helmet, in haze | 1 | 11 | 0 % | **91 %** |
| no helmet | 2 | 32 | 3 % | 3 % |

The median helmet score on navy helmets went from 0.09–0.35 to 0.43–0.77, well clear of the 0.25
threshold, so this doesn't depend on a lucky setting.

**The full chain on the clips** (detector, keypoints, tracker, rules, events):

| Clip | Old model | New model |
|---|---|---|
| 01 | 2 no-vest alerts | 2 no-vest alerts |
| 02 | 2 no-vest, **1 false no-helmet** (navy helmet, judged worn in 9 of 87 frames) | 5 no-vest, **1 false no-helmet** |

- **The old false alarm is gone.** The navy-helmet man is now judged "worn" in all 132 of his
  frames.
- **A new one appeared, on a harder case.** At 12.5 s in clip 02, a no-helmet alert on a man in a
  white helmet, about 155 px tall, in the thickest haze. His track had jumped between several
  people as the phone turned, so the 3 seconds of "missing" that raised the alert were not all
  his. A fixed camera would not switch like this, but haze and small, distant heads are real
  workshop conditions.
- **The extra no-vest alerts are correct by the rules.** The new model finds more of the people:
  in clip 02 it tracks 489 person-frames against the old model's 304, so more people stay in view
  for the 3 s an alert needs. Nobody at this site wears hi-vis.
- **The men who really had no helmet** still get a helmet in only 1 of their 32 frames.

## 5. Moving clips (the Phase 2 event check)

The 86 moving clips made from the Phase 1 test photos (everyone sways at walking pace, 15 fps),
through the whole chain, in the cloud (CPU). Old model: the cloud run of 24 Sep, same clips.

| | Old | New |
|---|---|---|
| No helmet: exactly one event | 42/47 | 40/47 |
| No helmet: missed | 5 | 7 |
| Helmet worn: false events | 5 | **4** |
| No vest: exactly one event | 55/68 | 55/68 |
| No vest: more than one event | 0 | 1 |
| No vest: missed | 13 | 12 |
| Vest worn: false events | 6 | **11** |
| ID switches | 113 | 107 |

**Helmets: about the same.** Three new misses and one fixed, and one false event fewer. The new
misses are not helmets judged wrongly: the new model barely finds these people at all (a woman
seated at an office desk, a welder seen from above behind a stock-photo watermark, a child in a
classroom), so they get no verdict and no event. None of them looks like a worker on a site.

**Vests: more false "no vest" events, 6 → 11.** This is the one measure that got worse, and it
runs against the one-frame check on the same photos (wearer judged missing 9.5 % → 7.0 %). All 6
old false events are still there, plus 5 new ones. Looking at them:

- **Clip 057 (5 of the 11; 2 of them new)** is a tight line-up of about ten people in green vests,
  shoulder to shoulder, where vests get matched to the wrong neighbour. The old model had 3 false
  events there.
- **Clip 005:** a cyclist in a plain yellow jacket that the Phase 1 labels count as a vest. The new
  model no longer calls it one. By decision 0001 (a vest is a hi-vis garment), the new model is
  arguably right and the label is generous.
- **Clip 033:** an orange strap vest that looks much like a harness. The old model scored it as a
  vest and the new one doesn't.
- **Clip 066:** a real yellow hi-vis vest on a man bending sideways, half hidden. Both models score
  it low; the new one just under the 0.25 threshold.

The likely cause is the extra data. GDUT-HWD has no vest labels, and SH17's are incomplete. Images
where the old model was *sure* of an unlabelled vest were left out, but less obvious vests stayed
in, unlabelled, and taught the model to be more cautious about vest-like clothing. That helps on
look-alikes and hurts on vests that are half hidden.

**Does it matter at your site?** No: hi-vis isn't required there, and the vest rule is now
switched off (`configs/rules.yaml`). It would matter at a site where vests are required.

**Would a lower vest threshold fix it? Barely.** I re-checked the thresholds with the new model the
way Phase 2 chose them (`scripts/check_ppe_rules.py --sweep`, one frame per photo):

| Vest threshold | Validation photos: vest wearer judged "missing" / no vest judged "worn" | Test photos: the same |
|---|---|---|
| **0.25 (in use)** | 9.8 % / 0.0 % (old model: 4.1 % / 3.9 %) | 7.0 % / 5.0 % (old model: 9.5 % / 8.6 %) |
| 0.20 | 8.8 % / 1.0 % | 7.0 % / 5.0 % |
| 0.15 | 7.4 % / 1.9 % | 6.9 % / 5.0 % |

- **The two photo sets disagree about vests:** worse than the old model on the validation photos,
  better on the test photos. The new model is more cautious about vests: it almost never gives a
  vest to someone without one, but misses more real ones.
- **A lower threshold changes little,** because most of the false "missing" answers are vests the
  model doesn't find at all. So the vest threshold stays at 0.25.
- **The fix, if vests matter at a site, is in the data:** give the extra images complete vest
  labels (or leave out every GDUT-HWD image with vest-like clothing in it), then fine-tune again
  the same way.

The helmet threshold stays at 0.25 too. On the validation photos the new model gets 3.5 % false
"missing" and 1.9 % bare heads judged "worn" (old model: 3.9 % and 3.9 %).

## What's next

1. **Put into use: done** (`mac_retrain.sh accept`, 25 Sep 22:13). It is in `models/`, where
   every script picks it up, with its Core ML and ONNX versions. `configs/ppe.yaml` names it as the
   model its thresholds were checked with (the thresholds are unchanged). From now on,
   `accept` updates that line itself.
2. **Workshop footage from a fixed camera.** Public data fixed the helmet colours. What it can't
   teach is this site's haze and distances. A few minutes of footage filmed from a fixed spot, with
   some frames labelled, is the next step (decision 0007, Consequences).
3. **Hi-vis at this site: done.** It isn't required, so the `no_vest` rule is switched off in
   `configs/rules.yaml`. Every no-vest alert on these clips came from that rule.
