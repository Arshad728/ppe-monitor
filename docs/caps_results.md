# Caps and hats are not helmets: fine-tuning with Open Images photos

**Why:** on real video from other sites (Phase 6, [results](phase6_results.md#real-video-from-other-sites-28-sep-2026)),
a bricklayer in a baseball cap was never alerted: the model took his cap for a helmet. On photos of
people in hats, the model then in use called about one hat in four a helmet. **Data:**
[data/caps/README.md](../data/caps/README.md). **Why it is built this way:**
[decision 0010](decisions/0010-caps-and-hats-are-not-helmets.md). **Book:** Chapter 23.

**Outcome:** the second attempt, `ppe4caps_yolo26n_caps2`, is the model in use since 29 Sep 2026.
People in hats judged "helmet worn" fell from 27.4 % to 2.5 %, with no more false alarms end to end
and no overfitting. One check failed narrowly (helmet AP@50 on the Phase 1 test photos, 0.012
lower), and the user accepted it with the reason recorded.

```bash
bash ~/Documents/ppe-monitor/scripts/mac_caps.sh run       # data, train, compare (~2-3 h)
bash ~/Documents/ppe-monitor/scripts/mac_caps.sh accept    # only if every check passed
bash ~/Documents/ppe-monitor/scripts/mac_caps.sh accept --override "reason"   # a person's decision over a failed check
```

## Attempt 1 (29 Sep 2026): not accepted

**Data:** version 1 of the manifest. datasets/ppe4 (5,051 training photos, 454 validation) plus
1,400 training photos of people in hats (700 the model gets wrong, 700 it gets right) and 250
validation ones.

**Training:** from the model in use, Ultralytics' default fine-tuning settings (AdamW chosen
automatically at 0.00143, 3 epochs of warm-up with the biases starting at 0.1). It stopped after 5
epochs, 38 minutes. The best epoch was the **first**:

| Epoch | Train loss | Val loss | Val mAP@50 | Val mAP@50-95 |
|---|---|---|---|---|
| 1 (kept) | 2.187 | 2.166 | 0.859 | 0.518 |
| 2 | 2.261 | 2.286 | 0.845 | 0.495 |
| 3 | 2.321 | 2.300 | 0.827 | 0.485 |
| 4 | 2.338 | 2.318 | 0.829 | 0.487 |
| 5 | 2.296 | 2.246 | 0.845 | 0.506 |

Training loss **rose** during warm-up. That is not overfitting (which shows as training loss
falling while validation loss rises): the learning rate was too high for a model already trained
for this task, and knocked it off what it had learned. No memorising: the gap between training
and validation mAP@50 was 0.066 (the model in use: 0.085).

**The comparison** (`runs/compare/20260929_011554/report.md`), on photos and clips neither model
trained on:

| Check | Model in use | Candidate | Result |
|---|---|---|---|
| People in hats judged "helmet worn" (449 held-out photos, 634 people) | 27.5 % | **6.4 %** | pass |
| Hats with a helmet box on them | 25.0 % | 6.4 % | |
| Phase 1 test photos, mAP@50 | 0.883 | 0.878 | pass |
| Extra test photos, mAP@50 (428) | 0.799 | 0.787 | **fail** (0.012 lower) |
| Helmet AP@50: Phase 1 / extra | 0.934 / 0.918 | 0.928 / 0.908 | **fail** (extra 0.010 lower) |
| Helmet wearers judged "missing" (Phase 2 rule check) | 2.7 % | 3.8 % | **fail** (+1.1 points) |
| Helmets found at the rule's threshold: blue / red / white / yellow | 91 / 89 / 89 / 88 % | 86 / 84 / 84 / 85 % | (not a check yet) |
| End to end, all 266 test clips: caught | 94 / 118 | 97 / 118 | pass |
| End to end: false alarms | 13 | 12 | pass |
| Real video (site set): caught / false | 4 of 6 / 1 | 5 of 6 / 1 | |

- **What it fixed:** hats called helmets fell by three quarters. On the real video, the
  bricklayer in the baseball cap was now alerted, and the tractor false alarm was gone.
- **What it broke:** it found helmets of every colour less often at the rule's threshold (about 5
  points), and a **navy helmet on your workshop clip 01 was missed**, raising a new false alarm.
  Navy helmets are what the fine-tuning of 25 Sep (decision 0007) had fixed.
- **So the model in use stayed.** Three checks failed, and the one that matters most for your site
  (navy helmets) wasn't even a check yet.

## Attempt 2: what changed, and why

1. **Gentler training.** AdamW at 0.0002 falling to 0.00002 (a seventh of before), 1 epoch of
   warm-up with the biases starting at 0, not 0.1. With AdamW, a bias learning rate of 0.1 moves
   the class scores a long way in the first epoch: the likely reason every helmet colour lost
   about 5 points at once. At most 15 epochs, early stop after 5 without improvement.
2. **Fewer photos the model already gets right:** 350 instead of 700. They teach little, and every
   photo with no helmet in it pushes helmet scores down. All 693 it gets wrong stay.
3. **A second review by eye** of every hat the model scores 0.05-0.25 as a helmet (585 hats): 13
   photos had a real helmet labelled "Hat" (pith, police, riding, bicycle, military) and were left out.
4. **A new acceptance check:** helmets of every colour must be found at the rule's threshold at most
   2 points less often. Attempt 1 would have failed it (blue -5 points).

Everything else is unchanged: the same test photos (448, by photographers never trained on), the
same checks, and both models end to end on every test clip, including your real video.

## Attempt 2 (29 Sep 2026): 9 of 10 checks passed; accepted by the user with the reason recorded

**Training** (`models/candidates/ppe4caps_yolo26n_caps2.md`): all 15 epochs, 1.7 h. Training and
validation loss both fell, and validation mAP@50 rose from 0.860 to 0.870. The last 3 epochs, without
mosaic, lowered training loss sharply (2.02 -> 1.78) but not validation loss: the start of
memorising, and not kept. The kept model is epoch 12: gap between training and validation mAP@50
**0.078**, under the 0.08 line and under the model in use's own 0.085. No sign of overfitting.

| Epoch | Train loss | Val loss | Val mAP@50 |
|---|---|---|---|
| 1 | 2.158 | 2.093 | 0.860 |
| 6 | 2.051 | 2.065 | 0.865 |
| 12 (kept) | 2.018 | 2.027 | 0.870 |
| 13 (no mosaic from here) | 1.807 | 2.042 | 0.863 |
| 15 | 1.778 | 2.032 | 0.866 |

**The comparison** (`runs/compare/20260929_212900/report.md`; end to end:
`runs/evaluation/20260929_211410_in_use` and `..._caps_candidate`):

| Check | Model in use | Candidate | Result |
|---|---|---|---|
| People in hats judged "helmet worn" (447 held-out photos, 631 people) | 27.4 % | **2.5 %** | pass |
| Hats with a helmet box on them (657 hats) | 25.0 % | 2.6 % | |
| Phase 1 test photos, mAP@50 | 0.883 | 0.877 | pass |
| Extra test photos, mAP@50 (428) | 0.799 | 0.804 | pass |
| Helmet AP@50: Phase 1 / extra | 0.934 / 0.918 | 0.922 / 0.920 | **fail** (Phase 1: 0.012 lower, limit 0.01) |
| Helmets found at the rule's threshold: blue / red / white / yellow | 91 / 89 / 89 / 88 % | 90 / 89 / 88 / 88 % | pass |
| Helmet wearers judged "missing" (Phase 2 rule check) | 2.7 % | 2.7 % | pass |
| Training minus validation mAP@50 | 0.085 | 0.078 | pass |
| End to end, all 266 test clips: false alarms | 12 | 12 | pass |
| End to end: caught | 97 / 118 | 98 / 118 | pass |
| Real video (site set): caught / false | 4 of 6 / 1 | 5 of 6 / 1 | |

**The one failed check, looked at** (in the cloud, the same 200 Phase 1 test photos, 350 helmets):
- At the rule's threshold (0.25) the candidate finds **5 helmets the model in use finds, and misses
  them; it finds 2 the model in use misses.** Helmet recall 92.0 % -> 91.1 %; precision 84.1 % ->
  85.8 % (fewer helmet boxes where there is no helmet).
- The 5 lost helmets: red helmets half hidden in crowds (3, 44-70 px), a small distant yellow one
  (19 px), a white one (29 px). None navy; none on people alone in view.
- A paired bootstrap over the 200 photos puts the AP difference at -0.012 (95 % interval -0.028 to
  +0.001): a real, small loss on these photos, not just noise.
- Where it would show, it doesn't: helmet wearers are judged "missing" as often as before (2.7 %),
  every colour is found as often (within 1 point), and end to end there are no more false alarms.

**Other differences, not checks:**
- The vest rule, off at your site: wearers judged "no vest" 7.0 % -> 9.5 %, people without one judged
  wearing it 5.0 % -> 3.6 %. Worth a look before the vest rule is ever switched on.
- The tractor false alarm on the brick-loading clip came back, 15 s later in the clip, with both
  models: the same borderline "person" at the frame's edge.
- Your workshop clips: no false alarm (attempt 1 had one, on a navy helmet).
- Run to run, the same model in use caught 94 and 97 of the 118 violations (29 Sep, 01:01 and 21:14),
  and raised 13 and 12 false alarms: differences of 1-3 events between two models are within that.

**Decision (the user, 29 Sep 2026):** accept it, with the failed check and this reason recorded in
the report and the model card: *helmet AP@50 on the Phase 1 test photos 0.012 lower (limit 0.01):
5 small or half-hidden helmets lost and 2 gained of 350 at the rule's threshold; wearers judged
missing, colour recall and end-to-end false alarms unchanged; people in hats judged "helmet worn"
27 % -> 2.5 %.*

```bash
bash ~/Documents/ppe-monitor/scripts/mac_caps.sh accept --override "Phase 1 helmet AP 0.012 lower (limit 0.01): 5 small or half-hidden helmets lost, 2 gained of 350; wearers judged missing, colours and false alarms unchanged; hats judged helmets 27% -> 2.5%. See docs/caps_results.md"
```

**Accepted** (29 Sep 2026, 21:43): `models/ppe4caps_yolo26n_caps2.pt` is the model in use; its card
(`models/ppe4caps_yolo26n_caps2.json`, "accepted") records the failed check and the reason, and
`configs/ppe.yaml` names it. The comparison run at acceptance (`runs/compare/20260929_214344`) gave
the same verdict. Its Core ML, Core ML 8-bit, ONNX and ONNX 8-bit versions were made; the export
script then aborted while shutting down (a macOS teardown crash after all four were written), which
`scripts/export_models.py` now avoids. Why it was built this way: [decision 0010](decisions/0010-caps-and-hats-are-not-helmets.md).
