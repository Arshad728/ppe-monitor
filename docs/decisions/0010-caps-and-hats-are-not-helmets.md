# 0010 — Teaching the detector that caps and hats are not helmets

**Status:** accepted by the user on 29 Sep 2026, with one failed check recorded
([results](../caps_results.md)) · **Model in use:** `ppe4caps_yolo26n_caps2` · **Book:** Chapters 7 and 23

## Context

- **The miss.** On real video from other sites (Phase 6), a bricklayer in a baseball cap was never
  alerted: the model took his cap for a helmet.
- **How big it was.** On Open Images photos of people in hats, the model then in use
  (`ppe4_yolo26n_finetune`) called about one hat in four a helmet, against 2.4 % of bare heads.
- **The cause.** Its training data has hard hats and bare heads, but almost no other headwear.
  A cap is a dome with a brim, like a hard hat.
- **What the user asked for:** photos of people in caps and hats, labelled as having no helmet, in
  training; and a model that doesn't overfit.

## Decision

1. **Photos from Open Images V7** (Google; boxes CC BY 4.0, photos CC BY 2.0 by Flickr authors, each
   credited), chosen by its own boxes: a "Hat", "Sun hat", "Fedora", "Cowboy hat" or "Sombrero" worn by
   a labelled person, and no helmet of any kind labelled (`scripts/select_caps.py`,
   `data/caps/README.md`).
2. **Labels checked before training.**
   - Two models checked each photo. The COCO model (yolo26l) looked for people nobody had boxed:
     it added those it was sure of, and photos with ones it was unsure of were left out. The model
     in use looked for unlabelled helmets and uncertain vests.
   - **A person looked at every hat the model half-believed was a helmet** (1,939 crops in two
     reviews). 63 photos showed a real helmet labelled "Hat" (hard, pith, police, riding, bicycle,
     rafting, military) and were left out.
   - A hat gets no label: to the detector it is background, which is the lesson.
3. **Split by photographer**, so the 448 test photos show no scene or person training has seen.
   They are a separate `test_caps` list; ppe4's own test lists are unchanged.
4. **Mostly the photos it gets wrong:** 693 training photos whose hat the model called a helmet, and
   350 it got right. Added to ppe4's 5,051 (`datasets/ppe4caps`).
5. **Gentle fine-tuning** from the model in use (`scripts/mac_caps.sh`):
   - AdamW at 0.0002, falling to 0.00002;
   - 1 epoch of warm-up, with the biases starting at 0;
   - at most 15 epochs, early stop after 5 without improvement, best checkpoint kept.
6. **Accepted only if it passes every check** (`compare_models.py --checks caps`), or a person decides
   a failed one is worth it and says why (`--override`, recorded in the report and the model card).
   The checks cover:
   - the old test sets;
   - helmets still found, including every colour at the rule's threshold;
   - wearers judged "missing";
   - hats on the held-out photos;
   - memorising: the gap between training and validation;
   - end to end on every test clip, including the real video.

## Consequences

- **Attempt 1 (Ultralytics' default fine-tuning) was rejected.**
  - Its warm-up (biases starting at a learning rate of 0.1) knocked the helmet scores down: every
    colour lost about 5 points.
  - A navy helmet in the user's workshop clip was missed, raising a false alarm.
  - Three checks failed. The colour check was added because of it.
- **Attempt 2 is in use.**
  - People in hats judged "helmet worn": 27.4 % -> 2.5 %.
  - The capped bricklayer is caught. The workshop clips are clean.
  - All 266 test clips: 98 caught (97 before); 12 false alarms, as before.
  - Every helmet colour is found within 1 point of before.
  - No overfitting: the gap between training and validation is 0.078, below the 0.08 line and the
    old model's 0.085.
- **Its cost, accepted by the user.** On the Phase 1 test photos, helmet AP@50 fell 0.012 (limit
  0.01). At the rule's threshold that is 5 helmets lost and 2 gained of 350, all small or half
  hidden. Wearers are judged "missing" as often as before.
- **Vests move a little** (wearers judged "no vest" 7.0 % -> 9.5 %): to check before the vest rule is
  switched on.
- **Retraining on false alarms** (Phase 6) now builds on the model in use's own training data, read
  from its model card, so the caps lesson isn't lost.
- **Licences:** Open Images adds CC BY material, credited in `datasets/ppe4caps/credits.csv`. The
  non-commercial caveat of decision 0007 still applies.
