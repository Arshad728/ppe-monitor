# 0007 — Fine-tuning on extra public datasets, without overtraining

**Status:** accepted. The candidate passed every check and has been the model in use since 25 Sep 2026 ([results](../finetune_results.md)) · **Book:** Chapters 7, 18 and 23

## Context

- **The failure.** The user's first two clips of a real workshop ([real_clips.md](../real_clips.md))
  show the Phase 1 model missing most navy, brown, and white-in-haze helmets. It found yellow ones
  98 % of the time and navy ones 39 %. The result was a false "no helmet" alert.
- **The cause.** Its training data (Construction-PPE and RF100 Construction Safety) is outdoor
  photos, mostly of yellow, white and orange helmets.
- **What the user asked for.** Train on public data from the internet, keep any relevant data
  whatever its licence (commercial use is undecided), and above all don't overtrain.

## Decision

1. **Three extra datasets** (`src/ppe_monitor/data/extra_sources.py`), each checked against a
   pinned checksum:

   | Dataset | Why | Licence |
   |---|---|---|
   | CHV (1,330 images) | people, hi-vis vests, and blue / red / white / yellow helmets | free to use, with citation |
   | GDUT-HWD (3,174 images) | 18,893 helmets in four colours (2,612 blue), bare heads, small distant people | none stated (research dataset); copy from Hugging Face, identical checksum to what was inspected |
   | SH17, part of it (528 images with a helmet or vest, and 700 of people with neither, head in view) | indoor factories and workshops, and bare heads at work | CC BY-NC-SA 4.0: **non-commercial** |

   - SH17's labels come from its Kaggle archive, read in place (6 MB of 14 GB).
   - Its photos come from Pexels at 1,280 px, which is all the training uses.
2. **Labels are checked before training** (`src/ppe_monitor/data/audit.py`). A person, helmet
   or vest that is visible but unlabelled teaches the model to ignore it, so such images are
   left out. Only whole images are dropped; no label is ever changed.
   - **GDUT-HWD has no person boxes.** A large COCO model (yolo26l) draws the people. An image
     is kept only if every labelled head sits at the top of one of those people; otherwise the
     model missed someone. GDUT-HWD also has no vest boxes, so any image where the current
     model sees a vest (≥ 0.3) is left out.
   - **SH17's vest labels are incomplete.** Images where the current model is sure of a vest
     (≥ 0.5) that has no label are left out. The same check runs on CHV for people, helmets and
     vests.
3. **The Phase 1 data is kept exactly** (`src/ppe_monitor/data/build_ext.py` → `datasets/ppe4`).
   - ppe3 is copied split for split. An extra image that looks like a ppe3 validation or test
     image is left out, so the old and new models can be compared on the same ppe3 test photos,
     which neither trained on.
   - Each extra dataset has its own test split. Half of GDUT-HWD's large test split is used for
     training.
4. **The model is fine-tuned, not trained from scratch, with these guards against overtraining**
   (`scripts/mac_retrain.sh train`):
   - **Starting point:** the Phase 1 model, which already knows the task, so few epochs are
     needed.
   - **Length:** at most 30 epochs, and early stopping after 8 epochs without improvement on
     validation.
   - **Validation set:** it mixes the old and the new data, so gaining on the new photos can't
     hide losing on the old ones.
   - **Checkpoint kept:** the best on validation, not the last.
   - **Report:** the model card has an overfitting check (`src/ppe_monitor/vision/overfit.py`)
     with, epoch by epoch, training against validation loss. It also gives the gap between
     accuracy on training images and on validation images, where more than 0.08 mAP@50 counts
     as memorising.
   - **Augmentation:** Ultralytics' usual augmentation (mosaic, colour, scale, flips) stays on.
     Colour changes matter here: the goal is to recognise helmets whatever their colour.
5. **The new model must prove itself before it is used.**
   - It is saved to `models/candidates/`, where no script picks it up.
   - The comparison uses the ppe3 test photos, and all the extra datasets' test photos pooled.
     A test run of the whole path showed how much the leak rules shrink CHV's test split: CHV
     shares many photos with GDUT-HWD and the ppe3 sources. GDUT-HWD's test split keeps about
     300 images with coloured helmets. On those, the model in use finds 64 % of blue helmets and
     55 % of red ones.
   - `scripts/compare_models.py` then compares it with the model in use, on photos neither
     trained on. It is accepted only if all three hold:
     - it is no more than 0.01 mAP@50 worse on the Phase 1 test photos;
     - it finds helmets better on the extra datasets' test photos;
     - it finds more blue helmets.
   - It is then also checked on the workshop clips. `mac_retrain.sh accept` puts it into use and
     remakes the Core ML / ONNX exports.

## Consequences

- **Licences.** The fine-tuned model is trained on SH17 data under a non-commercial licence, and
  on GDUT-HWD data with no stated licence.
  - That is fine for learning and for a portfolio.
  - If the system is ever sold, retrain without them: `scripts/prepare_more_data.py --sources chv`.
    This needs no code change.
  - Ultralytics itself is AGPL-3.0 either way (decision 0003).
- **Rule checks.** The Phase 2 rule checks and the Phase 4 benchmark still use the ppe3 test
  photos, so every earlier result stays comparable.
- **What public data can't teach.** It helps with colours, but not with this particular
  workshop's glare, haze and views from behind. If the workshop clips still show misses after
  fine-tuning, the next step is a few minutes of labelled footage from the workshop itself,
  kept apart from the test clips.

## Result (25 Sep 2026)

Full numbers: [finetune_results.md](../finetune_results.md).

- **Training:** 30 epochs in 193 min on the Mac; the kept checkpoint is epoch 24. Validation loss
  fell the whole time.
- **The three acceptance checks all passed:**
  - Phase 1 test photos, mAP@50 0.869 → 0.883;
  - helmet AP@50 on the extra test photos 0.666 → 0.918;
  - blue helmets found 128/220 → 201/220.
- **Workshop clips:** a helmet is found on the head in 96 % of navy-helmet frames (was 41 %), and
  91 % of brown and white (both 0 %). Bare heads stay at 3 %. One false helmet alert remains, on a
  small, hazy, hand-held track.
- **The overfitting check flagged the kept model by a small margin.** Its gap between training
  and validation mAP@50 is 0.085, against the 0.08 line. It is accepted anyway because every
  sign of real overtraining is absent:
  - validation loss never rose;
  - the validation score was flat for the last 7 epochs, not falling;
  - every test set improved.
  Most of the gap comes from the last 10 epochs being trained without mosaic (easier training
  images). The line stays at 0.08 for the next fine-tuning.
- **A side effect on vests.** The new model is more cautious about vests. On the Phase 2 moving
  clips, false "no vest" alerts on wearers rose from 6 to 11 of 81 (helmet events: about the
  same). The likely cause is this decision's own data: GDUT-HWD has no vest labels and SH17's are
  incomplete, and the label check left out only images with an *obvious* unlabelled vest. The
  thresholds were re-checked and stay at 0.25. If vests matter at a site, the next fine-tuning
  should give the extra images complete vest labels, or leave out every GDUT-HWD image with
  vest-like clothing.
