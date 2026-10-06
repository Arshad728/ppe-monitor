# 0003 — Training data sources, label merging and an honest test split

**Status:** accepted (Phase 1 review, 2026-09-23) · **Book:** Chapters 8, 9, 18 and 23

## Context

Phase 1 needs labelled photos of people, helmets and hi-vis vests to fine-tune YOLO, plus a
test split the model never trained on. Public PPE datasets differ in their label names
("hardhat", "Person", "safety vest"), in the extra classes they carry (gloves, boots, "no
helmet") and in their licences. Two problems showed up when the chosen datasets were inspected:

1. **Near-copies across splits.** One dataset contains a staged photo session: the same person
   against the same background, photographed many times with different PPE. Copies of it were in
   train, val *and* test, so the test score would reward memorising that scene.
2. **The same photo in both datasets.** 15 images appear in both sources, sometimes in train in
   one and test in the other.

## Decision

**Sources** (both downloaded by `scripts/prepare_data.py` and checked against a SHA-256):

| Source | Licence | Images | Why |
|---|---|---|---|
| Construction-PPE (Ultralytics) | AGPL-3.0 | 1,416 | Clean labels for all three classes; includes people with no PPE |
| Construction Safety (Roboflow 100) | CC BY 4.0 | 1,206 | Real site scenes with people further from the camera |

**Labels.** Every dataset label is mapped onto `person / helmet / vest` through the aliases in
`configs/classes.yaml`, or dropped through its ignore list. "No helmet", "no vest" and "none" (a
torso without a vest, checked by eye) are dropped: Phase 2 decides "missing PPE" by matching
helmets and vests to people, so the detector never needs a "missing" class (decision 0001).
**An unknown label stops the build.** A new dataset can't slip in a label nobody looked at.

**Honest splits.**

- Every image gets a 64-bit difference hash (dHash).
- Images within Hamming distance 6 of each other are grouped (union-find).
- Any group that spans more than one split is moved *entirely into train*, so val and test only
  lose images and never gain any.
- Photos that are exact duplicates across the two sources (distance ≤ 2) are kept once.
- Near-identical frames within one source are kept, because they are legitimate variants (same
  scene, different PPE).

The build writes `datasets/ppe3/report.md`, which lists every label decision, every image moved
and the final counts per split and per object size.

## Consequences

- 56 images moved out of val/test into train (the largest group had 60 members). The
  test split is smaller (200 images) but its score means something.
- **The public test split can't measure distant workers.** It has only 3 small people
  (< 32×32 px at 640). CCTV mostly sees people at that size. Recall on distant workers must be
  measured on real held-out clips (`data/README.md`), not inferred from this split.
- **Licences.** Ultralytics (the training library) and Construction-PPE are AGPL-3.0. That's fine
  for learning and portfolio use. A commercial deployment would need an Ultralytics licence or a
  differently licensed detector and data. The Roboflow 100 data requires attribution, which
  `data/README.md` and the model card give.
- The RF100 images were stretched to 640×640 when exported, so people in them look squashed.
  Training still helps, but it's one reason real clips matter.
