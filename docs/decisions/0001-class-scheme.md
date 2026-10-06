# 0001 — Detect person, helmet and vest; decide "wearing" with geometry

**Status:** accepted (Phase 0) · **Book:** Chapters 10 and 17

## Decision

The detector learns three plain classes — `person`, `helmet`, `vest` (`configs/classes.yaml`).
Whether a given person is *wearing* a helmet or vest is decided afterwards, in `rules/`, by
checking where the helmet and vest boxes sit relative to each person's box (Chapter 10).

## The alternative we rejected

Many public PPE datasets label violations directly: `NO-Hardhat`, `NO-Safety Vest`. Training on
those lets the model report "person without helmet" in one step.

## Why

- **"Absence" is hard to label and to learn.** A `NO-Hardhat` box has to be drawn around a
  bare head, and labellers disagree about partly hidden heads. A helmet is a visible object;
  its absence is a conclusion. Conclusions belong in code we can read and test.
- **The helmet-on-the-railing case.** A violation-class model sees a helmet in the picture and
  may call the scene compliant. With separate classes plus geometry, a helmet that isn't
  on anyone's head simply doesn't match any person.
- **Easier to extend.** Adding gloves or harnesses later means one new object class and one
  new matching rule, not a new "NO-" class for every item.
- **Evaluation stays honest.** mAP@50 measures the detector on things that are really there;
  the rules are measured separately at the event level in Phase 6.

## Consequences

- Datasets must be merged by renaming their labels onto our three classes. The alias table in
  `configs/classes.yaml` does this; `NO-*` labels are dropped.
- Phase 2 has to implement and test the matching geometry, including crowded scenes where boxes
  overlap.
- A detector miss on a helmet produces a "no helmet" conclusion. The dwell time and majority
  voting in Phase 2 exist to stop one bad frame from becoming an alert.

## Refinement (Phase 1 review, 23 Sep 2026): what "vest" means

**"Vest" means any high-visibility garment:** a hi-vis vest, jacket, coverall or shirt with
reflective strips. Wearing any of them counts as wearing a vest. Safety harnesses, life
jackets and ordinary vests without hi-vis material are **not** vests.

Why: the site rule is about being seen, and a vest is only one way to meet it. Phase 1 found the
public datasets inconsistent here. Harnesses and life jackets are labelled "vest", while
hi-vis jackets and coveralls are labelled inconsistently and often missed
(`docs/phase1_failure_cases.md`, case 4).

Consequences:
- The labelling guide for the held-out clips follows this definition.
- A worker in a hi-vis jacket that the detector misses would trigger a false "no vest" alarm.
  The held-out clips must include hi-vis jackets and coveralls, so this rate can be measured.
- The public training labels are not relabelled for now. If the held-out clips show hi-vis
  jackets being missed, the Phase 6 retraining adds examples of them.
