"""Ground truth for "is this person wearing a helmet / vest?", taken from dataset labels.

The detector only learns person, helmet and vest. But the source datasets also mark absences, for
example a bare head ("no_helmet") or a torso without a vest ("none"). `configs/classes.yaml` lists
these under `absent`. Combining both kinds of label gives, for many labelled people, a known answer
to "wearing a helmet?" and "wearing a vest?". The PPE rules (Phase 2) are checked against exactly
those people.

A person gets a known answer for an item only when it is clear-cut:
  - every helmet, vest or absence box is given to the one labelled person whose box contains it
    (at least 60% of the item's area); a box contained by two people is ambiguous and makes the
    answer unknown for both;
  - an item box and an absence box of the same kind on one person contradict each other: unknown;
  - neither kind of box on a person: unknown. The labeller may simply not have looked.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .labels import CLASSES_FILE, ClassMap, normalise
from .readers import Box

CONTAIN = 0.6  # share of an item's area that must lie inside a person box to belong to that person
ITEMS = ("helmet", "vest")


@dataclass(frozen=True)
class PersonTruth:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 as 0-1 image fractions
    helmet: bool | None                     # True: wearing; False: not wearing; None: not known
    vest: bool | None


def absence_labels(path: str | Path = CLASSES_FILE) -> dict[str, str]:
    """{normalised label: item} for labels that mark a missing helmet or vest."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    ignore = {normalise(n) for n in cfg.get("ignore", [])}
    out = {}
    for item, names in (cfg.get("absent") or {}).items():
        if item not in ITEMS:
            raise ValueError(f"`absent` lists {item!r}; only {ITEMS} are supported")
        for name in names:
            if normalise(name) not in ignore:
                raise ValueError(f"absence label {name!r} must also be in `ignore`, or the detector would train on it")
            out[normalise(name)] = item
    return out


def _containment(item: Box, person: Box) -> float:
    ix = max(0.0, min(item.x2, person.x2) - max(item.x1, person.x1))
    iy = max(0.0, min(item.y2, person.y2) - max(item.y1, person.y1))
    return ix * iy / item.area if item.area > 0 else 0.0


def person_truths(boxes: list[Box], class_map: ClassMap, absent: dict[str, str]) -> list[PersonTruth]:
    """One PersonTruth per labelled person in an image."""
    people, marks = [], []  # marks: (kind, item, box) with kind "has" or "lacks"
    for b in boxes:
        d = class_map.decide(b.label)
        if d.kind == "class" and d.class_name == "person":
            people.append(b)
        elif d.kind == "class" and d.class_name in ITEMS:
            marks.append(("has", d.class_name, b))
        elif normalise(b.label) in absent:
            marks.append(("lacks", absent[normalise(b.label)], b))

    has = [{i: 0 for i in ITEMS} for _ in people]
    lacks = [{i: 0 for i in ITEMS} for _ in people]
    ambiguous = [set() for _ in people]
    for kind, item, box in marks:
        owners = [k for k, p in enumerate(people) if _containment(box, p) >= CONTAIN]
        if len(owners) == 1:
            (has if kind == "has" else lacks)[owners[0]][item] += 1
        else:
            for k in owners:
                ambiguous[k].add(item)

    out = []
    for k, p in enumerate(people):
        answer = {}
        for item in ITEMS:
            if item in ambiguous[k] or (has[k][item] and lacks[k][item]):
                answer[item] = None
            elif has[k][item]:
                answer[item] = True
            elif lacks[k][item]:
                answer[item] = False
            else:
                answer[item] = None
        out.append(PersonTruth((p.x1, p.y1, p.x2, p.y2), answer["helmet"], answer["vest"]))
    return out
