"""The class scheme is a Phase 0 decision that Phase 1's dataset merging depends on, so it is
checked for internal consistency here."""

import re

import yaml

from ppe_monitor.config import PROJECT_ROOT


def normalise(name: str) -> str:
    return re.sub(r"[\s_\-]+", "", name.lower())


def load():
    return yaml.safe_load((PROJECT_ROOT / "configs" / "classes.yaml").read_text(encoding="utf-8"))


def test_three_classes_in_fixed_order():
    assert load()["classes"] == ["person", "helmet", "vest"]


def test_every_class_has_aliases_and_no_extras():
    cfg = load()
    assert set(cfg["aliases"]) == set(cfg["classes"])


def test_no_alias_maps_to_two_classes_or_is_also_ignored():
    cfg = load()
    owner = {}
    for cls, aliases in cfg["aliases"].items():
        for alias in aliases:
            key = normalise(alias)
            assert key not in owner, f"alias {alias!r} used by both {owner[key]} and {cls}"
            owner[key] = cls
    ignored = {normalise(name) for name in cfg["ignore"]}
    assert not ignored & set(owner), f"labels both mapped and ignored: {ignored & set(owner)}"
