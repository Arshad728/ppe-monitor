"""Mapping each dataset's own label names onto our classes (configs/classes.yaml)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..config import PROJECT_ROOT

CLASSES_FILE = PROJECT_ROOT / "configs" / "classes.yaml"


def normalise(name: str) -> str:
    """'Hard-Hat', 'hard hat' and 'HARD_HAT' all become 'hardhat'."""
    return re.sub(r"[\s_\-]+", "", str(name).lower())


@dataclass(frozen=True)
class Decision:
    kind: str  # "class", "ignore" or "unknown"
    class_id: int | None = None
    class_name: str | None = None
    via: str | None = None  # the alias or ignore entry that matched


class ClassMap:
    def __init__(self, classes: list[str], aliases: dict[str, list[str]], ignore: list[str]):
        self.classes = list(classes)
        self._alias: dict[str, tuple[int, str]] = {}
        for cls, names in aliases.items():
            if cls not in self.classes:
                raise ValueError(f"alias group {cls!r} is not one of the classes {self.classes}")
            for name in [cls, *names]:
                key = normalise(name)
                if key in self._alias and self._alias[key][0] != self.classes.index(cls):
                    raise ValueError(f"label {name!r} is an alias of two classes")
                self._alias[key] = (self.classes.index(cls), name)
        self._ignore = {normalise(n): n for n in ignore}
        clash = set(self._alias) & set(self._ignore)
        if clash:
            raise ValueError(f"labels both mapped and ignored: {sorted(clash)}")

    @classmethod
    def from_yaml(cls, path: str | Path = CLASSES_FILE) -> "ClassMap":
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(cfg["classes"], cfg.get("aliases", {}), cfg.get("ignore", []))

    def decide(self, label: str) -> Decision:
        key = normalise(label)
        if key in self._alias:
            idx, via = self._alias[key]
            return Decision("class", idx, self.classes[idx], via)
        if key in self._ignore:
            return Decision("ignore", via=self._ignore[key])
        return Decision("unknown")
