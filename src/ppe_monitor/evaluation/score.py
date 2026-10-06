"""Matching the alerts the system sent to the ground truth, and the five headline numbers.

Each alert is matched to at most one ground-truth episode of its kind, on the same clip:
  - **when:** the event was confirmed between `before` s before the episode began and `after` s
    after it ended (the event engine can confirm a little after the person has gone, and a clip's
    first frames have no tracks yet);
  - **where:** when both boxes are known, the event's person box overlaps the episode's box at that
    moment (IoU >= `min_iou`), or its centre lies inside it.

The best fit wins (the highest overlap); among episodes that fit about as well, a violation comes
before a "don't care" episode, which comes before a compliant person.

Then, per alert: **true** (the first alert on a sustained episode), **duplicate** (a later alert on
the same episode: right, but one too many), **unscored** (on a not-sustained episode, on nobody the
truth knows when it isn't complete, or of a kind this clip's truth doesn't cover) or **false** (on
a person known to be compliant, or on nobody when the truth is complete: a false alarm).
Per sustained episode: **caught** (at least one alert) or **missed**.

Precision counts true and duplicate alerts as right: each is about a real violation.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

from ..rules.scoring import iou
from .truth import ClipTruth, Episode


@dataclass
class AlertRecord:
    """One alert that reached the safety officer, with its times in seconds from the clip's first frame."""
    event_id: str
    kind: str
    started: float              # when the system saw the violation begin
    confirmed: float            # when it became an event (after the dwell)
    delivered: float | None     # when the message was handed over (Telegram answered); None: never
    box: tuple | None = None    # the person's box when confirmed, 0-1 fractions
    real_channel: bool = False  # delivered by the real Telegram, not the stand-in


@dataclass
class AlertOutcome:
    alert: AlertRecord
    verdict: str                # true / duplicate / unscored / false
    episode: int | None = None  # index into truth.episodes
    overlap: float = 0.0


@dataclass
class ClipResult:
    truth: ClipTruth
    camera: str
    alerts: list[AlertOutcome]
    caught: dict[int, AlertRecord]        # sustained episode index -> its first alert
    missed: list[int]
    events_stored: int = 0                # events in the database for this camera (with or without an alert)
    note: str = ""

    def count(self, verdict: str) -> int:
        return sum(1 for a in self.alerts if a.verdict == verdict)


def _where(ep: Episode, a: AlertRecord, min_iou: float) -> float | None:
    """How well the alert's box fits the episode's person: IoU, 1.0 when a box is missing on either
    side (only time can be checked), None when they don't fit."""
    gt = ep.box_at(a.confirmed)
    if gt is None or a.box is None:
        return 1.0
    o = iou(gt, a.box)
    cx, cy = (a.box[0] + a.box[2]) / 2, (a.box[1] + a.box[3]) / 2
    if o >= min_iou or (gt[0] <= cx <= gt[2] and gt[1] <= cy <= gt[3]):
        return max(o, min_iou)
    return None


def match_clip(truth: ClipTruth, camera: str, alerts: list[AlertRecord], *, before: float = 1.0,
               after: float = 4.0, min_iou: float = 0.3, events_stored: int = 0) -> ClipResult:
    outcomes: list[AlertOutcome] = []
    caught: dict[int, AlertRecord] = {}
    for a in sorted(alerts, key=lambda x: x.confirmed):
        if a.kind not in truth.checked:
            outcomes.append(AlertOutcome(a, "unscored"))
            continue
        fits = []
        for i, ep in enumerate(truth.episodes):
            if ep.kind != a.kind or not (ep.start - before <= a.confirmed <= ep.end + after):
                continue
            fit = _where(ep, a, min_iou)
            if fit is not None:
                rank = 0 if ep.compliant else 2 if ep.sustained else 1
                fits.append((fit, rank, i))
        if not fits:
            outcomes.append(AlertOutcome(a, "false" if truth.complete else "unscored"))
            continue
        top = max(f for f, _, _ in fits)
        fit, _, best = max((x for x in fits if x[0] >= top - 0.1), key=lambda x: (x[1], x[0]))
        ep = truth.episodes[best]
        if ep.compliant:
            outcomes.append(AlertOutcome(a, "false", best, fit))
        elif not ep.sustained:
            outcomes.append(AlertOutcome(a, "unscored", best, fit))
        elif best in caught:
            outcomes.append(AlertOutcome(a, "duplicate", best, fit))
        else:
            caught[best] = a
            outcomes.append(AlertOutcome(a, "true", best, fit))
    missed = [i for i, ep in enumerate(truth.episodes) if ep.sustained and not ep.compliant and i not in caught]
    return ClipResult(truth, camera, outcomes, caught, missed, events_stored)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % confidence interval of a proportion k/n (Wilson score interval): honest error bars for
    the small counts a test set gives."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def poisson_upper(k: int, tail: float = 0.025) -> float:
    """The exact upper end of the 95 % confidence interval of a Poisson count k: the rate at which
    seeing k or fewer would happen only 2.5 % of the time. For a rate measured from few events
    (0 false alarms in 1 hour is not "never": the upper end is 3.7 per hour)."""
    def cdf(lam: float) -> float:
        term = total = math.exp(-lam)
        for i in range(1, k + 1):
            term *= lam / i
            total += term
        return total
    lo, hi = 0.0, float(k) + 10 * math.sqrt(k + 1) + 10
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cdf(mid) > tail else (lo, mid)
    return hi


@dataclass
class Summary:
    clips: int = 0
    hours: float = 0.0                      # footage evaluated (clip time, per camera)
    sustained: int = 0
    caught: int = 0
    alerts: int = 0                         # delivered alerts, all verdicts
    true: int = 0
    duplicate: int = 0
    false: int = 0
    unscored: int = 0
    events_stored: int = 0
    latency: list = field(default_factory=list)       # s, violation start -> alert delivered
    to_event: list = field(default_factory=list)      # s, violation start -> event confirmed
    to_send: list = field(default_factory=list)       # s, event confirmed -> alert delivered
    real_latency: list = field(default_factory=list)  # the same as `latency`, real Telegram only
    real_to_send: list = field(default_factory=list)

    def add(self, r: ClipResult) -> None:
        self.clips += 1
        self.hours += r.truth.duration / 3600
        self.sustained += len(r.truth.sustained)
        self.caught += len(r.caught)
        self.alerts += len(r.alerts)
        for v in ("true", "duplicate", "false", "unscored"):
            setattr(self, v, getattr(self, v) + r.count(v))
        self.events_stored += r.events_stored
        for i, a in r.caught.items():
            ep = r.truth.episodes[i]
            self.to_event.append(a.confirmed - ep.start)
            if a.delivered is not None:
                self.latency.append(a.delivered - ep.start)
                self.to_send.append(a.delivered - a.confirmed)
                if a.real_channel:
                    self.real_latency.append(a.delivered - ep.start)
                    self.real_to_send.append(a.delivered - a.confirmed)

    @property
    def precision(self) -> float:
        n = self.true + self.duplicate + self.false
        return (self.true + self.duplicate) / n if n else float("nan")

    @property
    def recall(self) -> float:
        return self.caught / self.sustained if self.sustained else float("nan")

    @property
    def false_per_hour(self) -> float:
        return self.false / self.hours if self.hours else float("nan")

    def as_dict(self) -> dict:
        def stats(v):
            if not v:
                return None
            s = sorted(v)
            return {"n": len(s), "median": round(statistics.median(s), 2),
                    "p90": round(s[min(len(s) - 1, int(0.9 * len(s)))], 2), "max": round(s[-1], 2)}
        scored = self.true + self.duplicate + self.false
        return {"clips": self.clips, "hours": round(self.hours, 4), "sustained": self.sustained, "caught": self.caught,
                "recall": self.recall, "recall_ci": wilson(self.caught, self.sustained),
                "alerts": self.alerts, "true": self.true, "duplicate": self.duplicate, "false": self.false,
                "unscored": self.unscored, "precision": self.precision,
                "precision_ci": wilson(self.true + self.duplicate, scored),
                "false_per_hour": self.false_per_hour,
                "false_per_hour_ci": ((poisson_upper(self.false - 1, 0.975) if self.false else 0.0) / self.hours,
                                      poisson_upper(self.false) / self.hours) if self.hours else None,
                "events_stored": self.events_stored, "latency": stats(self.latency),
                "to_event": stats(self.to_event), "to_send": stats(self.to_send),
                "real_latency": stats(self.real_latency), "real_to_send": stats(self.real_to_send)}
