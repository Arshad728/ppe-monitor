"""Stress-testing the event logic with simulated detector mistakes.

A worker's per-frame verdicts are generated with a chosen error rate, then fed through the real
EventEngine. The question is how often it raises a false alarm on someone wearing their PPE, and
how quickly it catches a real violation.

Detector mistakes on video are rarely independent from frame to frame: a worker who turns so their
vest is hidden stays turned for a while. So errors come in bursts. A two-state (Gilbert-Elliott)
chain switches between "good" and "bad" frames. The average error rate and the average burst length
are set separately, and a bad frame gives the wrong answer.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .events import EventEngine, EventSettings
from .ppe import MISSING, UNKNOWN, WORN, Verdict

BOX = (0.40, 0.30, 0.50, 0.70)


@dataclass(frozen=True)
class Noise:
    error: float          # share of frames with the wrong answer
    burst_seconds: float  # average length of a run of wrong answers
    unknown: float = 0.0  # share of frames with no answer (too small, cut off)


class _Chain:
    def __init__(self, noise: Noise, fps: float, rng: random.Random):
        self.rng, self.noise = rng, noise
        burst_frames = max(1.0, noise.burst_seconds * fps)
        self.p_stay_bad = 1 - 1 / burst_frames
        # stationary share of bad frames = error: p_enter / (p_enter + p_leave) = error
        p_leave = 1 - self.p_stay_bad
        self.p_enter = min(1.0, noise.error * p_leave / max(1e-9, 1 - noise.error))
        self.bad = rng.random() < noise.error

    def step(self) -> bool:
        self.bad = self.rng.random() < (self.p_stay_bad if self.bad else self.p_enter)
        return self.bad


def simulate(settings: EventSettings, noise_wearing: Noise, noise_bare: Noise, *, fps: float = 15,
             compliant_hours: float = 2.0, episodes: int = 200, episode_seconds: float = 30.0,
             seed: int = 0) -> dict:
    """False events per hour on compliant workers, and detection of violations of a given length."""
    rng = random.Random(seed)

    # 1. compliant workers: every event is a false alarm
    engine = EventEngine("sim", settings)
    chain = _Chain(noise_wearing, fps, rng)
    frames = int(compliant_hours * 3600 * fps)
    for k in range(frames):
        if rng.random() < noise_wearing.unknown:
            answer = UNKNOWN
        else:
            answer = MISSING if chain.step() else WORN
        engine.update(k / fps, k, [(1, BOX, Verdict(answer, WORN))])
    false_per_hour = len(engine.events) / compliant_hours

    # 2. violations: 10 s compliant, then `episode_seconds` without a helmet, one new person each time.
    # Episodes where a false alarm fired before the violation began are left out here: its cooldown
    # would hide the real one, and false alarms are already measured in part 1.
    caught, extra, delays, counted = 0, 0, [], 0
    for e in range(episodes):
        engine = EventEngine("sim", settings)
        good, bad = _Chain(noise_wearing, fps, rng), _Chain(noise_bare, fps, rng)
        good.bad = False
        start = 10.0
        for k in range(int((start + episode_seconds) * fps)):
            t = k / fps
            wearing = t < start
            chain = good if wearing else bad
            if rng.random() < (noise_wearing if wearing else noise_bare).unknown:
                answer = UNKNOWN
            elif wearing:
                answer = MISSING if chain.step() else WORN
            else:
                answer = WORN if chain.step() else MISSING
            engine.update(t, k, [(1, BOX, Verdict(answer, WORN))])
        if any(ev.confirmed < start for ev in engine.events):
            continue
        counted += 1
        during = [ev for ev in engine.events if ev.confirmed >= start]
        if during:
            caught += 1
            delays.append(during[0].confirmed - start)
            extra += len(during) - 1
    delays.sort()
    return {"false_per_hour": false_per_hour, "caught": caught / counted if counted else float("nan"),
            "duplicates": extra,
            "median_delay": delays[len(delays) // 2] if delays else float("nan"),
            "p90_delay": delays[int(len(delays) * 0.9)] if delays else float("nan")}
