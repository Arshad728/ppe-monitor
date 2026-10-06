#!/usr/bin/env python3
"""How do dwell time and error bursts trade off? Simulated workers through the real event logic.

    python scripts/simulate_events.py                     # the table in docs/phase2_results.md
    python scripts/simulate_events.py --error 0.25 --bare-error 0.15

For a worker who IS wearing their PPE, the detector's per-frame verdict is wrong `--error` of the time,
in bursts of an average length (see src/ppe_monitor/rules/simulate.py). For each burst length and
dwell time the table shows false alerts per hour per worker, and how long it takes to report a real
30-second violation. Other settings come from configs/ppe.yaml.
"""

import argparse
import dataclasses
import sys

import _bootstrap  # noqa: F401

from ppe_monitor.pipeline import PPE_CONFIG, Settings
from ppe_monitor.rules.simulate import Noise, simulate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--error", type=float, default=0.10, help="per-frame error on people wearing it")
    parser.add_argument("--bare-error", type=float, default=0.09, help="per-frame error on people not wearing it")
    parser.add_argument("--hours", type=float, default=1.0, help="simulated hours per setting")
    parser.add_argument("--config", default=str(PPE_CONFIG))
    args = parser.parse_args()
    base = Settings.load(args.config).events

    print(f"Per-frame error {args.error:.0%} (wearing) / {args.bare_error:.0%} (not wearing), 15 fps, "
          f"vote window {base.window:.1f} s, ratio {base.ratio:.1f}, cooldown {base.cooldown:.0f} s\n")
    print("| errors come in bursts of | dwell | false alerts per hour per worker | real violation reported after (median / 90%) |")
    print("|---|---|---|---|")
    for burst in (0.0, 0.5, 1.0, 2.0):
        for dwell in (2.0, 3.0, 5.0):
            s = dataclasses.replace(base, dwell=dwell)
            r = simulate(s, Noise(args.error, burst), Noise(args.bare_error, burst), compliant_hours=args.hours, episodes=100)
            label = "single frames" if burst == 0 else f"{burst:.1f} s"
            print(f"| {label} | {dwell:.0f} s | {r['false_per_hour']:.0f} | {r['median_delay']:.1f} s / {r['p90_delay']:.1f} s |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
