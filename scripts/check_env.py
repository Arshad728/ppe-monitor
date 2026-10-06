#!/usr/bin/env python3
"""Check and record this machine's development environment.

    python scripts/check_env.py            # print the report
    python scripts/check_env.py --write    # also save it to docs/ENVIRONMENT.md

Exits with code 1 if anything Phase 0 needs is missing.
"""

import argparse
import sys

import _bootstrap  # noqa: F401  (makes `ppe_monitor` importable)

from ppe_monitor.config import PROJECT_ROOT
from ppe_monitor.env_check import collect, format_markdown, format_table, missing_required, recommended_device


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="save the report to docs/ENVIRONMENT.md")
    args = parser.parse_args()

    checks = collect()
    print(format_table(checks))
    print(f"\nRecommended compute device: {recommended_device(checks)}")

    if args.write:
        out = PROJECT_ROOT / "docs" / "ENVIRONMENT.md"
        out.write_text(format_markdown(checks), encoding="utf-8")
        print(f"Saved {out.relative_to(PROJECT_ROOT)}")

    missing = missing_required(checks)
    if missing:
        print("\nMissing for Phase 0:")
        for c in missing:
            print(f"  - {c.name}: {c.note or 'install it'}")
        return 1
    print("\nEverything Phase 0 needs is installed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
