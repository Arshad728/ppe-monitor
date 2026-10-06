#!/usr/bin/env python3
"""Photos of people in hats and caps (not helmets) -> datasets/ppe4caps, for the next fine-tuning.

    python scripts/prepare_caps_data.py          # ~5-10 min: download ~2,000 photos (~600 MB), build the dataset
    bash scripts/mac_caps.sh data                # the same, on the Mac

The photos come from Open Images (Google; photos by Flickr authors under CC BY 2.0), chosen and
checked as src/ppe_monitor/data/caps.py describes. Each is downloaded from Open Images' own storage
and refused if its SHA-256 differs from data/caps/openimages_caps.json. Then datasets/ppe4caps is
datasets/ppe4 exactly, plus:
  - training:   the manifest's `train` photos (about half with a hat the model in use calls a helmet);
  - validation: its `val` photos, so that training stops when hats start being called helmets again;
  - test_caps:  its `test` photos, a separate list, never trained on: how often a model calls a hat
                a helmet (scripts/compare_models.py --checks caps).
Credits for every photo used: datasets/ppe4caps/credits.csv. Report: datasets/ppe4caps/report.md.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401

from ppe_monitor.config import PROJECT_ROOT
from ppe_monitor.data.caps import MANIFEST, RAW, build, fetch, load_manifest

BASE = PROJECT_ROOT / "datasets" / "ppe4" / "data.yaml"
OUT = PROJECT_ROOT / "datasets" / "ppe4caps"


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=str(BASE), help="the dataset to add to (default datasets/ppe4)")
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--manifest", default=str(MANIFEST))
    args = parser.parse_args()
    base, out = Path(args.base), Path(args.out)
    if not base.is_file():
        print(f"No base dataset {base}. On the Mac it is made by: bash scripts/mac_retrain.sh data")
        return 1
    manifest = load_manifest(Path(args.manifest))
    print(f"{len(manifest['images'])} photos in {Path(args.manifest).name}: {manifest['counts']}")
    failed = fetch(manifest, RAW)
    counts = build(base, manifest, out, RAW, skip=set(failed))
    per = {s: sum(1 for im in manifest["images"] if im["split"] == s) for s in ("train", "val", "test")}
    from ppe_monitor.data.caps import test_hats
    hats = sum(len(im["hats"]) for im in test_hats(manifest, out / "data.yaml"))
    lines = [
        f"# Training dataset report: {out.name} ({base.parent.name} + people in hats and caps), "
        f"{datetime.now():%Y-%m-%d %H:%M}", "",
        f"Built from `{base}` (taken as it is) and the Open Images photos listed in `data/caps/openimages_caps.json` "
        "(how they were chosen and checked: `src/ppe_monitor/data/caps.py`).", "",
        "## Sources", "",
        f"- Everything in datasets/{base.parent.name} (see its report.md), unchanged.",
        f"- **Open Images V7**, people wearing hats and caps: boxes {manifest['licence']['boxes']}; photos "
        f"{manifest['licence']['photos']}. The author and page of every photo used: `credits.csv`.", "",
        "## What was added", "",
        "| Split | In the manifest | Added | Base photos |", "|---|---|---|---|",
        f"| train | {per['train']} | {counts['added']['train']} | {counts['base'].get('train', 0)} |",
        f"| val | {per['val']} | {counts['added']['val']} | {counts['base'].get('val', 0)} |",
        f"| test_caps (separate list; base test lists unchanged) | {per['test']} | {counts['added']['test']} | - |", "",
        f"- Photos that couldn't be downloaded, or whose checksum differed: {len(failed)}.",
        "- Left out because they look like a photo the models are tested on: "
        + ", ".join(f"{k}: {v}" for k, v in counts["left_out"].items()) + ".",
        f"- The test photos used show {hats} hats. Labels: people (Open Images' boxes, and people it didn't box that "
        "the COCO model is sure of) and vests the model in use is sure of. Hats have no label, so to the detector "
        "they are background.", "",
        "Next: `bash scripts/mac_caps.sh train`.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # the sources and their licences, for the model card (train_detector.py reads report.json)
    try:
        sources = json.loads((base.parent / "report.json").read_text(encoding="utf-8"))["sources"]
    except (OSError, ValueError, KeyError):
        sources = []
    sources.append({"name": "openimages-caps", "title": "Open Images V7: people in hats and caps (data/caps/)",
                    "licence": f"boxes {manifest['licence']['boxes']}; photos CC BY 2.0 (Flickr authors, see credits.csv)",
                    "attribution": "A. Kuznetsova et al., \"The Open Images Dataset V4\", IJCV 2020; "
                                   "https://storage.googleapis.com/openimages/web/index.html. Photo credits: "
                                   f"datasets/{out.name}/credits.csv"})
    (out / "report.json").write_text(json.dumps({"sources": sources, "caps": counts}, indent=1), encoding="utf-8")
    print("\n".join(lines[9:17]))
    print(f"\nDataset: {out / 'data.yaml'}\nReport:  {out / 'report.md'}")
    (out / "manifest_used.json").write_text(json.dumps({"manifest": str(args.manifest), "failed": failed}, indent=1))
    return 0 if counts["added"]["train"] and counts["added"]["test"] else 1


if __name__ == "__main__":
    sys.exit(main())
