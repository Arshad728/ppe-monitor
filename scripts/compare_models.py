#!/usr/bin/env python3
"""Is the fine-tuned model better than the one in use? Old and new, on images neither trained on.

    python scripts/compare_models.py                  # newest in models/candidates/ against the model in use
    python scripts/compare_models.py --accept         # ... and if it passes, put it into use

Measured for both models:

1. **The Phase 1 test photos** (datasets/ppe3, test split): mAP@50 per class. The new model must
   not lose more than 0.01 here: better on new pictures must not mean worse on the old ones.
2. **The extra datasets' test photos** (datasets/ppe4, lists/test_*.txt): mAP@50 per class, per
   dataset and all together (test_extra). GDUT-HWD's people were drawn by a COCO model and it has
   no vest labels, so its person and vest figures say less than its helmet figures.
3. **Helmets by colour**: how many labelled blue / red / white / yellow helmets each model finds
   (at the rule's helmet threshold, configs/ppe.yaml), on the test photos that record the colour.
4. **The PPE rule on one frame** (Phase 2 check, ppe3 test): how often a wearer is judged
   "missing" and a bare head "worn".

The new model is accepted when (1) holds and it finds more helmets on the extra test photos
(helmet AP@50) and more blue helmets.

With `--checks feedback` (a model retrained on the dashboard's false alarms, Phase 6) the checks
are instead: (1), and no more than 0.01 lower on all the extra test photos together.

With `--checks caps` (a model fine-tuned with photos of people in hats, datasets/ppe4caps) the
checks are: (1); no more than 0.01 lower on all the extra test photos; helmets still found (helmet
AP@50 no more than 0.01 lower on the Phase 1 and on the extra test photos, and helmets of every
colour found at the rule's threshold at most 2 points less often); wearers judged
"missing" at most 1 point more often (Phase 2 rule check); on the held-out photos of people in
hats (test_caps: photographers never trained on), hat wearers judged "helmet worn" at most half as
often (and not left unjudged more often: at most 5 points more); and no more memorising than the model in use (the model card's gap between training and
validation mAP@50: at most 0.08, or the model in use's own gap if that is larger). With
`--evaluations OLD NEW` (the metrics.json of scripts/evaluate_system.py for each model, on the
same clips) two more: fewer or as many false alarms, and no more than 2 real violations missed
that the model in use caught (the retrain must not buy quiet with blindness). `--accept` then copies it into models/, which makes it the
model every script uses (the newest .pt there), and names it in configs/ppe.yaml as the model the
thresholds were checked with. The Core ML / ONNX exports must then be made again:
bash scripts/mac_phase4.sh export (mac_retrain.sh accept does both).

`--accept --override REASON` accepts a candidate that failed a check, when a person has looked at
the failure and decided it is worth it: the report, and the accepted model's card
(models/<name>.json, "accepted"), record which checks failed and the reason given.
"""

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import yaml

from ppe_monitor.config import PROJECT_ROOT
from ppe_monitor.data import caps as caps_data
from ppe_monitor.device import best_device
from ppe_monitor.pipeline import PPE_CONFIG, Settings
from ppe_monitor.rules.check import attach_poses, check_detections, load_cases, summary
from ppe_monitor.vision.detector import MODELS_DIR, PRETRAINED_DIR, load_model, predict
from ppe_monitor.vision.evaluate import evaluate
from ppe_monitor.vision.matching import iou_matrix

PPE3 = PROJECT_ROOT / "datasets" / "ppe3" / "data.yaml"
PPE4 = PROJECT_ROOT / "datasets" / "ppe4" / "data.yaml"
PPE4CAPS = PROJECT_ROOT / "datasets" / "ppe4caps" / "data.yaml"
CANDIDATES = MODELS_DIR / "candidates"
COLOURS = ("blue", "red", "white", "yellow")


def colour_recall(model, colours: dict, root: Path, threshold: float, device) -> dict:
    """{colour: (helmets, found)}: labelled helmets matched (IoU >= 0.5) by a helmet detection."""
    names = sorted(colours)
    images = [root / "images" / "test" / n for n in names]
    preds = predict(model, images, conf=threshold, device=device)
    out = {c: [0, 0] for c in COLOURS}
    for name, dets in zip(names, preds):
        gts = colours[name]
        hel = [d.box for d in dets if d.cls == 1]
        ious = iou_matrix(np.array([g[1:] for g in gts]), np.array(hel)) if hel else None
        used = set()
        for k, g in enumerate(gts):
            out[g[0]][0] += 1
            if ious is None:
                continue
            for j in np.argsort(-ious[k]):
                if ious[k, j] < 0.5:
                    break
                if j not in used:
                    used.add(j)
                    out[g[0]][1] += 1
                    break
    return out


def verdict(checks: list, override: str | None) -> tuple[bool, list[str]]:
    """(accept?, the failed checks). A failed check is accepted only with a reason from a person."""
    failed = [what for what, ok, _ in checks if not ok]
    return (not failed) or bool(override and override.strip()), failed


def mark_config(config: Path, stem: str) -> bool:
    """Point the config's `model:` line (the detector its thresholds were checked with) at `stem`."""
    text = config.read_text(encoding="utf-8")
    new = re.sub(r"(?m)^model:[ \t]*\S*", f"model: {stem}", text, count=1)
    if new == text:
        return False
    config.write_text(new, encoding="utf-8")
    return True


def rule_check(model, pose, cases, settings, device) -> dict:
    dets = predict(model, [c.image for c in cases], conf=0.01, device=device)
    s = summary(check_detections(cases, dets, settings.rules, settings.thresholds, ["person", "helmet", "vest"]))
    return {k: s[item][m] for item in ("helmet", "vest") for m, k in
            (("false_alarm", f"{item}_false_alarm"), ("missed", f"{item}_missed"))}


def caps_check(model, images, cases, settings, device) -> dict:
    """On the held-out photos of people in hats: how often a hat is called a helmet (a helmet box on
    it, at the rule's threshold), and how often the rule judges a hat wearer "helmet worn"."""
    dets = predict(model, [c.image for c in cases], conf=0.01, device=device)
    s = summary(check_detections(cases, dets, settings.rules, settings.thresholds, ["person", "helmet", "vest"]))
    by_id = {Path(c.image).stem: d for c, d in zip(cases, dets)}
    kept = [im for im in images if im["id"] in by_id]
    on_hat = caps_data.hats_called_helmets(kept, [by_id[im["id"]] for im in kept], settings.thresholds["helmet"])
    return {"wearers": s["helmet"]["people_bare"], "judged_worn": s["helmet"]["missed"],
            "unknown": s["helmet"]["unknown_bare"], **on_hat}


def measure(name: str, weights: Path, settings, cases, device, caps=None) -> dict:
    shown = weights.relative_to(PROJECT_ROOT) if weights.is_relative_to(PROJECT_ROOT) else weights
    print(f"\n== {name}: {shown}")
    model = load_model(weights)
    r = {"weights": str(shown)}
    ev = evaluate(model, PPE3, "test", imgsz=640, device=device, with_ultralytics=False)
    r["ppe3_test"] = {"map50": ev.map50, **{c.name: c.ap50 for c in ev.classes}}
    print(f"  Phase 1 test photos: mAP@50 {ev.map50:.3f} "
          f"({', '.join(f'{c.name} {c.ap50:.3f}' for c in ev.classes)})")
    if PPE4.is_file():
        cfg = yaml.safe_load(PPE4.read_text(encoding="utf-8"))
        r["extra_tests"] = {}
        for key in (k for k in cfg if k.startswith("test_") and k != "test_ppe3"):   # each extra dataset, then all
            e = evaluate(model, PPE4, key, imgsz=640, device=device, with_ultralytics=False)
            r["extra_tests"][key[5:]] = {"images": e.images, "map50": e.map50, **{c.name: c.ap50 for c in e.classes}}
            print(f"  {key[5:]:10s} test photos ({e.images}): mAP@50 {e.map50:.3f} "
                  f"({', '.join(f'{c.name} {c.ap50:.3f}' for c in e.classes)})")
        colours = json.loads((PPE4.parent / "helmet_colours.json").read_text(encoding="utf-8"))
        r["colours"] = colour_recall(model, colours, PPE4.parent, settings.thresholds["helmet"], device)
        print("  helmets found by colour: " + ", ".join(f"{c} {f}/{n} ({f / n:.0%})" for c, (n, f) in r["colours"].items() if n))
    r["rule"] = rule_check(model, None, cases, settings, device)
    print("  PPE rule on one frame: " + ", ".join(f"{k} {v:.1%}" for k, v in r["rule"].items()))
    if caps:
        r["caps"] = caps_check(model, *caps, settings, device)
        c = r["caps"]
        print(f"  people in hats ({c['wearers']} on {len(caps[1])} held-out photos): judged 'helmet worn' "
              f"{c['judged_worn']:.1%}; hats with a helmet box on them {c['called_helmet']}/{c['hats']} ({c['rate']:.1%})")
    card = weights.with_suffix(".json")
    if card.is_file():
        r["overfitting"] = json.loads(card.read_text(encoding="utf-8")).get("overfitting")
    return r


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old", help="model in use (default: newest .pt in models/)")
    parser.add_argument("--new", help="candidate (default: newest .pt in models/candidates/)")
    parser.add_argument("--accept", action="store_true", help="if the candidate passes, copy it into models/")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checks", choices=["colours", "feedback", "caps"], default="colours",
                        help="colours: the fine-tuning of decision 0007; feedback: a retrain on false alarms (Phase 6); "
                             "caps: the fine-tuning with photos of people in hats (datasets/ppe4caps)")
    parser.add_argument("--override", metavar="REASON",
                        help="with --accept: accept even though a check failed, for this reason (a person's decision; "
                             "recorded in the report and the model card)")
    parser.add_argument("--evaluations", nargs=2, metavar=("OLD", "NEW"),
                        help="metrics.json of scripts/evaluate_system.py for the model in use and the candidate")
    args = parser.parse_args()

    old = Path(args.old) if args.old else max(MODELS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    new = Path(args.new) if args.new else max(CANDIDATES.glob("*.pt"), key=lambda p: p.stat().st_mtime, default=None)
    if new is None:
        print("No candidate in models/candidates/. Train one first: bash scripts/mac_retrain.sh train")
        return 1
    old, new = (p if p.is_absolute() else PROJECT_ROOT / p for p in (old, new))
    device = best_device() if args.device == "auto" else args.device
    settings = Settings.load(PPE_CONFIG)
    cases = load_cases(PPE3, "test")
    pose = load_model(PRETRAINED_DIR / Path(settings.pose["model"]).name)
    attach_poses(cases, pose, keypoint_conf=settings.pose["keypoint_conf"], device=device)
    caps = None
    if PPE4CAPS.is_file() and caps_data.MANIFEST.is_file():
        images = caps_data.test_hats(caps_data.load_manifest(), PPE4CAPS)
        caps_cases = caps_data.cases(images)
        attach_poses(caps_cases, pose, keypoint_conf=settings.pose["keypoint_conf"], device=device)
        caps = (images, caps_cases)
    elif args.checks == "caps":
        print("No datasets/ppe4caps (or data/caps/openimages_caps.json). Run: bash scripts/mac_caps.sh data")
        return 1
    res = {"old": measure("In use", old, settings, cases, device, caps),
           "new": measure("Candidate", new, settings, cases, device, caps)}

    o, n = res["old"], res["new"]
    checks = [("not worse on the Phase 1 test photos (mAP@50 at most 0.01 lower)",
               n["ppe3_test"]["map50"] >= o["ppe3_test"]["map50"] - 0.01,
               f"{o['ppe3_test']['map50']:.3f} -> {n['ppe3_test']['map50']:.3f}")]
    if args.checks == "caps":
        if "extra_tests" in n and "extra" in n["extra_tests"]:
            oe, ne = o["extra_tests"]["extra"], n["extra_tests"]["extra"]
            checks.append((f"not worse on all the extra test photos (mAP@50 at most 0.01 lower, {ne['images']} photos)",
                           ne["map50"] >= oe["map50"] - 0.01, f"{oe['map50']:.3f} -> {ne['map50']:.3f}"))
            checks.append(("still finds helmets (helmet AP@50 at most 0.01 lower on the Phase 1 and the extra test photos)",
                           n["ppe3_test"]["helmet"] >= o["ppe3_test"]["helmet"] - 0.01 and ne["helmet"] >= oe["helmet"] - 0.01,
                           f"Phase 1 {o['ppe3_test']['helmet']:.3f} -> {n['ppe3_test']['helmet']:.3f}; "
                           f"extra {oe['helmet']:.3f} -> {ne['helmet']:.3f}"))
        ow, nw = o["rule"]["helmet_false_alarm"], n["rule"]["helmet_false_alarm"]
        checks.append(("helmet wearers judged 'missing' at most 1 point more often (Phase 2 rule check)",
                       nw <= ow + 0.01, f"{ow:.1%} -> {nw:.1%}"))
        oc, nc = o["caps"], n["caps"]
        checks.append((f"people in hats judged 'helmet worn' at most half as often, and not by judging fewer of them "
                       f"({nc['wearers']} people on held-out photos by photographers never trained on)",
                       nc["judged_worn"] <= oc["judged_worn"] / 2 and nc["unknown"] <= oc["unknown"] + 0.05,
                       f"{oc['judged_worn']:.1%} -> {nc['judged_worn']:.1%} (not judged {oc['unknown']:.1%} -> "
                       f"{nc['unknown']:.1%}); hats with a helmet box {oc['called_helmet']}/{oc['hats']} -> "
                       f"{nc['called_helmet']}/{nc['hats']}"))
        if "colours" in n:
            worst = min(((c, n["colours"][c][1] / tot - o["colours"][c][1] / tot) for c, (tot, _) in o["colours"].items()
                         if tot), key=lambda t: t[1])
            checks.append(("finds helmets of every colour as often, at the rule's threshold (at most 2 points fewer; "
                           "a navy helmet missed means a false alarm)", worst[1] >= -0.02,
                           ", ".join(f"{c} {o['colours'][c][1] / tot:.0%} -> {n['colours'][c][1] / tot:.0%}"
                                     for c, (tot, _) in o["colours"].items() if tot)))
        og = (o.get("overfitting") or {}).get("gap") or 0.0
        ng = (n.get("overfitting") or {}).get("gap")
        line = max(0.08, og)
        checks.append((f"no more memorising than the model in use (training minus validation mAP@50 at most {line:.3f})",
                       ng is not None and ng <= line and not (n.get("overfitting") or {}).get("later_epochs_memorising"),
                       f"{og:+.3f} -> " + (f"{ng:+.3f}" if ng is not None else "no model card")))
    elif args.checks == "feedback":
        if "extra_tests" in n and "extra" in n["extra_tests"]:
            oe, ne = o["extra_tests"]["extra"]["map50"], n["extra_tests"]["extra"]["map50"]
            checks.append((f"not worse on all the extra test photos (mAP@50 at most 0.01 lower, "
                           f"{n['extra_tests']['extra']['images']} photos)", ne >= oe - 0.01, f"{oe:.3f} -> {ne:.3f}"))
    elif "extra_tests" in n and "extra" in n["extra_tests"]:
        oh, nh = o["extra_tests"]["extra"]["helmet"], n["extra_tests"]["extra"]["helmet"]
        checks.append((f"finds helmets better on the extra datasets' test photos (helmet AP@50, "
                       f"all {n['extra_tests']['extra']['images']} together)", nh > oh, f"{oh:.3f} -> {nh:.3f}"))
        ob, nb = o["colours"]["blue"], n["colours"]["blue"]
        if ob[0]:
            checks.append(("finds more blue helmets", nb[1] > ob[1], f"{ob[1]}/{ob[0]} -> {nb[1]}/{nb[0]}"))
    if args.evaluations:
        eo, en = (json.loads(Path(x).read_text(encoding="utf-8"))["summaries"]["all"] for x in args.evaluations)
        checks.append(("no more false alarms on the test clips (end to end)", en["false"] <= eo["false"],
                       f"{eo['false']} -> {en['false']} ({eo['false_per_hour']:.1f} -> {en['false_per_hour']:.1f} per camera-hour)"))
        checks.append(("misses at most 2 more real violations than the model in use", en["caught"] >= eo["caught"] - 2,
                       f"caught {eo['caught']}/{eo['sustained']} -> {en['caught']}/{en['sustained']}"))
    passed = all(ok for _, ok, _ in checks)
    accept_ok, failed = verdict(checks, args.override if args.accept else None)

    out = PROJECT_ROOT / "runs" / "compare" / f"{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"# Old model against the fine-tuned candidate: {datetime.now():%Y-%m-%d %H:%M}", "",
             f"- In use: `{o['weights']}`", f"- Candidate: `{n['weights']}`", "", "## Verdict", "",
             "| Check | Result | Detail |", "|---|---|---|"]
    lines += [f"| {what} | {'PASS' if ok else 'FAIL'} | {detail} |" for what, ok, detail in checks]
    lines += ["", f"**{'Accept' if passed else 'Keep the model in use'}.**" if passed or not accept_ok else
              f"**Failed {len(failed)} check(s); accepted with an override:** {args.override}", "",
              "## mAP@50 on test photos neither model trained on", "",
              "| Test photos | Model | mAP@50 | person | helmet | vest |", "|---|---|---|---|---|---|"]
    rows = [("Phase 1 (ppe3)", "ppe3_test")]
    for name, key in rows:
        for who in ("old", "new"):
            v = res[who][key]
            lines.append(f"| {name} | {who} | {v['map50']:.3f} | {v['person']:.3f} | {v['helmet']:.3f} | {v['vest']:.3f} |")
    for src in n.get("extra_tests", {}):
        for who in ("old", "new"):
            v = res[who]["extra_tests"][src]
            lines.append(f"| {src} ({v['images']}) | {who} | {v['map50']:.3f} | {v['person']:.3f} | {v['helmet']:.3f} | "
                         f"{v['vest']:.3f} |")
    if "colours" in n:
        lines += ["", f"## Helmets found, by colour (at the rule's threshold {settings.thresholds['helmet']})", "",
                  "| Colour | Helmets | Old | New |", "|---|---|---|---|"]
        for c in COLOURS:
            (tot, fo), (_, fn) = o["colours"][c], n["colours"][c]
            if tot:
                lines.append(f"| {c} | {tot} | {fo / tot:.0%} | {fn / tot:.0%} |")
    if "caps" in n:
        oc, nc = o["caps"], n["caps"]
        lines += ["", f"## People in hats and caps (held-out Open Images photos, {nc['wearers']} people, {nc['hats']} hats)", "",
                  "| | Old | New |", "|---|---|---|",
                  f"| Hat wearer judged 'helmet worn' (the rule, one frame) | {oc['judged_worn']:.1%} | {nc['judged_worn']:.1%} |",
                  f"| Hat with a helmet box on it (threshold {settings.thresholds['helmet']}) | {oc['rate']:.1%} | {nc['rate']:.1%} |",
                  f"| Hat wearer not judged (too small, head hidden, or not found) | {oc['unknown']:.1%} | {nc['unknown']:.1%} |"]
    if n.get("overfitting"):
        of = n["overfitting"]
        lines += ["", "## Overfitting (from the candidate's model card)", "",
                  f"- Kept epoch {of['best_epoch']} of {of['epochs']}; validation loss {of['val_loss_best']:.3f} there, "
                  f"{of['val_loss_last']:.3f} at the last epoch.",
                  f"- mAP@50 on training images {of['train_map50_sample']:.3f}, on validation {of['val_map50']:.3f} "
                  f"(gap {of['gap']:+.3f}; the model in use: {(o.get('overfitting') or {}).get('gap', float('nan')):+.3f})."]
    lines += ["", "## The PPE rule on one frame (Phase 1 test photos)", "", "| | Old | New |", "|---|---|---|"]
    labels = {"helmet_false_alarm": "Helmet: wearer judged missing", "helmet_missed": "Helmet: bare head judged worn",
              "vest_false_alarm": "Vest: wearer judged missing", "vest_missed": "Vest: no vest judged worn"}
    for k, label in labels.items():
        lines.append(f"| {label} | {o['rule'][k]:.1%} | {n['rule'][k]:.1%} |")
    report = "\n".join(lines) + "\n"
    (out / "report.md").write_text(report, encoding="utf-8")
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print("\n" + "\n".join(lines[:10 + len(checks)]))
    print(f"\nFull report: {out / 'report.md'}")

    if args.accept:
        if not accept_ok:
            print("\nNot accepted: the candidate did not pass every check. The model in use stays.")
            return 1
        if not passed:
            print(f"\nAccepted although {len(failed)} check(s) failed, with the reason given: {args.override}")
        dest = MODELS_DIR / new.name
        for suffix in (".pt", ".md", ".json"):
            if new.with_suffix(suffix).is_file():
                shutil.copy2(new.with_suffix(suffix), dest.with_suffix(suffix))
        card = dest.with_suffix(".json")
        if card.is_file():
            info = json.loads(card.read_text(encoding="utf-8"))
            info["accepted"] = {"when": f"{datetime.now():%Y-%m-%d %H:%M}", "report": str(out.relative_to(PROJECT_ROOT)),
                                "replaced": old.stem, "failed_checks": failed, "override": args.override if failed else None}
            card.write_text(json.dumps(info, indent=1, default=float), encoding="utf-8")
        dest.touch()                                   # newest .pt in models/ = the model every script uses
        print(f"\nAccepted: {dest.relative_to(PROJECT_ROOT)} is now the model in use. "
              "Make its Core ML / ONNX versions: bash scripts/mac_phase4.sh export")
        if mark_config(PPE_CONFIG, dest.stem):
            print(f"{PPE_CONFIG.relative_to(PROJECT_ROOT)} now says its thresholds are for {dest.stem} "
                  "(re-checked on the validation and test photos: docs/finetune_results.md).")
    return 0 if passed or (args.accept and accept_ok) else 2


if __name__ == "__main__":
    sys.exit(main())
