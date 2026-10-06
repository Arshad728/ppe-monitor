#!/usr/bin/env python3
"""How data/caps/openimages_caps.json was made: Open Images photos of people in hats (not helmets).

Run once, in the cloud, on 28 Sep 2026 (the Mac only downloads the chosen photos: prepare_caps_data.py).
Kept so that the selection can be checked and repeated. Work folder (--work) with:
  validation-annotations-bbox.csv, test-annotations-bbox.csv      Open Images boxes (storage.googleapis.com/openimages)
  train-hat-images-bbox.csv                                       the training split's box file (2.26 GB), streamed
                                                                  through a filter that keeps hat/helmet/person rows
                                                                  of images with a hat
  validation-images-with-rotation.csv, test-images-with-rotation.csv, train-images-selected.csv
                                                                  each photo's author, licence and page (credits)

    python scripts/select_caps.py candidates --work W   # -> W/screen.json and W/urls.txt (then download the photos to W/img)
    python scripts/select_caps.py detect --work W       # the model in use and the COCO model on every photo
    python scripts/select_caps.py sheets --work W       # crops of every hat the model calls a helmet, to look at
    python scripts/select_caps.py manifest --work W     # checks, the review, splits -> data/caps/openimages_caps.json
    python scripts/select_caps.py revise --work W       # after the first training (29 Sep): a second review, and half
                                                        # the easy training photos (see revise() below)

The rules are in src/ppe_monitor/data/caps.py's description.
"""

import argparse
import collections
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from ppe_monitor.config import PROJECT_ROOT
from ppe_monitor.data.caps import MANIFEST

HAT = {"/m/02dl1y": "hat", "/m/02wbtzl": "sun hat", "/m/02fq_6": "fedora", "/m/025rp__": "cowboy hat",
       "/m/02jfl0": "sombrero"}
HELMET = {"/m/0zvk5", "/m/03p3bw", "/m/07qxg_"}                         # Helmet, Bicycle helmet, Football helmet
PERSON = {"/m/01g317", "/m/04yx4", "/m/03bt1vf", "/m/01bl7v", "/m/05r655"}   # Person, Man, Woman, Boy, Girl
BOX_FILES = (("validation", "validation-annotations-bbox.csv"), ("test", "test-annotations-bbox.csv"),
             ("train", "train-hat-images-bbox.csv"))
META_FILES = ("validation-images-with-rotation.csv", "test-images-with-rotation.csv", "train-images-selected.csv")
S3 = "https://open-images-dataset.s3.amazonaws.com"
# The share of each split, by photographer (so no test photo shows a scene training has seen)
SPLIT_SHARES = (("test", 0.20), ("val", 0.10), ("train", 0.70))
# How many photos go to the Mac: every training-split photo whose hat the model calls a helmet at
# >= HARD, as many photos it gets right, and the validation and test splits up to these sizes.
HARD = 0.10
LIMITS = {"train_hard": 700, "val": 250, "test": 450}
SEED = 28


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def on_hat(hat, box) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return (hat[0] <= cx <= hat[2] and hat[1] <= cy <= hat[3]) or iou(hat, box) > 0.3


def dedupe(items: list[dict], thr: float) -> list[dict]:
    """One hat often has two or three classes (hat + sun hat + fedora); one person has Person + Man."""
    out = []
    for it in items:
        for o in out:
            if iou(it["box"], o["box"]) > thr:
                o["kinds"] |= it["kinds"]
                break
        else:
            out.append({"box": it["box"], "kinds": set(it["kinds"])})
    return out


def candidates(work: Path) -> int:
    imgs = collections.defaultdict(lambda: {"hats": [], "persons": [], "bad": set()})
    for split, name in BOX_FILES:
        for r in csv.DictReader(open(work / name, encoding="utf-8")):
            lab = r["LabelName"]
            if lab not in HAT and lab not in HELMET and lab not in PERSON:
                continue
            d = imgs[(split, r["ImageID"])]
            box = [float(r[k]) for k in ("XMin", "YMin", "XMax", "YMax")]
            group, depiction = int(r["IsGroupOf"]), int(r["IsDepiction"])
            if lab in HELMET:
                d["bad"].add("a helmet")
            elif lab in HAT:
                if depiction or group:
                    d["bad"].add("a drawn hat, or a group of hats")
                else:
                    d["hats"].append({"box": box, "kinds": {HAT[lab]}})
            elif group:
                d["bad"].add("a crowd")
            elif depiction:
                d["bad"].add("a drawn or printed person")
            else:
                d["persons"].append({"box": box, "kinds": {lab}})
    stats, keep = collections.Counter(), []
    for (split, iid), d in sorted(imgs.items()):
        if not d["hats"]:
            continue
        stats[f"{split}: with a hat"] += 1
        if d["bad"]:
            stats["left out: " + sorted(d["bad"])[0]] += 1
            continue
        hats, persons = dedupe(d["hats"], 0.5), dedupe(d["persons"], 0.7)
        worn = []
        for h in hats:
            hb = h["box"]
            cx, cy = (hb[0] + hb[2]) / 2, (hb[1] + hb[3]) / 2
            on = [p for p in persons if p["box"][0] <= cx <= p["box"][2]
                  and p["box"][1] - 0.1 * (p["box"][3] - p["box"][1]) <= cy <= p["box"][1] + 0.35 * (p["box"][3] - p["box"][1])
                  and hb[2] - hb[0] <= 0.9 * (p["box"][2] - p["box"][0])]
            if on:
                p = min(on, key=lambda p: (p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1]))
                worn.append({"hat": hb, "kinds": sorted(h["kinds"]), "person": p["box"]})
        if not persons:
            stats["left out: no person"] += 1
        elif not worn:
            stats["left out: no hat on a head"] += 1
        elif len(worn) < len(hats):
            stats["left out: a hat not worn (held, on a table)"] += 1
        else:
            keep.append({"split": split, "id": iid, "worn": worn, "persons": [p["box"] for p in persons]})
            stats[f"{split}: kept"] += 1
    # the photos to screen: every validation and test one; from the training split, all whose hat is
    # small in the photo (as on CCTV) and a sample of the rest
    rng = random.Random(7)

    def hat_w(x):
        return min(w["hat"][2] - w["hat"][0] for w in x["worn"])
    tr = [x for x in keep if x["split"] == "train"]
    small = [x for x in tr if hat_w(x) < 0.1]
    mid = [x for x in tr if 0.1 <= hat_w(x) < 0.2]
    big = [x for x in tr if 0.2 <= hat_w(x) < 0.35]
    screen = small + rng.sample(mid, min(1900, len(mid))) + rng.sample(big, min(300, len(big))) + \
        [x for x in keep if x["split"] != "train"]
    (work / "screen.json").write_text(json.dumps(screen), encoding="utf-8")
    (work / "urls.txt").write_text("\n".join(f"{S3}/{x['split']}/{x['id']}.jpg" for x in screen) + "\n")
    print(json.dumps(stats, indent=1))
    print(f"{len(screen)} photos to screen: download them to {work / 'img'} (see urls.txt)")
    return 0


def detect(work: Path) -> int:
    from ultralytics import YOLO

    screen = json.loads((work / "screen.json").read_text())
    ids = [x["id"] for x in screen]
    for name, weights, kw in (("ppe_dets.json", PROJECT_ROOT / "models" / "ppe4_yolo26n_finetune.pt", {"conf": 0.05}),
                              ("coco_dets.json", PROJECT_ROOT / "models" / "pretrained" / "yolo26l.pt",
                               {"conf": 0.35, "classes": [0]})):
        out = json.loads((work / name).read_text()) if (work / name).is_file() else {}
        model = YOLO(str(weights))
        todo = [i for i in ids if i not in out]
        for k in range(0, len(todo), 8):
            for i, r in zip(todo[k:k + 8], model.predict([str(work / "img" / f"{i}.jpg") for i in todo[k:k + 8]],
                                                        imgsz=640, verbose=False, **kw)):
                out[i] = {"W": r.orig_shape[1], "H": r.orig_shape[0],
                          "dets": [[int(c), round(s, 4), *[round(v, 5) for v in b]] for c, s, b in
                                   zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxyn.tolist())]}
            if k % 400 == 0:
                (work / name).write_text(json.dumps(out))
        (work / name).write_text(json.dumps(out))
    return 0


def hat_confidences(screen, ppe) -> dict:
    """(image id, hat number) -> the model's highest helmet confidence on that hat."""
    out = {}
    for x in screen:
        helmets = [(e[1], e[2:]) for e in ppe[x["id"]]["dets"] if e[0] == 1]
        for k, w in enumerate(x["worn"]):
            out[(x["id"], k)] = max([s for s, b in helmets if on_hat(w["hat"], b)], default=0.0)
    return out


def sheets(work: Path) -> int:
    import cv2
    import numpy as np

    screen = json.loads((work / "screen.json").read_text())
    conf = hat_confidences(screen, json.loads((work / "ppe_dets.json").read_text()))
    hats = {(x["id"], k): w["hat"] for x in screen for k, w in enumerate(x["worn"])}
    rev = sorted((key for key, c in conf.items() if c >= 0.25), key=lambda key: -conf[key])
    (work / "review").mkdir(exist_ok=True)
    size, cols, per = 128, 10, 80
    index = []
    for s in range(0, len(rev), per):
        tiles = []
        for n, key in enumerate(rev[s:s + per], s):
            im = cv2.imread(str(work / "img" / f"{key[0]}.jpg"))
            H, W = im.shape[:2]
            h = hats[key]
            m = max((h[2] - h[0]) * W, (h[3] - h[1]) * H) * 0.8
            crop = im[int(max(0, h[1] * H - 0.6 * m)):int(min(H, h[3] * H + 1.2 * m)),
                      int(max(0, h[0] * W - m)):int(min(W, h[2] * W + m))]
            sc = (size - 18) / max(crop.shape[:2])
            crop = cv2.resize(crop, (max(1, int(crop.shape[1] * sc)), max(1, int(crop.shape[0] * sc))))
            t = np.full((size, size, 3), 255, np.uint8)
            t[:crop.shape[0], :crop.shape[1]] = crop
            cv2.putText(t, f"{n} {conf[key]:.2f}", (2, size - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 200), 1)
            tiles.append(t)
            index.append({"n": n, "id": key[0], "hat": key[1], "conf": conf[key]})
        while len(tiles) % cols:
            tiles.append(np.full((size, size, 3), 255, np.uint8))
        grid = np.vstack([np.hstack(tiles[j:j + cols]) for j in range(0, len(tiles), cols)])
        cv2.imwrite(str(work / "review" / f"sheet_{s // per:02d}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    (work / "review" / "index.json").write_text(json.dumps(index))
    print(f"{len(rev)} hats in {(len(rev) + per - 1) // per} sheets: {work / 'review'}. Write the numbers of real "
          "helmets and unclear ones into review/flags.txt, then: manifest")
    return 0


def split_of(author: str) -> str:
    """The same split for every photo by one photographer, from a hash of their Flickr page."""
    u = int(hashlib.sha256(author.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    edge = 0.0
    for name, share in SPLIT_SHARES:
        edge += share
        if u < edge:
            return name
    return SPLIT_SHARES[-1][0]


def manifest(work: Path, out: Path) -> int:
    screen = json.loads((work / "screen.json").read_text())
    ppe = json.loads((work / "ppe_dets.json").read_text())
    coco = json.loads((work / "coco_dets.json").read_text())
    meta = {}
    for name in META_FILES:
        for r in csv.DictReader(open(work / name, encoding="utf-8")):
            meta[r["ImageID"]] = r
    index = {e["n"]: e for e in json.loads((work / "review" / "index.json").read_text())}
    flagged = {index[int(t)]["id"] for line in (work / "review" / "flags.txt").read_text().splitlines()
               if not line.startswith("#") for t in line.split()}
    conf = hat_confidences(screen, ppe)
    stats, rows = collections.Counter(), []
    for x in screen:
        i, m = x["id"], meta.get(x["id"])
        dets = ppe[i]["dets"]
        people = x["persons"]
        hats = [w["hat"] for w in x["worn"]]
        found = coco[i]["dets"] if isinstance(coco[i], dict) else [[0, *e] for e in coco[i]]   # [0, conf, box]
        unlabelled = [e for e in found if max((iou(e[2:], p) for p in people), default=0) < 0.3]
        big = [p for p in people if p[3] - p[1] >= 0.15]
        not_found = [p for p in big if max((iou(e[2:], p) for e in found), default=0) < 0.3]
        why = ""
        if m is None or m.get("Rotation") not in ("", "0.0", "0"):
            why = "no credit, or the photo is stored turned"
        elif i in flagged:
            why = "a real helmet labelled as a hat, or unclear (looked at by eye)"
        elif any(e[0] == 1 and e[1] >= 0.5 and not any(on_hat(h, e[2:]) for h in hats) for e in dets):
            why = "the model is sure of a helmet away from the hats (maybe a real one, unlabelled)"
        elif any(e[0] == 2 and 0.25 <= e[1] < 0.5 for e in dets):
            why = "the model is unsure about a vest"
        elif any(e[1] < 0.6 for e in unlabelled):
            why = "a person the COCO model may see (0.35-0.6) is not labelled"
        elif big and len(not_found) > len(big) / 2:
            why = "most labelled people aren't found by the COCO model (a turned photo, or wrong boxes)"
        if why:
            stats["left out: " + why] += 1
            continue
        # people Open Images didn't box (often in the background) that the COCO model is sure of (>= 0.6)
        extra = [[0, *[round(v, 5) for v in e[2:]]] for e in unlabelled]
        stats["people added from the COCO model"] += len(extra)
        vests = [[2, *e[2:]] for e in dets if e[0] == 2 and e[1] >= 0.5]
        best = max(conf[(i, k)] for k in range(len(x["worn"])))
        rows.append({"id": i, "oi_split": x["split"], "split": split_of(m["AuthorProfileURL"] or m["OriginalLandingURL"]),
                     "labels": [[0, *[round(v, 5) for v in p]] for p in people] + extra + vests,
                     "hats": [{"box": [round(v, 5) for v in w["hat"]], "kinds": w["kinds"],
                               "person": [round(v, 5) for v in w["person"]], "helmet_conf": round(conf[(i, k)], 3)}
                              for k, w in enumerate(x["worn"])],
                     "hardest": best, "credit": {"author": m["Author"], "licence": m["License"],
                                                 "page": m["OriginalLandingURL"], "title": m["Title"]}})
        stats["checked and kept"] += 1
    rng = random.Random(SEED)
    chosen = []
    by = collections.defaultdict(list)
    for r in rows:
        by[r["split"]].append(r)
    for split in ("train", "val", "test"):
        rng.shuffle(by[split])
    hard = [r for r in by["train"] if r["hardest"] >= HARD][:LIMITS["train_hard"]]
    easy = [r for r in by["train"] if r["hardest"] < HARD][:len(hard)]
    chosen = hard + easy + by["val"][:LIMITS["val"]] + by["test"][:LIMITS["test"]]
    for r in chosen:
        path = work / "img" / f"{r['id']}.jpg"
        r["url"] = f"{S3}/{r['oi_split']}/{r['id']}.jpg"
        r["bytes"] = path.stat().st_size
        r["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        r["hardest"] = round(r["hardest"], 3)
    chosen.sort(key=lambda r: (r["split"], r["id"]))
    doc = {
        "about": "Photos of people wearing hats and caps (not helmets), from Open Images V7, for the detector's "
                 "fine-tuning. Rules and checks: src/ppe_monitor/data/caps.py; how it was made: scripts/select_caps.py.",
        "made": "2026-09-28",
        "licence": {"boxes": "CC BY 4.0, Google LLC (Open Images)",
                    "photos": "listed as CC BY 2.0 by their Flickr authors; each photo's author and page are in its credit"},
        "source": "https://storage.googleapis.com/openimages/web/index.html",
        "labels": "[class, x1, y1, x2, y2] as 0-1 fractions; 0 person (Open Images boxes, plus people Open Images "
                  "didn't box that the COCO model yolo26l is sure of, >= 0.6), 2 vest (the model in use, >= 0.5). "
                  "No helmet: none is in these photos. Hats get no label.",
        "hard": f"training photos: every one whose hat the model in use calls a helmet at >= {HARD} (up to "
                f"{LIMITS['train_hard']}), and as many it gets right",
        "counts": dict(collections.Counter(r["split"] for r in chosen)),
        "screening": dict(stats),
        "images": chosen,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=None, separators=(",", ":")), encoding="utf-8")
    print(json.dumps(doc["screening"], indent=1))
    print("chosen:", doc["counts"], f"{sum(r['bytes'] for r in chosen) / 1e6:.0f} MB", "->", out)
    return 0


def revise(work: Path, out: Path, easy: int = 350) -> int:
    """The second version of the manifest (29 Sep 2026). The first fine-tuning, on version 1, called
    far fewer hats helmets but found every colour of helmet less often (blue 91 % -> 86 %), and a navy
    helmet in the workshop clip was missed. So, without adding any photo:
    - a second review, by eye, of every hat the model in use scores 0.05-0.25 as a helmet (585 hats,
      work/review2): the photos with a real helmet labelled "Hat", or unclear, are left out;
    - the training photos whose hat the model already gets right ("easy") are cut to `easy`, so that
      fewer photos without any helmet push its helmet scores down. All the "hard" ones stay."""
    doc = json.loads(out.read_text(encoding="utf-8"))
    if doc.get("version", 1) >= 2:
        print("Already revised.")
        return 0
    index = {e["n"]: e for e in json.loads((work / "review2" / "index.json").read_text())}
    flagged = {index[int(t)]["id"] for line in (work / "review2" / "flags.txt").read_text().splitlines()
               if not line.startswith("#") for t in line.split()}
    images = [im for im in doc["images"] if im["id"] not in flagged]
    easy_train = sorted((im for im in images if im["split"] == "train" and im["hardest"] < HARD),
                        key=lambda im: hashlib.sha256(im["id"].encode()).hexdigest())
    dropped = {im["id"] for im in easy_train[easy:]}
    images = [im for im in images if im["id"] not in dropped]
    doc["images"] = images
    doc["version"] = 2
    doc["counts"] = dict(collections.Counter(im["split"] for im in images))
    doc["hard"] = (f"training photos: every one whose hat the model in use calls a helmet at >= {HARD}, and "
                   f"{easy} it gets right")
    doc["screening"]["left out in version 2: a real helmet labelled as a hat, or unclear (second review, 0.05-0.25)"] = \
        len(flagged)
    doc["screening"]["left out in version 2: training photos the model already gets right, beyond " + str(easy)] = \
        len(dropped)
    out.write_text(json.dumps(doc, indent=None, separators=(",", ":")), encoding="utf-8")
    print("version 2:", doc["counts"], f"{sum(im['bytes'] for im in images) / 1e6:.0f} MB;",
          f"{len(flagged)} left out by the second review, {len(dropped)} easy training photos left out")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["candidates", "detect", "sheets", "manifest", "revise"])
    parser.add_argument("--work", required=True, help="the work folder with the Open Images files and photos")
    parser.add_argument("--out", default=str(MANIFEST))
    args = parser.parse_args()
    work = Path(args.work)
    if args.stage == "candidates":
        return candidates(work)
    if args.stage == "detect":
        return detect(work)
    if args.stage == "sheets":
        return sheets(work)
    if args.stage == "revise":
        return revise(work, Path(args.out))
    return manifest(work, Path(args.out))


if __name__ == "__main__":
    sys.exit(main())
