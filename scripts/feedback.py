#!/usr/bin/env python3
"""The false alarms marked on the dashboard, as training data for the next model (Phase 6).

    python scripts/feedback.py export      # false alarms from the database -> datasets/feedback/
    python scripts/feedback.py status      # what is there, and what is ready
    python scripts/feedback.py build       # datasets/ppe5 = the model in use's training data + the ready ones (training only)
    bash scripts/mac_phase6.sh retrain     # all of it, then fine-tune, compare and evaluate

`export` looks at every event marked "false alarm" whose clean frame is still kept (images of false
alarms are never deleted), from cameras whose footage may be trained on (not the test clips; see
src/ppe_monitor/data/feedback.py). It labels each frame with what the model in use finds, then
adds what the verdict says the model missed: the helmet on the head of a person wrongly alerted as
"no helmet", found again at a lower confidence. The result:

    datasets/feedback/images/<event>.jpg     the clean frames
    datasets/feedback/labels/<event>.txt     their labels (YOLO format)
    datasets/feedback/feedback.json          each one's status: ready, needs_box, review, not_needed, excluded
    datasets/feedback/review.html            every frame with its boxes; the orange ones are the lessons added

Look at review.html before `build`. To leave an image out, put its event id in
datasets/feedback/skip.txt (one per line). Running `export` again adds new false alarms and keeps
what is already there (--relabel labels everything again).
"""

import argparse
import base64
import html
import shutil
import sys
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401

import cv2

from ppe_monitor.config import PROJECT_ROOT, ConfigError, load_cameras
from ppe_monitor.data.feedback import (FEEDBACK_DIR, FeedbackItem, base_dataset, build_dataset, draw_review,
                                       eligible_camera, label_false_alarm, load_catalog, load_index, save_index,
                                       skipped, write_label_file)
from ppe_monitor.pipeline import PPE_CONFIG, Settings

STATUS_TEXT = {"ready": "ready to train on", "needs_box": "needs a box drawn by hand", "review": "for a person to look at",
               "not_needed": "not needed (the detector was right)", "excluded": "not allowed (test footage)"}


def open_store(args):
    from ppe_monitor.backend import localpg
    from ppe_monitor.backend.settings import ServerSettings
    from ppe_monitor.backend.store import EventStore

    s = ServerSettings.load()
    url = args.database_url or s.database_url
    if not args.database_url and localpg.in_use(s):
        localpg.start()
    return EventStore(url, Path(args.storage) if args.storage else s.storage_dir, create=False)


def export(args) -> int:
    from sqlalchemy import select

    from ppe_monitor.backend.db import cameras as cameras_t
    from ppe_monitor.backend.db import events as events_t
    from ppe_monitor.vision.detector import MODELS_DIR, load_model, predict

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "review").mkdir(parents=True, exist_ok=True)
    store = open_store(args)
    with store.engine.connect() as c:
        rows = c.execute(select(events_t).where(events_t.c.status == "false_alarm").order_by(events_t.c.confirmed_at)).all()
        names = {r.id: r.name for r in c.execute(select(cameras_t.c.id, cameras_t.c.name))}
    print(f"{len(rows)} event(s) marked \"false alarm\" in {store.engine.dialect.name}")
    catalog = load_catalog()
    try:
        simulated = {cam.id for cam in load_cameras() if cam.sim_source}
    except ConfigError:
        simulated = set()
    old = {i.event_id: i for i in load_index(out)} if not args.relabel else {}
    settings = Settings.load(PPE_CONFIG)
    weights = max(MODELS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    model = None
    items = []
    for r in rows:
        if r.id in old:
            items.append(old[r.id])
            continue
        item = FeedbackItem(r.id, r.camera, r.kind, r.confirmed_at.isoformat(timespec="seconds"), list(r.box or []),
                            note=r.note or "")
        ok, why = eligible_camera(r.camera, names.get(r.camera, ""), catalog, simulated)
        if not ok and not args.include_test:
            item.status, item.reason = "excluded", why
            items.append(item)
            continue
        frame = store.image_path(r.id, "frame")
        if frame is None:
            item.status, item.reason = "excluded", "its clean frame is no longer kept"
            items.append(item)
            continue
        if model is None:
            model = load_model(weights)
            print(f"Labelling with the model in use: {weights.name}")
        dets = predict(model, [frame], conf=0.01)[0]
        label_false_alarm(item, dets, settings.thresholds, settings.rules)
        if not ok:
            item.reason += f" (TEST FOOTAGE, --include-test: {why})"
        shutil.copy2(frame, out / "images" / f"{r.id}.jpg")
        write_label_file(out / "labels" / f"{r.id}.txt", item.labels)
        img = cv2.imread(str(frame))
        if img is not None:
            cv2.imwrite(str(out / "review" / f"{r.id}.jpg"), draw_review(img, item), [cv2.IMWRITE_JPEG_QUALITY, 80])
        items.append(item)
    save_index(out, items)
    write_review(out, items)
    store.close()
    show_status(out, items)
    print(f"\nLook at {out / 'review.html'} before building.")
    return 0


def write_review(out: Path, items: list[FeedbackItem]) -> None:
    skip = skipped(out)
    cards = []
    for i in items:
        thumb = out / "review" / f"{i.event_id}.jpg"
        img = (f'<img src="data:image/jpeg;base64,{base64.b64encode(thumb.read_bytes()).decode()}" alt="">'
               if thumb.is_file() else "<div class=none>no picture</div>")
        state = "skipped (skip.txt)" if i.event_id in skip else STATUS_TEXT.get(i.status, i.status)
        cards.append(f'<figure class="{html.escape(i.status)}">{img}<figcaption><b>{html.escape(i.kind)}</b> · '
                     f'{html.escape(i.camera)} · {html.escape(i.when[:16].replace("T", " "))}<br>'
                     f'<span class=st>{html.escape(state)}</span>: {html.escape(i.reason)}'
                     + (f"<br><i>{html.escape(i.note)}</i>" if i.note else "")
                     + f'<br><code>{html.escape(i.event_id)}</code></figcaption></figure>')
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>False alarms to learn from</title><style>
body{{font:14px/1.4 system-ui,sans-serif;margin:16px;background:#f6f6f4;color:#222}}
h1{{font-size:20px}} .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}}
figure{{margin:0;background:#fff;border:1px solid #ddd;border-radius:6px;overflow:hidden}}
figure img{{width:100%;display:block}} figcaption{{padding:8px}} code{{font-size:11px;color:#666}}
.ready .st{{color:#0a7a2f;font-weight:600}} .needs_box .st,.review .st{{color:#b35c00;font-weight:600}}
.excluded,.not_needed{{opacity:.55}} .none{{padding:40px;text-align:center;color:#999}}
@media (prefers-color-scheme:dark){{body{{background:#1b1b1b;color:#ddd}} figure{{background:#262626;border-color:#333}}}}
</style></head><body><h1>False alarms to learn from ({len(items)})</h1>
<p>Grey boxes: what the model in use finds. <b style="color:#e8750a">Orange</b>: what the verdict adds (the item the
model missed). Red outline: the person the alert was about. To leave one out, add its id to <code>skip.txt</code>.
Written {datetime.now():%Y-%m-%d %H:%M} by scripts/feedback.py.</p>
<div class=grid>{''.join(cards)}</div></body></html>"""
    (out / "review.html").write_text(page, encoding="utf-8")


def show_status(out: Path, items: list[FeedbackItem] | None = None) -> None:
    items = items if items is not None else load_index(out)
    skip = skipped(out)
    counts = {}
    for i in items:
        key = "skipped" if i.event_id in skip else i.status
        counts[key] = counts.get(key, 0) + 1
    print(f"\n{out}: {len(items)} false alarm(s)")
    for key in ("ready", "needs_box", "review", "not_needed", "excluded", "skipped"):
        if counts.get(key):
            print(f"  {counts[key]:4d}  {STATUS_TEXT.get(key, key)}")
    reasons = {}
    for i in items:
        if i.status == "excluded":
            reasons[i.reason] = reasons.get(i.reason, 0) + 1
    for r, n in sorted(reasons.items(), key=lambda x: -x[1])[:5]:
        print(f"        {n} x {r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["export", "status", "build"])
    parser.add_argument("--out", default=str(FEEDBACK_DIR), help="the feedback folder (default datasets/feedback)")
    parser.add_argument("--database-url", help="read another database (e.g. the evaluation's) instead of the live one")
    parser.add_argument("--storage", help="with --database-url: where that database's images are")
    parser.add_argument("--include-test", action="store_true",
                        help="also take false alarms from test footage: ONLY to try the tools, never to train a model you keep")
    parser.add_argument("--relabel", action="store_true", help="label every false alarm again")
    parser.add_argument("--base", help="build: the dataset to add to (default: the one the model in use was trained on, "
                                       "from its model card; datasets/ppe4 if it doesn't say)")
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "datasets" / "ppe5"), help="build: the new dataset's folder")
    parser.add_argument("--repeat", type=int, default=5, help="build: how many times each feedback image is used per epoch")
    parser.add_argument("--min", type=int, default=10, help="build: refuse with fewer ready images than this")
    args = parser.parse_args()
    out = Path(args.out)
    if args.action == "export":
        return export(args)
    if args.action == "status":
        show_status(out)
        return 0
    ready = [i for i in load_index(out) if i.status == "ready" and i.event_id not in skipped(out)]
    if len(ready) < args.min:
        print(f"Only {len(ready)} false alarm(s) ready to train on (at least {args.min} needed; --min to change it). "
              "Too few to change a model trained on thousands of photos: keep marking false alarms on the dashboard.")
        return 3
    base = Path(args.base) if args.base else base_dataset()
    if not base.is_file():
        print(f"No base dataset {base}. On the Mac it is made by: bash scripts/mac_retrain.sh data")
        return 1
    c = build_dataset(base, Path(args.dataset), out, args.repeat)
    print(f"{args.dataset}: {c['base_train']} training images from {base.parent.name}, plus {c['feedback']} false alarm(s) "
          f"x {c['repeat']} = {c['train']} per epoch; validation and test unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
