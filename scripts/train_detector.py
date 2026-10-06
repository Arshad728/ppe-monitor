#!/usr/bin/env python3
"""Fine-tune a pretrained YOLO model to find people, helmets and vests.

    python scripts/train_detector.py --quick     # ~5 min: prove GPU training works, estimate full run time
    python scripts/train_detector.py             # full training, then evaluation and a model card

    # fine-tune the Phase 1 model on datasets/ppe4 (Phase 1 data + extra datasets); the result goes
    # to models/candidates/ until scripts/compare_models.py shows it is better
    python scripts/train_detector.py --model models/ppe3_yolo26n_baseline.pt --data datasets/ppe4/data.yaml \
        --epochs 30 --patience 8 --name finetune --out-name ppe4_yolo26n_finetune --save-dir models/candidates

Runs on the best available device (the Apple GPU on a Mac). The trained model is saved as
models/<name>.pt, with a model card (models/<name>.md) recording the data, settings, time and
per-class results on the validation and test splits. Ultralytics' own training logs, curves and
sample predictions are in runs/detect/<name>/.
"""

import argparse
import csv
import json
import math
import platform
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401  (also enables the Apple GPU fallback before torch loads)

from ppe_monitor.config import PROJECT_ROOT
from ppe_monitor.device import best_device
from ppe_monitor.vision.detector import MODELS_DIR, load_model
from ppe_monitor.vision.evaluate import evaluate
from ppe_monitor.vision.overfit import overfitting_check

DATA = PROJECT_ROOT / "datasets" / "ppe3" / "data.yaml"
RUNS = PROJECT_ROOT / "runs" / "detect"


def losses_are_finite(run_dir) -> bool:
    rows = list(csv.DictReader(open(run_dir / "results.csv", encoding="utf-8")))
    loss_cols = [k for k in rows[-1] if "loss" in k]
    return bool(loss_cols) and all(math.isfinite(float(rows[-1][k])) for k in loss_cols)


def dataset_attributions(report_json: Path) -> str:
    """One line per source dataset (licence + attribution), from the build report prepare_data.py wrote."""
    try:
        sources = json.loads(report_json.read_text(encoding="utf-8"))["sources"]
    except (OSError, ValueError, KeyError):
        return f"(no build report at {report_json}; record the data sources and their licences here by hand)"
    return "\n".join(f"- **{s['title']}** ({s['licence']}): {s['attribution']}" for s in sources)


def _train_sample(data_yaml: Path, run_dir: Path, n: int = 300, seed: int = 0) -> Path:
    """A data.yaml whose 'train_sample' split lists n random training images."""
    import random

    import yaml
    from ppe_monitor.vision.detector import split_items

    items = split_items(data_yaml, "train")
    pick = random.Random(seed).sample(items, min(n, len(items)))
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(cfg.get("path") or data_yaml.parent)
    lst = run_dir / "train_sample.txt"
    # a dataset built on another (ppe4caps, ppe5) lists the base's images by their full path
    lst.write_text("\n".join(str(p.relative_to(root)) if p.is_relative_to(root) else str(p) for p, _ in pick) + "\n",
                   encoding="utf-8")
    out = run_dir / "train_sample.yaml"
    out.write_text(yaml.safe_dump({"path": str(root), "train_sample": str(lst), "names": cfg["names"]}), encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="time one epoch on a quarter of the data, then stop")
    parser.add_argument("--model", default="yolo26n.pt", help="pretrained starting point (default yolo26n.pt)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4, help="data-loading processes")
    parser.add_argument("--patience", type=int, default=15, help="stop early after this many epochs without improvement")
    parser.add_argument("--device", default="auto", help="auto (default), mps, cuda or cpu")
    parser.add_argument("--fraction", type=float, default=1.0, help="use only this share of the training images")
    parser.add_argument("--name", default="baseline", help="run name; the model is saved as models/ppe3_<model>_<name>.pt")
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--out-name", help="file name of the saved model, without .pt (default ppe3_<model>_<name>)")
    parser.add_argument("--save-dir", default=str(MODELS_DIR),
                        help="where the model goes (default models/; a fine-tuned candidate: models/candidates)")
    parser.add_argument("--lr0", type=float, help="starting learning rate (default: Ultralytics chooses)")
    parser.add_argument("--optimizer", default="auto",
                        help="auto (Ultralytics picks one and its learning rate, ignoring --lr0), AdamW or SGD")
    parser.add_argument("--lrf", type=float, default=0.01, help="final learning rate, as a fraction of the first")
    parser.add_argument("--warmup-epochs", type=float, default=3.0, help="epochs of warm-up (0: none)")
    parser.add_argument("--warmup-bias-lr", type=float, default=0.1,
                        help="the biases' learning rate at the start of warm-up (Ultralytics' default 0.1). For "
                             "fine-tuning a model already trained for this task, 0: a large start knocks its "
                             "confidence scores off")
    parser.add_argument("--close-mosaic", type=int, default=10,
                        help="train the last N epochs without mosaic (Ultralytics' default 10). For a short fine-tuning, "
                             "fewer: epochs without mosaic are easier and widen the gap between training and validation")
    args = parser.parse_args()

    data = Path(args.data)
    args.data = str(data if data.is_absolute() or data.exists() else PROJECT_ROOT / data)
    if not Path(args.data).is_file():
        print(f"No training dataset at {args.data}. Run:  python scripts/prepare_data.py")
        return 1
    device = best_device() if args.device == "auto" else args.device
    model = load_model(args.model)

    if args.quick:
        fraction = 0.25
        print(f"Quick check on {device}: 1 epoch on {fraction:.0%} of the training images ...")
        # Time the two parts of an epoch separately: training scales with the number of images,
        # the validation pass doesn't (it always covers the whole val split).
        marks = {}
        model.add_callback("on_train_epoch_start", lambda trainer: marks.setdefault("start", time.monotonic()))
        model.add_callback("on_train_epoch_end", lambda trainer: marks.setdefault("trained", time.monotonic()))
        model.add_callback("on_fit_epoch_end", lambda trainer: marks.setdefault("validated", time.monotonic()))
        model.train(data=args.data, epochs=1, imgsz=args.imgsz, batch=args.batch, device=device, workers=args.workers,
                    fraction=fraction, val=False, plots=False, project=str(RUNS), name="quick", exist_ok=True,
                    close_mosaic=0, seed=0, verbose=False)  # close_mosaic=0: time a normal (mosaic) epoch
        run_dir = model.trainer.save_dir
        ok = losses_are_finite(run_dir)
        train_s = marks["trained"] - marks["start"]
        val_s = marks["validated"] - marks["trained"]
        per_epoch = train_s / fraction + val_s
        lines = [
            f"Quick training check - {datetime.now():%Y-%m-%d %H:%M}",
            f"device             {device}",
            f"model              {args.model}, image size {args.imgsz}, batch {args.batch}",
            f"time               {train_s:.0f} s training on {fraction:.0%} of the images + {val_s:.0f} s validation",
            f"losses             {'finite - training works on this device' if ok else 'NOT FINITE - training is broken on this device'}",
            f"estimate           ~{per_epoch / 60:.1f} min per full epoch -> ~{per_epoch * args.epochs / 3600:.1f} h for "
            f"{args.epochs} epochs (usually less: it stops early once it stops improving)",
        ]
        print("\n" + "\n".join(lines))
        (PROJECT_ROOT / "logs").mkdir(exist_ok=True)
        (PROJECT_ROOT / "logs" / "train_quick.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return 0 if ok else 1

    print(f"Training {args.model} on {device} for up to {args.epochs} epochs (stops after {args.patience} "
          f"without improvement on val) ...")
    start = time.monotonic()
    extra = {"lr0": args.lr0} if args.lr0 else {}
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=device,
                workers=args.workers, patience=args.patience, fraction=args.fraction, project=str(RUNS),
                name=args.name, exist_ok=False, seed=0, plots=True, close_mosaic=args.close_mosaic,
                optimizer=args.optimizer, lrf=args.lrf, warmup_epochs=args.warmup_epochs,
                warmup_bias_lr=args.warmup_bias_lr, **extra)
    train_seconds = time.monotonic() - start
    trainer = model.trainer
    run_dir = trainer.save_dir
    epochs_done = trainer.epoch + 1

    save_dir = Path(args.save_dir)
    save_dir = save_dir if save_dir.is_absolute() else PROJECT_ROOT / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    out_name = args.out_name or f"ppe3_{args.model.rsplit('.', 1)[0]}_{args.name}"
    weights = save_dir / f"{out_name}.pt"
    shutil.copy2(trainer.best, weights)

    print("\nEvaluating the best checkpoint ...")
    best = load_model(weights)
    val = evaluate(best, args.data, "val", imgsz=args.imgsz, device=device)               # picks the threshold
    test = evaluate(best, args.data, "test", threshold=val.threshold, imgsz=args.imgsz, device=device)
    import yaml
    data_cfg = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    per_source = {k: evaluate(best, args.data, k, threshold=val.threshold, imgsz=args.imgsz, device=device,
                              with_ultralytics=False) for k in data_cfg if k.startswith("test_")}
    # a sample of the training images, to see how much better the model does on what it trained on
    sample_yaml = _train_sample(Path(args.data), run_dir)
    train_sample = evaluate(best, sample_yaml, "train_sample", threshold=val.threshold, imgsz=args.imgsz,
                            device=device, with_ultralytics=False)
    overfit_text, overfit = overfitting_check(run_dir / "results.csv", train_sample.map50, train_sample.images,
                                              val.map50)

    import torch
    import ultralytics
    attributions = dataset_attributions(Path(args.data).parent / "report.json")
    start_from = (f"`{args.model}` (COCO-pretrained)" if Path(args.model).parent == Path(".")
                  else f"`{args.model}` (fine-tuned from an already trained model)")
    data_name = Path(args.data).parent.name
    per_source_md = ""
    if per_source:
        per_source_md = "\n".join(
            ["", "### Test split by source", "",
             "| Source | Images | mAP@50 | " + " | ".join(c.name for c in test.classes) + " |",
             "|---|---|---|" + "---|" * len(test.classes)]
            + [f"| {k[5:]} | {r.images} | {r.map50:.3f} | " + " | ".join(f"{c.ap50:.3f}" for c in r.classes) + " |"
               for k, r in per_source.items()]) + "\n"
    card = f"""# Model card: {out_name}

Trained {datetime.now():%Y-%m-%d %H:%M} on {device} ({platform.machine()}, {platform.system()} {platform.mac_ver()[0] or platform.release()}).

| Setting | Value |
|---|---|
| Starting point | {start_from} |
| Data | `datasets/{data_name}` - see its report.md |
| Image size / batch | {args.imgsz} / {args.batch} |
| Epochs | {epochs_done} of {args.epochs} (early-stop patience {args.patience}; the last {args.close_mosaic} without mosaic) |
| Optimiser | {args.optimizer}{f", learning rate {args.lr0:g} falling to {args.lr0 * args.lrf:g}" if args.lr0 else ""}; warm-up {args.warmup_epochs:g} epoch(s), biases from {args.warmup_bias_lr:g} |
| Training time | {train_seconds / 60:.0f} min |
| Software | ultralytics {ultralytics.__version__}, torch {torch.__version__}, Python {platform.python_version()} |
| Working confidence threshold | **{val.threshold:.3f}** (best F1 on val; used for precision/recall below) |

## Validation split ({val.images} images)

{val.table()}
Ultralytics' own figures for the same split: mAP@50 {val.map50_ultralytics:.3f}, mAP@50-95 {val.map50_95_ultralytics:.3f}.

## Test split ({test.images} images)

{test.table()}
Ultralytics' own figures for the same split: mAP@50 {test.map50_ultralytics:.3f}, mAP@50-95 {test.map50_95_ultralytics:.3f}.
{per_source_md}
## Overfitting check

{overfit_text}

**Why the two mAP figures differ.** The tables above score exactly what the live detector outputs:
one class per box. Ultralytics' validator lets one box carry several classes at once, which
helps a class that is often the runner-up at the same spot (for example "person" under a vest).
On a well-trained model the two usually agree within a few points; a large gap means the model
often ranks a second class almost as high as the first.

**What these numbers are, and aren't.** Both splits come from the same public datasets as the
training data, with near-duplicate images removed. They measure how well the model learned *these
datasets*. The project target (mAP@50 >= 0.85) is defined on scenes the model never trained on;
that measurement needs your own labelled clips (see data/README.md). Distant, small people are
almost absent from these splits, so recall on small objects is barely tested here.

## Training data and licences

{attributions}

Trained with Ultralytics YOLO (AGPL-3.0). Fine for learning and portfolio use; a commercial product
needs an Ultralytics licence or a differently licensed detector and data.
"""
    (save_dir / f"{out_name}.md").write_text(card, encoding="utf-8")
    (save_dir / f"{out_name}.json").write_text(json.dumps({
        "val": val.to_dict(), "test": test.to_dict(), "test_by_source": {k: r.to_dict() for k, r in per_source.items()},
        "overfitting": overfit, "epochs": epochs_done, "train_seconds": train_seconds, "device": device,
        "run_dir": str(run_dir), "start_from": args.model, "data": args.data}, indent=2))
    print(card)
    shown = weights.relative_to(PROJECT_ROOT) if weights.is_relative_to(PROJECT_ROOT) else weights
    print(f"Saved {shown} and its model card {out_name}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
