"""Did training start memorising its images? Read from Ultralytics' per-epoch results.csv.

Three signs:
  - where the best epoch (highest val mAP@50-95: the checkpoint that is kept) falls in the run;
  - validation loss after the best epoch: rising while training loss keeps falling is memorising;
  - the gap between mAP@50 on (a sample of) the training images and on the validation images.
Only the best epoch's weights are kept, so memorising in later epochs doesn't reach the model;
the report says so, and says whether the kept model itself shows a large train/val gap.
"""

from __future__ import annotations

import csv
from pathlib import Path

VAL_LOSS_RISE = 0.05     # validation loss 5 % above its value at the best epoch ...
TRAIN_LOSS_FALL = 0.02   # ... while training loss fell by 2 % or more: memorising
GAP = 0.08               # mAP@50 on training images this much above val: memorising


def read_results(path: Path) -> list[dict]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    return [{k.strip(): float(v) for k, v in r.items() if k and v not in (None, "")} for r in rows]


def overfitting_check(results_csv: Path, train_map50: float, train_images: int, val_map50: float) -> tuple[str, dict]:
    rows = read_results(Path(results_csv))
    key = next(k for k in rows[0] if "mAP50-95" in k)
    key50 = next(k for k in rows[0] if "mAP50" in k and "95" not in k)
    tl = [sum(v for k, v in r.items() if k.startswith("train/") and "loss" in k) for r in rows]
    vl = [sum(v for k, v in r.items() if k.startswith("val/") and "loss" in k) for r in rows]
    best = max(range(len(rows)), key=lambda i: rows[i][key])
    last = len(rows) - 1
    val_rise = (vl[last] - vl[best]) / vl[best] if vl[best] else 0.0
    train_fall = (tl[best] - tl[last]) / tl[best] if tl[best] else 0.0
    gap = train_map50 - val_map50
    later, kept = [], []
    if last - best >= 3 and val_rise > VAL_LOSS_RISE and train_fall > TRAIN_LOSS_FALL:
        later.append(f"after epoch {best + 1}, validation loss rose {val_rise:.0%} while training loss fell "
                     f"{train_fall:.0%}: those later epochs were memorising, and are not in the kept model")
    if gap > GAP:
        kept.append(f"the kept model scores {gap:.3f} higher (mAP@50) on training images than on validation ones")
    if best == last:
        note = "The best epoch was the last one: it was still improving when it stopped (under-, not over-trained)."
    else:
        note = f"Training went on {last - best} epoch(s) past the best one without beating it, then stopped."
    verdict = "Signs of overfitting in the kept model." if kept else "No sign of overfitting in the kept model."
    lines = [f"- **{verdict}**"] + [f"  - {s}" for s in kept] + [
        f"- Epochs run: {len(rows)}; kept: epoch {best + 1}, the highest val mAP@50-95 ({rows[best][key]:.3f}). {note}",
        f"- Validation loss {vl[best]:.3f} at the kept epoch, {vl[last]:.3f} at the last; training loss "
        f"{tl[best]:.3f} and {tl[last]:.3f}."] + [f"  - {s}" for s in later] + [
        f"- mAP@50 on {train_images} training images {train_map50:.3f}, on validation {val_map50:.3f} "
        f"(gap {gap:+.3f}; above {GAP} would mean memorising)."]
    table = ["", "| Epoch | Train loss | Val loss | Val mAP@50 | Val mAP@50-95 |", "|---|---|---|---|---|"]
    for i, r in enumerate(rows):
        table.append(f"| {i + 1}{' (kept)' if i == best else ''} | {tl[i]:.3f} | {vl[i]:.3f} | {r[key50]:.3f} | "
                     f"{r[key]:.3f} |")
    info = {"epochs": len(rows), "best_epoch": best + 1, "val_loss_best": vl[best], "val_loss_last": vl[last],
            "train_loss_best": tl[best], "train_loss_last": tl[last], "train_map50_sample": train_map50,
            "val_map50": val_map50, "gap": gap, "kept_model_overfit": bool(kept), "later_epochs_memorising": bool(later)}
    return "\n".join(lines + table), info
