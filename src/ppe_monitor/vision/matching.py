"""Deciding which predictions hit a real object (book, Chapter 6).

A prediction counts as a hit (true positive) when it has the same class as a labelled box
and overlaps it with IoU >= 0.5, and that labelled box hasn't already been claimed by a more
confident prediction. Everything else is a false positive; labelled boxes nobody claimed are
misses (false negatives). All boxes here are (x1, y1, x2, y2) in 0-1 image fractions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Det:
    cls: int
    box: tuple[float, float, float, float]
    score: float = 1.0


@dataclass
class ImageMatch:
    tp: list[int] = field(default_factory=list)       # indices of predictions that hit
    fp: list[int] = field(default_factory=list)       # indices of predictions that missed
    fn: list[int] = field(default_factory=list)       # indices of labelled boxes nobody found
    pairs: dict[int, int] = field(default_factory=dict)  # prediction index -> labelled box index


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (N x 4) and every box in b (M x 4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


def match_image(preds: list[Det], gts: list[Det], iou_threshold: float = 0.5) -> ImageMatch:
    """Greedy matching, most confident prediction first, one labelled box per prediction."""
    result = ImageMatch()
    order = sorted(range(len(preds)), key=lambda i: -preds[i].score)
    claimed: set[int] = set()
    if preds and gts:
        ious = iou_matrix(np.array([p.box for p in preds]), np.array([g.box for g in gts]))
    for i in order:
        best, best_iou = None, iou_threshold
        for j, g in enumerate(gts):
            if j in claimed or g.cls != preds[i].cls:
                continue
            if ious[i, j] >= best_iou:
                best, best_iou = j, ious[i, j]
        if best is None:
            result.fp.append(i)
        else:
            claimed.add(best)
            result.tp.append(i)
            result.pairs[i] = best
    result.fn = [j for j in range(len(gts)) if j not in claimed]
    return result


def average_precision(scores: list[float], is_tp: list[bool], n_gt: int) -> float:
    """Area under the precision-recall curve (all-point interpolation, as in Chapter 6)."""
    if n_gt == 0:
        return float("nan")
    if not scores:
        return 0.0
    order = np.argsort(-np.asarray(scores), kind="stable")
    tp = np.asarray(is_tp, dtype=float)[order]
    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(1 - tp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    # make precision monotonically decreasing, then integrate over recall
    mrec = np.concatenate([[0.0], recall, [recall[-1]]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    idx = np.nonzero(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _scored_hits(images: list[tuple[list[Det], list[Det]]]) -> tuple[np.ndarray, np.ndarray, int]:
    """Scores and hit/miss flags of every prediction, most confident first, plus the object count.

    Matching runs most-confident-first, so whether a prediction is a hit never depends on the
    weaker predictions below it. One matching pass therefore gives the exact result at every
    confidence threshold: keeping predictions scoring >= t just keeps a prefix of this list.
    """
    scores, hits, n_gt = [], [], 0
    for preds, gts in images:
        m = match_image(preds, gts)
        scores += [p.score for p in preds]
        hits += [i in m.pairs for i in range(len(preds))]
        n_gt += len(gts)
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    return np.asarray(scores, dtype=float)[order], np.asarray(hits, dtype=bool)[order], n_gt


def f1_by_threshold(images: list[tuple[list[Det], list[Det]]], thresholds: np.ndarray) -> np.ndarray:
    """Overall F1 (all classes together) when only predictions scoring >= t are kept, for each t."""
    scores, hits, n_gt = _scored_hits(images)
    f1 = []
    for t in thresholds:
        kept = scores >= t
        tp = int(hits[kept].sum())
        fp = int(kept.sum()) - tp
        f1.append(2 * tp / max(2 * tp + fp + (n_gt - tp), 1))
    return np.array(f1)


def best_f1_threshold(images: list[tuple[list[Det], list[Det]]]) -> tuple[float, float]:
    """(threshold, F1): the confidence cut-off that best balances false alarms against misses,
    searched over every score the model actually produced."""
    scores, hits, n_gt = _scored_hits(images)
    if len(scores) == 0 or n_gt == 0:
        return 0.5, 0.0
    tp = np.cumsum(hits)
    fp = np.cumsum(~hits)
    f1 = 2 * tp / np.maximum(2 * tp + fp + (n_gt - tp), 1)
    # only consider cut points between different scores (ties must be kept or dropped together)
    last_of_run = np.append(scores[1:] != scores[:-1], True)
    best = int(np.argmax(np.where(last_of_run, f1, -1)))
    return float(scores[best]), float(f1[best])
