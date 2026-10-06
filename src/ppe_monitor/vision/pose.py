"""Body keypoints: where a person's head and torso really are, whatever their posture.

The box-based PPE rule (rules/ppe.py) assumes an upright person, with the head at the top of the
box. That breaks for a worker who bends over to pick something up, crouches, or lies on the ground:
the head can be halfway down the box or at one side, so a worn helmet falls outside the "head
region" and one frame reads as "no helmet". Bending for a few seconds would then be enough to
raise a false alert.

A pose model (YOLO26n-pose, COCO-pretrained; no training needed) finds 17 keypoints per person:
nose, eyes, ears, shoulders, elbows, wrists, hips, knees and ankles. From those:
  head    the average of the visible nose / eye / ear points
  torso   the box around the visible shoulders and hips
  scale   the neck length (head to shoulder midpoint), or failing that the shoulder width
  posture upright / bending / crouching / lying, from the torso angle and the legs

All geometry here uses "frame heights" as the unit (x is multiplied by width / height), so
distances mean the same on a 16:9 camera as on a square photo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER = 0, 1, 2, 3, 4, 5, 6
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16
HEAD = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)
SHOULDERS, HIPS = (L_SHOULDER, R_SHOULDER), (L_HIP, R_HIP)
UPRIGHT, BENDING, CROUCHING, LYING = "upright", "bending", "crouching", "lying"


@dataclass
class Pose:
    box: tuple[float, float, float, float]  # 0-1 fractions of the frame
    xy: np.ndarray                          # (17, 2) keypoints, 0-1 fractions of the frame
    conf: np.ndarray                        # (17,) keypoint confidence
    aspect: float = 1.0                     # frame width / height
    min_conf: float = 0.5

    def _pts(self, idx) -> np.ndarray:
        """Visible keypoints among idx, in frame-height units."""
        sel = [i for i in idx if self.conf[i] >= self.min_conf]
        if not sel:
            return np.zeros((0, 2))
        p = self.xy[sel].astype(float).copy()
        p[:, 0] *= self.aspect
        return p

    def _mean(self, idx):
        p = self._pts(idx)
        return p.mean(axis=0) if len(p) else None

    @property
    def head(self):
        return self._mean(HEAD)

    @property
    def shoulders(self):
        return self._mean(SHOULDERS)

    @property
    def hips(self):
        return self._mean(HIPS)

    @property
    def ankles(self):
        """Where the person stands: the average of the visible ankles (frame-height units)."""
        return self._mean((L_ANKLE, R_ANKLE))

    @property
    def scale(self) -> float | None:
        """A body length unit: the neck (head to shoulders), or half the shoulder width."""
        head, sh = self.head, self.shoulders
        cands = []
        if head is not None and sh is not None:
            cands.append(float(np.linalg.norm(head - sh)))
        s = self._pts(SHOULDERS)
        if len(s) == 2:
            cands.append(float(np.linalg.norm(s[0] - s[1])) * 0.5)
        return max(cands) if cands and max(cands) > 0 else None

    @property
    def torso_box(self):
        """Box around the visible shoulders and hips (frame-height units), or None if too few."""
        p = np.vstack([self._pts(SHOULDERS), self._pts(HIPS)])
        if len(self._pts(SHOULDERS)) == 0 or len(p) < 2:
            return None
        return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())

    def posture(self) -> str | None:
        """upright / bending / crouching / lying, or None when the keypoints can't tell."""
        sh, hp = self.shoulders, self.hips
        if sh is None or hp is None:
            return None
        torso = sh - hp                       # from hips up to shoulders
        length = float(np.linalg.norm(torso))
        if length <= 0:
            return None
        # 0 = upright, 90 = horizontal, over 90 = shoulders below the hips (bent right over)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -torso[1] / length))))
        knees, ankles = self._pts((L_KNEE, R_KNEE)), self._pts((L_ANKLE, R_ANKLE))
        legs = ankles if len(ankles) else knees
        legs_flat = None
        if len(legs):
            leg = legs.mean(axis=0) - hp
            legs_flat = abs(leg[1]) < 0.5 * abs(leg[0])
        if tilt > 60 and (legs_flat or (legs_flat is None and tilt > 75)):
            return LYING
        if len(ankles):
            crouched = float(ankles[:, 1].mean() - hp[1]) < 0.6 * length   # hips close to the feet
        elif len(knees):
            crouched = float(knees[:, 1].mean() - hp[1]) < 0.25 * length
        else:
            crouched = False
        if crouched:
            return CROUCHING
        if tilt > 35:
            return BENDING
        return UPRIGHT


def poses_from_result(result, aspect: float, min_conf: float = 0.5) -> list[Pose]:
    """Ultralytics pose Results -> Pose list."""
    if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxyn.cpu().numpy()
    xy = result.keypoints.xyn.cpu().numpy()
    conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else np.ones(xy.shape[:2])
    return [Pose(tuple(float(v) for v in b), k, c, aspect, min_conf) for b, k, c in zip(boxes, xy, conf)]


def carried(pose: Pose, old_box, new_box) -> Pose:
    """The same keypoints moved from a person's box in an earlier frame to their box now.

    Used when the keypoint model skips a frame (pose.every in configs/ppe.yaml): posture hardly
    changes in a tenth of a second, but the person may have walked a little, so each keypoint keeps
    its place *within* the person's box, and the box is today's."""
    ox1, oy1, ox2, oy2 = old_box
    nx1, ny1, nx2, ny2 = new_box
    sx = (nx2 - nx1) / (ox2 - ox1) if ox2 > ox1 else 1.0
    sy = (ny2 - ny1) / (oy2 - oy1) if oy2 > oy1 else 1.0
    xy = pose.xy.astype(float).copy()
    xy[:, 0] = nx1 + (xy[:, 0] - ox1) * sx
    xy[:, 1] = ny1 + (xy[:, 1] - oy1) * sy
    b = pose.box
    box = (nx1 + (b[0] - ox1) * sx, ny1 + (b[1] - oy1) * sy, nx1 + (b[2] - ox1) * sx, ny1 + (b[3] - oy1) * sy)
    return Pose(box, xy, pose.conf, pose.aspect, pose.min_conf)


def match_poses(people: list, poses: list[Pose], min_iou: float = 0.5) -> list[Pose | None]:
    """The pose belonging to each person box (greedy by IoU, one-to-one), or None."""
    pairs = []
    for i, p in enumerate(people):
        for j, q in enumerate(poses):
            b = q.box
            ix = max(0.0, min(p[2], b[2]) - max(p[0], b[0]))
            iy = max(0.0, min(p[3], b[3]) - max(p[1], b[1]))
            inter = ix * iy
            union = (p[2] - p[0]) * (p[3] - p[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            iou = inter / union if union > 0 else 0.0
            if iou >= min_iou:
                pairs.append((iou, i, j))
    out: list[Pose | None] = [None] * len(people)
    used = set()
    for _, i, j in sorted(pairs, reverse=True):
        if out[i] is None and j not in used:
            out[i] = poses[j]
            used.add(j)
    return out
