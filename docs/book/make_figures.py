#!/usr/bin/env python3
"""The book's figures: the two architecture diagrams (Part III) and the result charts of the
build notes (Part IV), in the book's palette. Writes docs/book/figures/*.png.

    python docs/book/make_figures.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path

# ---- Palette (matches the book) ----
TEAL = "#1f6f5c"
TEAL_DARK = "#123f34"
INK = "#20302d"
SUBTLE = "#5b6b68"
LIGHT_BG = "#eef4f2"
AMBER = "#d9a441"
WHITE = "#ffffff"
LINE = "#c7d6d2"

plt.rcParams["font.family"] = "DejaVu Sans"

from pathlib import Path

OUT_DIR = str(Path(__file__).resolve().parent / "figures")


def box(ax, x, y, w, h, text, fc=WHITE, ec=TEAL, tc=INK, fontsize=9.5, weight="normal",
        boxstyle="round,pad=0.02,rounding_size=0.06", zorder=3, lw=1.4):
    patch = FancyBboxPatch((x, y), w, h, boxstyle=boxstyle, linewidth=lw,
                            edgecolor=ec, facecolor=fc, zorder=zorder)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
             color=tc, weight=weight, zorder=zorder + 1, linespacing=1.3)
    return patch


def group_box(ax, x, y, w, h, label, fc=LIGHT_BG, ec=TEAL):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                            linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=1,
                            linestyle="solid")
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.16, label, ha="center", va="top", fontsize=10.5,
             color=TEAL_DARK, weight="bold", zorder=2)
    return patch


def arrow(ax, x1, y1, x2, y2, color=TEAL_DARK, lw=1.6, style="-|>", dashed=False,
          connectionstyle="arc3,rad=0.0"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                         linewidth=lw, color=color, zorder=2,
                         linestyle=(0, (4, 2)) if dashed else "solid",
                         connectionstyle=connectionstyle)
    ax.add_patch(a)


# =====================================================================
# DIAGRAM A -- The Big Picture (end-to-end architecture)
# =====================================================================
fig, ax = plt.subplots(figsize=(12.4, 7.6), dpi=200)
ax.set_xlim(0, 12.4)
ax.set_ylim(0, 7.6)
ax.axis("off")

def elbow(ax, x1, y1, x2, y2, color=TEAL_DARK, lw=1.6, dashed=False):
    """Simple right-angle connector: horizontal then vertical (or straight if aligned)."""
    if abs(y1 - y2) < 0.02 or abs(x1 - x2) < 0.02:
        arrow(ax, x1, y1, x2, y2, color=color, lw=lw, dashed=dashed)
        return
    xm = (x1 + x2) / 2
    ls = (0, (4, 2)) if dashed else "solid"
    ax.plot([x1, xm], [y1, y1], color=color, lw=lw, linestyle=ls, zorder=2)
    ax.plot([xm, xm], [y1, y2], color=color, lw=lw, linestyle=ls, zorder=2)
    arrow(ax, xm, y2, x2, y2, color=color, lw=lw, dashed=dashed)

# --- Column bands (bottom-y, height) ---
TOP_Y, TOP_H = 6.05, 0.95      # top band  (Ingestion / YOLO / FastAPI)
MID_Y, MID_H = 4.55, 0.95      # mid band  (ByteTrack / PostgreSQL)
LOW_Y, LOW_H = 3.05, 0.95      # low band  (PPE+Zone rules)
BOT_Y, BOT_H = 1.55, 0.95      # bottom band (Event confirm / Alert queue+workers)

# --- Column 1: Cameras (three, feeding one ingestion point) ---
box(ax, 0.3, 6.35, 1.55, 0.65, "Camera 1\n(RTSP)", fontsize=8.6)
box(ax, 0.3, 5.35, 1.55, 0.65, "Camera 2\n(RTSP)", fontsize=8.6)
box(ax, 0.3, 4.35, 1.55, 0.65, "Camera N\n(RTSP)", fontsize=8.6)
ax.text(1.07, 7.25, "CAMERAS", ha="center", fontsize=10.5, color=TEAL_DARK, weight="bold")

# --- Column 2: Ingestion (top band) ---
group_box(ax, 2.35, 0.9, 2.05, 6.7, "")
ax.text(3.37, 7.25, "INGESTION", ha="center", fontsize=10.5, color=TEAL_DARK, weight="bold")
box(ax, 2.5, TOP_Y, 1.75, TOP_H, "Reader thread +\nlatest-frame queue\n(one per camera)", fontsize=8.2)
ax.text(3.37, 5.55, "Each camera runs its\nown copy of this unit,\nkeeping only the\nnewest frame.",
        ha="center", va="top", fontsize=7.6, color=SUBTLE, style="italic")

# converge camera arrows into one point on the ingestion box
conv_x, conv_y = 2.5, TOP_Y + TOP_H / 2
elbow(ax, 1.85, 6.67, conv_x, conv_y, lw=1.4)
elbow(ax, 1.85, 5.67, conv_x, conv_y, lw=1.4)
elbow(ax, 1.85, 4.67, conv_x, conv_y, lw=1.4)

# --- Column 3: Vision Engine (grouped, 4 bands) ---
group_box(ax, 4.75, 0.9, 2.55, 6.7, "")
ax.text(6.02, 7.25, "VISION ENGINE", ha="center", fontsize=10.5, color=TEAL_DARK, weight="bold")
box(ax, 4.9, TOP_Y, 2.25, TOP_H, "Batched YOLO\ndetection\n(ONNX / TensorRT)", fontsize=8.4)
box(ax, 4.9, MID_Y, 2.25, MID_H, "ByteTrack\n(assigns track IDs)", fontsize=8.6)
box(ax, 4.9, LOW_Y, 2.25, LOW_H, "PPE matching +\nzone rules", fontsize=8.6)
box(ax, 4.9, BOT_Y, 2.25, BOT_H, "Event confirmation\n(dwell + cooldown)", fontsize=8.4)
elbow(ax, 6.02, TOP_Y, 6.02, MID_Y + MID_H, lw=1.4)
elbow(ax, 6.02, MID_Y, 6.02, LOW_Y + LOW_H, lw=1.4)
elbow(ax, 6.02, LOW_Y, 6.02, BOT_Y + BOT_H, lw=1.4)

# ingestion -> vision engine (top band, same height)
elbow(ax, 4.25, TOP_Y + TOP_H / 2, 4.9, TOP_Y + TOP_H / 2, lw=1.7)

# --- Column 4: Backend (grouped, 3 bands: top / mid / bottom) ---
group_box(ax, 7.65, 0.9, 2.85, 6.7, "")
ax.text(9.07, 7.25, "BACKEND", ha="center", fontsize=10.5, color=TEAL_DARK, weight="bold")
box(ax, 7.8, TOP_Y, 2.55, TOP_H, "FastAPI\n(dashboard + API)", fontsize=8.6)
box(ax, 7.8, MID_Y, 2.55, MID_H, "PostgreSQL\n(cameras, zones, rules,\nviolations)", fontsize=8.2)
box(ax, 7.8, BOT_Y, 2.55, BOT_H, "Alert queue -> workers\n(Telegram / Email /\nWhatsApp)", fontsize=8.0)

# FastAPI <-> PostgreSQL (adjacent bands, reads & writes)
elbow(ax, 9.07, TOP_Y, 9.07, MID_Y + MID_H, lw=1.5)
ax.text(9.75, (TOP_Y + MID_Y + MID_H) / 2, "reads /\nwrites", fontsize=6.8, color=SUBTLE, ha="left")

# vision engine -> backend (bottom band: event -> alert queue; and event -> postgres)
elbow(ax, 7.15, BOT_Y + BOT_H / 2, 7.8, BOT_Y + BOT_H / 2, lw=1.7)
elbow(ax, 7.4, BOT_Y + BOT_H / 2, 7.4, MID_Y + MID_H / 2, lw=1.4)
elbow(ax, 7.4, MID_Y + MID_H / 2, 7.8, MID_Y + MID_H / 2, lw=1.4)
ax.text(6.75, 2.35, "writes event", fontsize=6.8, color=SUBTLE, ha="center")

# --- Column 5: People ---
box(ax, 10.85, 1.3, 1.35, 5.75, "Safety\nOfficer\n\n(phone +\ndashboard\nbrowser)", fontsize=8.4,
    fc=LIGHT_BG, ec=TEAL_DARK)

elbow(ax, 10.35, TOP_Y + TOP_H / 2, 10.85, TOP_Y + TOP_H / 2, lw=1.6)   # FastAPI -> officer
elbow(ax, 10.35, BOT_Y + BOT_H / 2, 10.85, BOT_Y + BOT_H / 2, lw=1.6)   # Alert workers -> officer

# Feedback: officer confirms/rejects -> PostgreSQL (dashed amber, same mid band)
elbow(ax, 10.85, MID_Y + MID_H / 2 - 0.18, 10.35, MID_Y + MID_H / 2 - 0.18,
      color=AMBER, lw=1.5, dashed=True)
ax.text(9.07, MID_Y - 0.3, "confirms / rejects -> later used to retrain the model",
        fontsize=7.6, color="#8a6415", ha="center", style="italic")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/diagram_architecture.png", bbox_inches="tight", facecolor="white")
plt.close()
print("Saved diagram_architecture.png")


# =====================================================================
# DIAGRAM B — Life of a Violation (vertical timeline)
# =====================================================================
fig, ax = plt.subplots(figsize=(8.6, 10.5), dpi=200)
ax.set_xlim(0, 8.6)
ax.set_ylim(0, 10.5)
ax.axis("off")

steps = [
    ("t = 0.0s", "A worker removes their helmet near Zone A and sets it on a railing.",
     False),
    ("t = 0.07s", "The next camera frame arrives and overwrites the reader's\nlatest-frame queue.", False),
    ("t = 0.09s", "Batched YOLO detects: person (0.98), no matching helmet box\nin the head region.", False),
    ("t = 0.09s", "ByteTrack matches this person to existing Track #482\n(seen continuously for the last 40 seconds).", False),
    ("t = 0.1 – 2.0s", "The PPE-missing state persists across the majority-vote\nwindow — not a one-frame flicker.", False),
    ("t = 2.0s", "Dwell time satisfied → violation CONFIRMED as one event\nfor Track #482.", True),
    ("t = 2.1s", "Snapshot saved; event written to PostgreSQL; message\ndropped onto the alert queue (vision loop keeps running).", False),
    ("t = 3.6s", "Alert worker sends the Telegram message: photo, camera,\nzone, timestamp.", False),
    ("~ 1 – 3 min", "Safety officer walks over; worker puts the helmet back on.\nOfficer confirms the alert on the dashboard.", False),
    ("later", "Confirmed (or rejected) alerts accumulate as labeled\nexamples for the next model retraining pass.", False),
]

n = len(steps)
top = 9.9
bottom = 0.5
step_h = (top - bottom) / n
box_h = step_h * 0.72

for i, (t, text, highlight) in enumerate(steps):
    y_top = top - i * step_h
    y = y_top - box_h
    fc = "#fdf1e0" if highlight else WHITE
    ec = AMBER if highlight else TEAL
    # timestamp label
    ax.text(0.05, y + box_h / 2, t, ha="left", va="center", fontsize=9, color=TEAL_DARK,
            weight="bold")
    # node dot on the spine
    ax.plot([1.85], [y + box_h / 2], marker="o", markersize=7,
            color=AMBER if highlight else TEAL, zorder=4)
    # box
    box(ax, 2.15, y, 6.15, box_h, text, fc=fc, ec=ec, fontsize=8.8, tc=INK)
    # connecting spine line to next
    if i < n - 1:
        ax.plot([1.85, 1.85], [y, y - (step_h - box_h)], color=LINE, lw=1.8, zorder=1)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/diagram_violation_timeline.png", bbox_inches="tight", facecolor="white")
plt.close()
print("Saved diagram_violation_timeline.png")


# =====================================================================
# CHART C — Phase 4: frames analysed per camera, by model format (Mac M4)
# =====================================================================
cams = [1, 2, 4, 8]
series = [                       # docs/phase4_results.md, Mac benchmarks 25 Sep 2026
    ("Core ML FP16 (Neural Engine)", [10, 10, 10, 10.0], TEAL_DARK, "o", "-"),
    ("PyTorch FP16 (GPU)", [10, 10, 10, 9.2], TEAL, "s", "-"),
    ("ONNX FP32 (CPU)", [10, 6.1, 4.4, 2.5], AMBER, "D", "--"),
    ("Core ML, keypoints on every frame (first run)", [10, 10, 10, 7.7], "#9aa5a1", "o", ":"),
]
fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=200)
for name, ys, colour, marker, ls in series:
    ax.plot(cams, ys, marker=marker, color=colour, lw=2, ls=ls, label=name, markersize=6)
ax.axhline(10, color=LINE, lw=1.2, zorder=0)
ax.text(8.1, 10.15, "target: 10 frames/s", fontsize=8, color=SUBTLE, ha="right", va="bottom")
ax.set_xscale("log", base=2)
ax.set_xticks(cams)
ax.set_xticklabels([str(c) for c in cams])
ax.set_xlabel("cameras watched at once", fontsize=9, color=INK)
ax.set_ylabel("frames analysed per second,\nper camera", fontsize=9, color=INK)
ax.set_ylim(0, 11.5)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(LINE)
ax.tick_params(colors=INK, labelsize=8.5)
ax.legend(frameon=False, fontsize=8, loc="lower left")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/chart_phase4_throughput.png", bbox_inches="tight", facecolor="white")
plt.close()
print("Saved chart_phase4_throughput.png")


# =====================================================================
# CHART D — the first real clips: helmets found on the head, by colour, before and after fine-tuning
# =====================================================================
import numpy as np

colours = ["yellow", "navy", "brown", "white\n(in haze)", "no helmet\n(bare head)"]
frames = [200, 251, 23, 11, 32]
before = [98, 41, 0, 0, 3]      # docs/real_clips.md: a helmet found on the head (score >= 0.25), Phase 1 model
after = [100, 96, 91, 91, 3]    # the same frames, fine-tuned model (docs/finetune_results.md)
x = np.arange(len(colours))
fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=200)
b1 = ax.bar(x - 0.19, before, 0.36, color=LINE, edgecolor=SUBTLE, lw=0.6, label="Phase 1 model")
b2 = ax.bar(x + 0.19, after, 0.36, color=TEAL_DARK, label="fine-tuned model")
for bars in (b1, b2):
    for b in bars:
        v = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%", ha="center", va="bottom", fontsize=7.5,
                color=INK)
ax.set_xticks(x)
ax.set_xticklabels([f"{c}\n{n} frames" for c, n in zip(colours, frames)])
ax.set_ylim(0, 115)
ax.set_ylabel("frames with a helmet found\non the person's head (%)", fontsize=9, color=INK)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(LINE)
ax.tick_params(colors=INK, labelsize=8)
ax.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 1.08), ncol=2)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/chart_helmet_colours.png", bbox_inches="tight", facecolor="white")
plt.close()
print("Saved chart_helmet_colours.png")


# =====================================================================
# DIAGRAM E — the system as built (Phase 6): README and Chapter 23
# =====================================================================
fig, ax = plt.subplots(figsize=(12.4, 7.2), dpi=200)
ax.set_xlim(0, 12.4)
ax.set_ylim(0, 7.2)
ax.axis("off")

# column headers
for x, label in ((0.95, "CAMERAS"), (3.55, "CAMERA SERVICE"), (6.55, "STORAGE"), (8.95, "SERVICES"), (11.35, "PEOPLE")):
    ax.text(x, 6.95, label, ha="center", fontsize=10.2, color=TEAL_DARK, weight="bold")

# cameras
box(ax, 0.15, 5.55, 1.6, 0.8, "IP cameras\n(RTSP)", fontsize=8.4)
box(ax, 0.15, 4.45, 1.6, 0.8, "Test clips\n(played once,\nas cameras)", fontsize=8.0)

# camera service (one process; container "cameras")
group_box(ax, 2.1, 0.55, 2.9, 6.2, "")
ax.text(3.55, 6.55, "run_cameras.py  ·  one process, all cameras", ha="center", fontsize=7.2, color=SUBTLE, style="italic")
steps = [
    "Reader thread per camera\nnewest frame only; reconnects",
    "Detector, one batch per tick\nYOLO26n fine-tuned · Core ML / ONNX\n/ PyTorch · 10 frames/s per camera",
    "Keypoints (every 2nd frame)\nwhere the head is; posture",
    "Tracker\nByteTrack + re-linking lost people",
    "PPE rule + zone rules\nconfigs/rules.yaml, zones.yaml",
    "Event engine\nvote · dwell 3 s · cooldown 60 s",
]
ys = [5.62, 4.6, 3.72, 2.9, 2.08, 1.26]
hs = [0.72, 0.86, 0.66, 0.66, 0.66, 0.66]
for (y, h, t) in zip(ys, hs, steps):
    box(ax, 2.25, y, 2.6, h, t, fontsize=7.3)
for i in range(len(ys) - 1):
    arrow(ax, 3.55, ys[i], 3.55, ys[i + 1] + hs[i + 1], lw=1.3)
ax.text(3.55, 0.78, "hands events to a writer thread:\nthe video loop never waits", ha="center", fontsize=6.9,
        color=SUBTLE, style="italic")
elbow(ax, 1.75, 5.95, 2.25, 5.98, lw=1.4)
elbow(ax, 1.75, 4.85, 2.25, 5.98, lw=1.4)

# storage
box(ax, 5.35, 3.55, 2.4, 1.55, "PostgreSQL 16\nevents · alerts (the queue)\ncameras · rules · zones\nheartbeats", fontsize=7.8)
box(ax, 5.35, 1.9, 2.4, 1.1, "Evidence images\nannotated + clean frame\n(false alarms kept)", fontsize=7.6)
box(ax, 5.35, 5.55, 2.4, 0.8, "Live camera pictures\n(one JPEG per camera, 1/s)", fontsize=7.4)
elbow(ax, 4.85, 1.59, 5.35, 4.32, lw=1.5)      # event engine -> database
elbow(ax, 4.85, 1.59, 5.35, 2.45, lw=1.3)      # -> images
elbow(ax, 4.85, 5.98, 5.35, 5.95, lw=1.2)      # reader/frames -> live tiles

# services
box(ax, 8.1, 3.55, 1.7, 1.55, "Alert worker\nrate limit · retries\nsummaries\ncamera-down\nalerts", fontsize=7.4)
box(ax, 8.1, 5.2, 1.7, 1.15, "Dashboard\nFastAPI + one page\npassword", fontsize=7.6)
box(ax, 8.1, 1.9, 1.7, 1.1, "Telegram\n(email optional)", fontsize=7.8)
arrow(ax, 7.75, 4.32, 8.1, 4.32, lw=1.5)       # queue -> worker
arrow(ax, 8.95, 3.55, 8.95, 3.0, lw=1.5)       # worker -> telegram
elbow(ax, 7.75, 5.95, 8.1, 5.78, lw=1.2)       # tiles -> dashboard
elbow(ax, 7.75, 4.9, 8.1, 5.45, lw=1.2)        # db -> dashboard

# people
box(ax, 10.25, 1.9, 2.0, 4.45, "Safety officer\n\nalert with the picture\non the phone\n(3-4 s after the event)\n\n"
    "reviews on the\ndashboard:\nconfirm / false alarm", fontsize=7.7, fc=LIGHT_BG, ec=TEAL_DARK)
arrow(ax, 9.8, 2.45, 10.25, 2.45, lw=1.5)
arrow(ax, 9.8, 5.78, 10.25, 5.78, lw=1.5)

# feedback loop (amber, dashed): verdicts -> training data -> retrain -> the detector
ax.plot([11.25, 11.25], [1.9, 0.3], color=AMBER, lw=1.5, linestyle=(0, (4, 2)), zorder=2)
ax.plot([11.25, 3.55], [0.3, 0.3], color=AMBER, lw=1.5, linestyle=(0, (4, 2)), zorder=2)
ax.text(7.4, 0.36, "false alarms -> feedback.py -> retrain -> compare -> accept: the next detector (Chapter 14's loop)",
        ha="center", va="bottom", fontsize=7.3, color="#8a6415", style="italic")
arrow(ax, 3.0, 0.3, 3.0, 0.55, color=AMBER, lw=1.5, dashed=True)
ax.plot([3.55, 3.0], [0.3, 0.3], color=AMBER, lw=1.5, linestyle=(0, (4, 2)), zorder=2)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/diagram_as_built.png", bbox_inches="tight", facecolor="white")
plt.close()
print("Saved diagram_as_built.png")
