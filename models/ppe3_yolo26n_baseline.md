# Model card: ppe3_yolo26n_baseline

Trained 2026-09-23 05:33 on mps (arm64, Darwin 27.0).

| Setting | Value |
|---|---|
| Starting point | `yolo26n.pt` (COCO-pretrained) |
| Data | `datasets/ppe3` - see its report.md |
| Image size / batch | 640 / 16 |
| Epochs | 54 of 60 (early-stop patience 15) |
| Training time | 242 min |
| Software | ultralytics 8.4.159, torch 2.14.0, Python 3.11.16 |
| Working confidence threshold | **0.350** (best F1 on val; used for precision/recall below) |

## Validation split (237 images)

| Class | Objects | AP@50 | Precision | Recall | Recall: small / medium / large |
|---|---|---|---|---|---|
| person | 450 | 0.928 | 0.903 | 0.907 | 0.00 (2) / 0.50 (26) / 0.94 (422) |
| helmet | 403 | 0.879 | 0.896 | 0.836 | 0.60 (45) / 0.84 (244) / 0.93 (114) |
| vest | 282 | 0.880 | 0.868 | 0.816 | 0.00 (1) / 0.69 (77) / 0.87 (204) |
| **all** | 1135 | **0.896** | | | |

Ultralytics' own figures for the same split: mAP@50 0.901, mAP@50-95 0.528.

## Test split (200 images)

| Class | Objects | AP@50 | Precision | Recall | Recall: small / medium / large |
|---|---|---|---|---|---|
| person | 409 | 0.838 | 0.826 | 0.848 | 0.00 (3) / 0.56 (18) / 0.87 (388) |
| helmet | 350 | 0.921 | 0.880 | 0.857 | 0.46 (24) / 0.86 (249) / 0.96 (77) |
| vest | 270 | 0.849 | 0.852 | 0.789 | - (0) / 0.70 (53) / 0.81 (217) |
| **all** | 1029 | **0.869** | | | |

Ultralytics' own figures for the same split: mAP@50 0.872, mAP@50-95 0.502.

**Why the two mAP figures differ.** The tables above score exactly what the live detector outputs:
one class per box. Ultralytics' validator lets one box carry several classes at once, which
helps a class that is often the runner-up at the same spot (for example "person" under a vest).
On a well-trained model the two usually agree within a few points; a large gap means the model
often ranks a second class almost as high as the first.

**What these numbers are, and aren't.** Both splits come from the same two public datasets as the
training data, with near-duplicate images removed. They measure how well the model learned *these
datasets*. The project target (mAP@50 >= 0.85) is defined on scenes the model never trained on;
that measurement needs your own labelled clips (see data/README.md). Distant, small people are
almost absent from these splits, so recall on small objects is barely tested here.

## Training data and licences

- **Construction-PPE (Ultralytics)** (AGPL-3.0): Construction-PPE dataset by Ultralytics, https://docs.ultralytics.com/datasets/detect/construction-ppe/
- **Construction Safety (Roboflow 100)** (CC BY 4.0): construction safety dataset, Roboflow Universe (Roboflow 100 benchmark), CC BY 4.0, https://universe.roboflow.com/object-detection/construction-safety-gsnvb

Trained with Ultralytics YOLO (AGPL-3.0). Fine for learning and portfolio use; a commercial product
needs an Ultralytics licence or a differently licensed detector and data.
