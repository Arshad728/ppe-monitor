# Model card: ppe4_yolo26n_finetune

Trained 2026-09-25 18:37 on mps (arm64, Darwin 27.0).

| Setting | Value |
|---|---|
| Starting point | `models/ppe3_yolo26n_baseline.pt` (fine-tuned from an already trained model) |
| Data | `datasets/ppe4` - see its report.md |
| Image size / batch | 640 / 16 |
| Epochs | 30 of 30 (early-stop patience 8) |
| Training time | 193 min |
| Software | ultralytics 8.4.159, torch 2.14.0, Python 3.11.16 |
| Working confidence threshold | **0.333** (best F1 on val; used for precision/recall below) |

## Validation split (454 images)

| Class | Objects | AP@50 | Precision | Recall | Recall: small / medium / large |
|---|---|---|---|---|---|
| person | 1312 | 0.891 | 0.879 | 0.844 | 0.28 (57) / 0.67 (282) / 0.93 (973) |
| helmet | 885 | 0.878 | 0.886 | 0.838 | 0.64 (221) / 0.89 (486) / 0.96 (178) |
| vest | 359 | 0.804 | 0.917 | 0.710 | 0.00 (27) / 0.55 (106) / 0.87 (226) |
| **all** | 2556 | **0.857** | | | |

Ultralytics' own figures for the same split: mAP@50 0.866, mAP@50-95 0.539.

## Test split (628 images)

| Class | Objects | AP@50 | Precision | Recall | Recall: small / medium / large |
|---|---|---|---|---|---|
| person | 2587 | 0.898 | 0.870 | 0.842 | 0.32 (123) / 0.74 (645) / 0.91 (1819) |
| helmet | 1714 | 0.921 | 0.885 | 0.869 | 0.73 (448) / 0.92 (1028) / 0.90 (238) |
| vest | 311 | 0.836 | 0.851 | 0.772 | 0.07 (14) / 0.69 (67) / 0.84 (230) |
| **all** | 4612 | **0.885** | | | |

Ultralytics' own figures for the same split: mAP@50 0.884, mAP@50-95 0.571.

### Test split by source

| Source | Images | mAP@50 | person | helmet | vest |
|---|---|---|---|---|---|
| chv | 7 | 0.862 | 0.948 | 0.904 | 0.733 |
| extra | 428 | 0.799 | 0.907 | 0.918 | 0.573 |
| gdut-hwd | 320 | 0.915 | 0.909 | 0.922 | nan |
| ppe3 | 200 | 0.883 | 0.839 | 0.934 | 0.878 |
| sh17 | 101 | 0.780 | 0.874 | 0.843 | 0.622 |

## Overfitting check

- **Signs of overfitting in the kept model.**
  - the kept model scores 0.085 higher (mAP@50) on training images than on validation ones
- Epochs run: 30; kept: epoch 24, the highest val mAP@50-95 (0.539). Training went on 6 epoch(s) past the best one without beating it, then stopped.
- Validation loss 2.235 at the kept epoch, 2.187 at the last; training loss 1.848 and 1.749.
- mAP@50 on 300 training images 0.942, on validation 0.857 (gap +0.085; above 0.08 would mean memorising).

| Epoch | Train loss | Val loss | Val mAP@50 | Val mAP@50-95 |
|---|---|---|---|---|
| 1 | 2.600 | 2.573 | 0.819 | 0.466 |
| 2 | 2.531 | 2.595 | 0.821 | 0.456 |
| 3 | 2.574 | 2.578 | 0.818 | 0.448 |
| 4 | 2.590 | 2.560 | 0.815 | 0.449 |
| 5 | 2.535 | 2.581 | 0.820 | 0.457 |
| 6 | 2.531 | 2.492 | 0.842 | 0.480 |
| 7 | 2.512 | 2.527 | 0.829 | 0.473 |
| 8 | 2.464 | 2.472 | 0.833 | 0.487 |
| 9 | 2.460 | 2.388 | 0.837 | 0.494 |
| 10 | 2.403 | 2.429 | 0.844 | 0.494 |
| 11 | 2.404 | 2.402 | 0.849 | 0.498 |
| 12 | 2.358 | 2.372 | 0.844 | 0.510 |
| 13 | 2.326 | 2.337 | 0.853 | 0.510 |
| 14 | 2.301 | 2.359 | 0.843 | 0.505 |
| 15 | 2.270 | 2.311 | 0.860 | 0.507 |
| 16 | 2.258 | 2.328 | 0.862 | 0.515 |
| 17 | 2.217 | 2.323 | 0.857 | 0.516 |
| 18 | 2.197 | 2.274 | 0.859 | 0.518 |
| 19 | 2.189 | 2.248 | 0.870 | 0.527 |
| 20 | 2.153 | 2.238 | 0.862 | 0.532 |
| 21 | 1.955 | 2.239 | 0.862 | 0.529 |
| 22 | 1.909 | 2.259 | 0.857 | 0.526 |
| 23 | 1.878 | 2.211 | 0.866 | 0.538 |
| 24 (kept) | 1.848 | 2.235 | 0.866 | 0.539 |
| 25 | 1.832 | 2.204 | 0.864 | 0.538 |
| 26 | 1.798 | 2.213 | 0.863 | 0.538 |
| 27 | 1.780 | 2.202 | 0.866 | 0.539 |
| 28 | 1.778 | 2.194 | 0.866 | 0.539 |
| 29 | 1.763 | 2.193 | 0.865 | 0.537 |
| 30 | 1.749 | 2.187 | 0.867 | 0.539 |

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

- **Construction-PPE (Ultralytics)** (AGPL-3.0): Construction-PPE dataset by Ultralytics, https://docs.ultralytics.com/datasets/detect/construction-ppe/
- **Construction Safety (Roboflow 100)** (CC BY 4.0): construction safety dataset, Roboflow Universe (Roboflow 100 benchmark), CC BY 4.0, https://universe.roboflow.com/object-detection/construction-safety-gsnvb
- **CHV: Colour Helmet and Vest (Wang et al., 2021)** (free to use, with citation (no formal licence)): Z. Wang, Y. Wu, L. Yang, A. Thirunavukarasu, C. Evison, Y. Zhao, "Fast personal protective equipment detection for real construction sites using deep learning approaches", Sensors 21(10), 2021. https://github.com/ZijianWang-ZW/PPE_detection
- **GDUT-HWD: Hardhat Wearing Detection (Wu et al., 2019)** (no licence stated (published for research)): J. Wu, N. Cai, W. Chen, H. Wang, G. Wang, "Automatic detection of hardhats worn by construction personnel: A deep learning approach and benchmark dataset", Automation in Construction 106, 2019. https://github.com/wujixiu/helmet-detection (copy used: Hugging Face worstprogrammer/GDUT-HWD)
- **SH17: Human Safety and PPE in Manufacturing (Ahmad & Rahimi, 2024)** (CC BY-NC-SA 4.0 (non-commercial); images under the Pexels licence): H. M. Ahmad, A. Rahimi, "SH17: A dataset for human safety and personal protective equipment detection in manufacturing industry", Journal of Safety Science and Resilience, 2024. https://github.com/ahmadmughees/SH17dataset. Photos from Pexels (https://www.pexels.com).

Trained with Ultralytics YOLO (AGPL-3.0). Fine for learning and portfolio use; a commercial product
needs an Ultralytics licence or a differently licensed detector and data.
