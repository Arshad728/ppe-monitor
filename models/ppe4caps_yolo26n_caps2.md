# Model card: ppe4caps_yolo26n_caps2

Trained 2026-09-29 21:14 on mps (arm64, Darwin 27.0).

| Setting | Value |
|---|---|
| Starting point | `models/ppe4_yolo26n_finetune.pt` (fine-tuned from an already trained model) |
| Data | `datasets/ppe4caps` - see its report.md |
| Image size / batch | 640 / 16 |
| Epochs | 15 of 15 (early-stop patience 5; the last 3 without mosaic) |
| Optimiser | AdamW, learning rate 0.0002 falling to 2e-05; warm-up 1 epoch(s), biases from 0 |
| Training time | 102 min |
| Software | ultralytics 8.4.159, torch 2.14.0, Python 3.11.16 |
| Working confidence threshold | **0.336** (best F1 on val; used for precision/recall below) |

## Validation split (702 images)

| Class | Objects | AP@50 | Precision | Recall | Recall: small / medium / large |
|---|---|---|---|---|---|
| person | 1961 | 0.893 | 0.878 | 0.831 | 0.26 (69) / 0.63 (372) / 0.91 (1520) |
| helmet | 885 | 0.877 | 0.889 | 0.842 | 0.70 (221) / 0.88 (486) / 0.92 (178) |
| vest | 362 | 0.824 | 0.913 | 0.751 | 0.04 (27) / 0.66 (106) / 0.88 (229) |
| **all** | 3208 | **0.865** | | | |

Ultralytics' own figures for the same split: mAP@50 0.871, mAP@50-95 0.541.

## Test split (628 images)

| Class | Objects | AP@50 | Precision | Recall | Recall: small / medium / large |
|---|---|---|---|---|---|
| person | 2587 | 0.902 | 0.875 | 0.850 | 0.37 (123) / 0.74 (645) / 0.92 (1819) |
| helmet | 1714 | 0.921 | 0.880 | 0.862 | 0.73 (448) / 0.92 (1028) / 0.87 (238) |
| vest | 311 | 0.835 | 0.806 | 0.762 | 0.29 (14) / 0.66 (67) / 0.82 (230) |
| **all** | 4612 | **0.886** | | | |

Ultralytics' own figures for the same split: mAP@50 0.887, mAP@50-95 0.579.

### Test split by source

| Source | Images | mAP@50 | person | helmet | vest |
|---|---|---|---|---|---|
| chv | 7 | 0.832 | 0.893 | 0.964 | 0.638 |
| extra | 428 | 0.804 | 0.914 | 0.920 | 0.577 |
| gdut-hwd | 320 | 0.919 | 0.916 | 0.922 | nan |
| ppe3 | 200 | 0.877 | 0.830 | 0.922 | 0.880 |
| sh17 | 101 | 0.811 | 0.895 | 0.884 | 0.652 |
| caps | 447 | 0.833 | 0.867 | nan | 0.799 |

## Overfitting check

- **No sign of overfitting in the kept model.**
- Epochs run: 15; kept: epoch 12, the highest val mAP@50-95 (0.541). Training went on 3 epoch(s) past the best one without beating it, then stopped.
- Validation loss 2.027 at the kept epoch, 2.032 at the last; training loss 2.018 and 1.778.
- mAP@50 on 300 training images 0.943, on validation 0.865 (gap +0.078; above 0.08 would mean memorising).

| Epoch | Train loss | Val loss | Val mAP@50 | Val mAP@50-95 |
|---|---|---|---|---|
| 1 | 2.158 | 2.093 | 0.860 | 0.531 |
| 2 | 2.120 | 2.058 | 0.862 | 0.536 |
| 3 | 2.083 | 2.072 | 0.865 | 0.535 |
| 4 | 2.073 | 2.051 | 0.863 | 0.536 |
| 5 | 2.078 | 2.050 | 0.866 | 0.536 |
| 6 | 2.051 | 2.065 | 0.865 | 0.533 |
| 7 | 2.056 | 2.039 | 0.867 | 0.537 |
| 8 | 2.034 | 2.028 | 0.868 | 0.538 |
| 9 | 2.023 | 2.024 | 0.868 | 0.540 |
| 10 | 2.013 | 2.024 | 0.869 | 0.541 |
| 11 | 2.014 | 2.023 | 0.870 | 0.541 |
| 12 (kept) | 2.018 | 2.027 | 0.870 | 0.541 |
| 13 | 1.807 | 2.042 | 0.863 | 0.534 |
| 14 | 1.777 | 2.033 | 0.866 | 0.532 |
| 15 | 1.778 | 2.032 | 0.866 | 0.535 |

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
- **Open Images V7: people in hats and caps (data/caps/)** (boxes CC BY 4.0, Google LLC (Open Images); photos CC BY 2.0 (Flickr authors, see credits.csv)): A. Kuznetsova et al., "The Open Images Dataset V4", IJCV 2020; https://storage.googleapis.com/openimages/web/index.html. Photo credits: datasets/ppe4caps/credits.csv

Trained with Ultralytics YOLO (AGPL-3.0). Fine for learning and portfolio use; a commercial product
needs an Ultralytics licence or a differently licensed detector and data.
