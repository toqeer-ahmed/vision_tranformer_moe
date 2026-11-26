# Phase 12 Experiment Protocol: Pretrained ResNet-50 U-Net

## 1. Experimental Overview

- **Hypothesis**: ImageNet-1K pretraining provides strong semantic low-level and mid-level visual features (edges, textures, object boundaries) that may mitigate the severe cross-dataset domain penalty observed in randomly initialized CNNs (Standard U-Net: $50.79\%$, Attention U-Net: $45.35\%$).
- **Primary Controlled Comparison**: Swin-Tiny (ImageNet-1K pretrained ViT) vs. ResNet-50 U-Net (ImageNet-1K pretrained CNN).
- **Execution Target**: Kaggle GPU environment (T4 x2 or P100) running [notebooks/Phase12_ResNet50_UNet_Reproducibility.ipynb](file:///D:/Study%20Material/Research/CV/vision_transformer_research/notebooks/Phase12_ResNet50_UNet_Reproducibility.ipynb).

---

## 2. Dataset Protocol and Split Integrity

### A. In-Domain Training Dataset: BUSI
- **Total Valid Images**: 780 ultrasound images with paired binary masks.
- **Data Partitioning**: 80% train (624 images), 10% validation (78 images), 10% in-domain test (78 images).
- **Partitioning Mechanism**: Deterministic pseudo-random split controlled by `seed` across 10 matched seeds:
  `[42, 123, 314, 777, 999, 1001, 2026, 2027, 2718, 9001]`.
- **Target Classes**: Multi-class ($C=3$: 0 = Background, 1 = Benign, 2 = Malignant).

### B. External Zero-Shot Evaluation Dataset: BUS-BRA
- **Total Unseen Images**: 1,063 high-resolution breast ultrasound scans ($N = 1,063$).
- **Zero-Shot Boundary**:
  - BUS-BRA is **never** seen during training.
  - BUS-BRA is **never** used for validation or early stopping.
  - BUS-BRA is **never** used for hyperparameter tuning or decision threshold calibration.
- **Evaluation Target**: Binary foreground lesion segmentation ($C=1$ vs $C=0$).

---

## 3. Training Hyperparameters and Parity Controls

All optimization hyperparameters are matched to Phase 9B, Phase 10, and Phase 11:

| Hyperparameter | Value | Scientific Justification |
|---|---|---|
| **Optimizer** | AdamW | Decoupled weight decay provides stable gradient updates for fine-tuning |
| **Learning Rate ($lr$)** | $1.0 \times 10^{-4}$ ($0.0001$) | Standard fine-tuning rate across all audited phases |
| **Weight Decay ($wd$)** | $0.05$ | Regularization identical to Swin-Tiny training protocol |
| **LR Scheduler** | `CosineAnnealingLR` | Smooth decay from $1e-4$ to $1e-6$ over 50 epochs |
| **Batch Size** | 16 | Matched batch statistics across all models |
| **Input Spatial Size** | $224 \times 224 \times 3$ | Standard ImageNet resolution |
| **Loss Function** | Combined Focal Tversky Loss | $\alpha=0.7, \beta=0.3, \gamma=4/3$, handles foreground-background class imbalance |
| **Early Stopping Patience** | 10 epochs | Checkpoint selected by peak validation mIoU on held-in BUSI validation set |
| **Max Epochs** | 50 | Allows full convergence while early stopping prevents overfitting |

---

## 4. Evaluation and Output Logging Standards

For each seed $s \in \{42, 123, \dots, 9001\}$:
1. Model checkpoint `best_model.pth` is saved based on validation mIoU.
2. In-domain evaluation on BUSI held-in test split ($N = 78$) is recorded in `test_metrics.json`.
3. Out-of-distribution evaluation on BUS-BRA ($N = 1,063$) evaluates:
   - Foreground IoU (FG IoU)
   - Foreground Dice (FG Dice)
   - Foreground Precision
   - Foreground Recall
   - Pixel Accuracy
   - Predicted Volume, Ground Truth Volume, and Volume Ratio ($\frac{\text{Predicted Foreground}}{\text{Ground Truth Foreground}}$)
4. Per-seed metrics are compiled into `RESNET50_UNET_PER_SEED_RESULTS.csv`.
5. Aggregate summary statistics (mean $\pm$ sample std) are recorded in `RESNET50_UNET_SUMMARY_RESULTS.csv`.
6. All artifacts are packaged into `results_phase12_resnet50_unet.zip`.
