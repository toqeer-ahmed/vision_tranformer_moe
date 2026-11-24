# Phase 9B Reproducibility & Pipeline Audit

**Experiment Identifier**: `phase9_auxloss_corrected`  
**Auditor**: Senior Computer Vision Research Engineer & Scientific Auditor  
**Repository**: `D:\Study Material\Research\CV\vision_transformer_research`  
**Date of Audit**: 2026-11-24  
**Audit Scope**: Verification of all 20 experimental runs (10 Vanilla Swin-Tiny, 10 Standard-MLP Swin-MoE across matched random seeds).

---

## 1. Experimental Runs & Directory Verification

All 20 experiment directories were inspected in `outputs/results_phase9_auxloss_corrected/outputs/`. Every run completed all 50 epochs (or early stopping) and successfully generated all required training artifacts:

| Model | Seed | Directory Name | Checkpoint (`best_model.pth`) | Event Log | Log File | In-Domain `test_metrics.json` | Status |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Vanilla Swin** | 42 | `swin_vanilla_seed42_20261123_231215` | Present (325 MB) | Present | Present | Present | **Verified Complete** |
| **Vanilla Swin** | 123 | `swin_vanilla_seed123_20261123_232729` | Present (325 MB) | Present | Present | Present | **Verified Complete** |
| **Vanilla Swin** | 314 | `swin_vanilla_seed314_20261124_002352` | Present (325 MB) | Present | Present | Present | **Verified Complete** |
| **Vanilla Swin** | 777 | `swin_vanilla_seed777_20261123_235241` | Present (325 MB) | Present | Present | Present | **Verified Complete** |
| **Vanilla Swin** | 999 | `swin_vanilla_seed999_20261124_000903` | Present (325 MB) | Present | Present | Present | **Verified Complete** |
| **Vanilla Swin** | 1001 | `swin_vanilla_seed1001_20261124_005241` | Present (325 MB) | Present | Present | Present | **Verified Complete** |
| **Vanilla Swin** | 2026 | `swin_vanilla_seed2026_20261123_234321` | Present (325 MB) | Present | Present | Present | **Verified Complete** |
| **Vanilla Swin** | 2027 | `swin_vanilla_seed2027_20261124_010855` | Present (325 MB) | Present | Present | Present | **Verified Complete** |
| **Vanilla Swin** | 2718 | `swin_vanilla_seed2718_20261124_003709` | Present (325 MB) | Present | Present | Present | **Verified Complete** |
| **Vanilla Swin** | 9001 | `swin_vanilla_seed9001_20261124_012647` | Present (325 MB) | Present | Present | Present | **Verified Complete** |
| **Standard MoE** | 42 | `swin_standard_moe_seed42_20261123_231745` | Present (325 MB) | Present | Present | Present | **Verified Complete** |
| **Standard MoE** | 123 | `swin_standard_moe_seed123_20261123_233159` | Present (325 MB) | Present | Present | Present | **Verified Complete** |
| **Standard MoE** | 314 | `swin_standard_moe_seed314_20261124_003123` | Present (325 MB) | Present | Present | Present | **Verified Complete** |
| **Standard MoE** | 777 | `swin_standard_moe_seed777_20261123_235848` | Present (325 MB) | Present | Present | Present | **Verified Complete** |
| **Standard MoE** | 999 | `swin_standard_moe_seed999_20261124_001525` | Present (325 MB) | Present | Present | Present | **Verified Complete** |
| **Standard MoE** | 1001 | `swin_standard_moe_seed1001_20261124_010010` | Present (325 MB) | Present | Present | Present | **Verified Complete** |
| **Standard MoE** | 2026 | `swin_standard_moe_seed2026_20261123_234659` | Present (325 MB) | Present | Present | Present | **Verified Complete** |
| **Standard MoE** | 2027 | `swin_standard_moe_seed2027_20261124_011436` | Present (325 MB) | Present | Present | Present | **Verified Complete** |
| **Standard MoE** | 2718 | `swin_standard_moe_seed2718_20261124_004355` | Present (325 MB) | Present | Present | Present | **Verified Complete** |
| **Standard MoE** | 9001 | `swin_standard_moe_seed9001_20261124_013140` | Present (325 MB) | Present | Present | Present | **Verified Complete** |

**Storage Protection Verification**: All intermediate `last_checkpoint.pth` files (~975 MB each) were removed immediately upon run completion as specified in `train_swin_vanilla.py` and `train_swin_standard_moe.py`, successfully preventing disk overflow on Kaggle.

---

## 2. Dataset Partitioning & Benchmark Splits

### A. In-Domain Training & Validation (BUSI Dataset)
- **Source**: Breast Ultrasound Images Dataset (BUSI), $N = 780$ pairs across normal, benign, and malignant cases.
- **Split Formulation**: Implemented in `datasets/medical_dataset.py#L210-L217`:
  - Training: $80\%$ ($624$ images)
  - Validation: $10\%$ ($78$ images)
  - In-Domain Test: $10\%$ ($78$ images)
- **Random Split Generator**: `torch.Generator().manual_seed(seed)`.
  - **Audit Finding**: For each seed $s$, both Vanilla Swin and Standard-MLP MoE received the **identical pseudo-random partition** of the BUSI dataset, guaranteeing matched pairwise training conditions.

### B. Out-of-Distribution Zero-Shot Benchmark (BUS-BRA Dataset)
- **Source**: Breast Ultrasound Bi-Rads Assessment dataset (BUS-BRA), $N = 1,063$ images acquired on different clinical ultrasound scanners.
- **Benchmark Split**: $100\%$ ($1,063$ images) evaluated strictly zero-shot. Neither model had access to any BUS-BRA image during training or validation.

---

## 3. Training Pipeline & Hyperparameter Verification

Inspection of `configs/phase9_auxloss_corrected/` confirmed identical hyperparameters across all 20 configurations:

| Hyperparameter | Vanilla Swin-Tiny | Standard-MLP Swin-MoE | Match Status |
| :--- | :---: | :---: | :---: |
| **Backbone Architecture** | `microsoft/swin-tiny-patch4-window7-224` | `microsoft/swin-tiny-patch4-window7-224` | **Identical** |
| **Input Resolution** | $224 \times 224 \times 3$ | $224 \times 224 \times 3$ | **Identical** |
| **Batch Size** | $16$ | $16$ | **Identical** |
| **Optimizer** | AdamW | AdamW | **Identical** |
| **Initial Learning Rate** | $1.0 \times 10^{-4}$ | $1.0 \times 10^{-4}$ | **Identical** |
| **Weight Decay** | $0.05$ | $0.05$ | **Identical** |
| **LR Scheduler** | CosineAnnealingLR ($T_{\max}=50, \eta_{\min}=10^{-6}$) | CosineAnnealingLR ($T_{\max}=50, \eta_{\min}=10^{-6}$) | **Identical** |
| **Loss Function** | Combined Focal Tversky Loss ($\text{dice\_weight}=1.0$) | Combined Focal Tversky Loss + $\mathcal{L}_{\text{aux}}$ | **Intended Difference** |
| **Mixed Precision** | PyTorch AMP (`torch.amp.autocast('cuda')`) | PyTorch AMP (`torch.amp.autocast('cuda')`) | **Identical** |
| **Checkpoint Selection** | Highest Validation mIoU (`best_model.pth`) | Highest Validation mIoU (`best_model.pth`) | **Identical** |
| **Early Stopping** | Patience = 10 epochs | Patience = 10 epochs | **Identical** |
| **MoE Routing Configuration** | N/A | 4 Experts, Top-2 Routing, $\lambda_{\text{balance}}=0.01$ | **Verified Correct** |

---

## 4. Data Augmentation Pipeline Verification

The augmentations implemented in `datasets/medical_dataset.py#get_medical_transforms` were verified:
- **Training Pipeline**:
  - `A.Resize(224, 224)`
  - `A.HorizontalFlip(p=0.5)`
  - `A.VerticalFlip(p=0.5)`
  - `A.RandomRotate90(p=0.5)`
  - `A.Affine(scale=(0.85, 1.15), rotate=(-30, 30), translate_percent=(-0.1, 0.1), p=0.5)`
  - `A.ElasticTransform(alpha=1, sigma=50, p=0.3)`
  - `A.GridDistortion(p=0.3)`
  - `A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.3)`
  - `A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))`
  - `ToTensorV2()`
- **Validation & OOD Testing Pipeline**:
  - `A.Resize(224, 224)`
  - `A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))`
  - `ToTensorV2()` (Strictly no stochastic augmentations applied during evaluation).

---

## 5. Audit Conclusion

All 20 runs followed an identical, deterministic, and scientifically sound training protocol. There are no missing logs, corrupted checkpoints, or hyperparameter discrepancies between the matched pairs.
