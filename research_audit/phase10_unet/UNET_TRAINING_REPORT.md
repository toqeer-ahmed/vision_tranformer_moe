# Standard U-Net Training Protocol & Full 10-Seed Empirical Training Report

**Experiment Identifier**: `phase10_unet`  
**Auditor**: Senior Computer Vision Research Engineer & Scientific Auditor  
**Repository**: `D:\Study Material\Research\CV\vision_transformer_research`  
**Execution Environment**: Kaggle GPU Dual T4 / P100  
**Status**: 100% Complete across 10 Matched Seeds  

---

## 1. Common Experimental Controls & Hyperparameters

To ensure strict comparability with the audited Swin transformer experiments, the Standard U-Net training pipeline utilizes identical optimization hyperparameters:

| Hyperparameter | Phase 9B Vanilla Swin-Tiny | Phase 9B Standard Swin-MoE | Phase 10 Standard U-Net | Control Status |
| :--- | :---: | :---: | :---: | :---: |
| **Model Family** | Hierarchical ViT | Hierarchical ViT + MoE | 2D CNN (Ronneberger) | Baseline Comparison |
| **Pretrained Weights** | ImageNet-1K (`pretrained=True`) | ImageNet-1K (`warm_init=True`) | None (`from_scratch=True`) | Controlled Inductive Bias |
| **Input Resolution** | $224 \times 224$ | $224 \times 224$ | $224 \times 224$ | **Identical** |
| **Batch Size** | $16$ | $16$ | $16$ | **Identical** |
| **Optimizer** | AdamW | AdamW | AdamW | **Identical** |
| **Initial Learning Rate** | $1.0 \times 10^{-4}$ | $1.0 \times 10^{-4}$ | $1.0 \times 10^{-4}$ | **Identical** |
| **Weight Decay** | $0.05$ | $0.05$ | $0.05$ | **Identical** |
| **LR Scheduler** | CosineAnnealingLR | CosineAnnealingLR | CosineAnnealingLR | **Identical** |
| **Max Epochs** | $50$ | $50$ | $50$ | **Identical** |
| **Early Stopping** | Patience $= 10$ | Patience $= 10$ | Patience $= 10$ | **Identical** |
| **Loss Function** | Focal Tversky Loss | Focal Tversky + $\mathcal{L}_{\text{aux}}$ | Focal Tversky Loss | **Identical Task Objective** |
| **Mixed Precision** | PyTorch AMP | PyTorch AMP | PyTorch AMP | **Identical** |
| **Checkpoint Metric** | Validation mIoU | Validation mIoU | Validation mIoU | **Identical** |

---

## 2. 10-Seed Training Dynamics & Convergence Summary

All 10 training runs converged without divergence or NaN gradients. Early stopping reliably identified optimal validation checkpoints:

| Seed | Output Directory | Checkpoint | Size (MB) | Total Batches Tracked | In-Domain BUSI Test mIoU (%) | In-Domain BUSI Fg IoU (%) | In-Domain BUSI Fg Dice (%) | In-Domain BUSI Pixel Acc (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | `unet_seed42_20261124_100829` | `best_model.pth` | $124.24$ | $1,521$ (39 epochs) | $82.66\%$ | $52.14\%$ | $68.54\%$ | $96.02\%$ |
| **123** | `unet_seed123_20261124_102223` | `best_model.pth` | $124.24$ | $1,443$ (37 epochs) | $85.39\%$ | $60.00\%$ | $75.00\%$ | $96.37\%$ |
| **314** | `unet_seed314_20261124_103546` | `best_model.pth` | $124.24$ | $1,482$ (38 epochs) | $86.17\%$ | $62.04\%$ | $76.57\%$ | $96.65\%$ |
| **777** | `unet_seed777_20261124_104825` | `best_model.pth` | $124.24$ | $1,560$ (40 epochs) | $84.44\%$ | $57.90\%$ | $73.34\%$ | $95.69\%$ |
| **999** | `unet_seed999_20261124_110132` | `best_model.pth` | $124.24$ | $1,443$ (37 epochs) | $83.71\%$ | $55.71\%$ | $71.55\%$ | $95.67\%$ |
| **1001** | `unet_seed1001_20261124_111452` | `best_model.pth` | $124.24$ | $1,521$ (39 epochs) | $84.37\%$ | $57.86\%$ | $73.31\%$ | $95.53\%$ |
| **2026** | `unet_seed2026_20261124_112346` | `best_model.pth` | $124.24$ | $1,365$ (35 epochs) | $83.58\%$ | $55.11\%$ | $71.06\%$ | $95.85\%$ |
| **2027** | `unet_seed2027_20261124_113710` | `best_model.pth` | $124.24$ | $1,482$ (38 epochs) | $81.03\%$ | $48.70\%$ | $65.50\%$ | $94.69\%$ |
| **2718** | `unet_seed2718_20261124_115033` | `best_model.pth` | $124.24$ | $1,521$ (39 epochs) | $84.29\%$ | $57.37\%$ | $72.91\%$ | $95.76\%$ |
| **9001** | `unet_seed9001_20261124_120035` | `best_model.pth` | $124.24$ | $1,521$ (39 epochs) | $82.89\%$ | $53.76\%$ | $69.92\%$ | $95.20\%$ |
| **Mean** | — | — | — | **$1,486$ (~38 ep)** | **$83.85\%$** | **$56.06\%$** | **$71.77\%$** | **$95.74\%$** |
| **Std** | — | — | — | **$55$ (1.4 ep)** | **$1.45\%$** | **$3.88\%$** | **$3.22\%$** | **$0.56\%$** |

---

## 3. Checkpoint & Architecture Integrity Audit

Every saved `best_model.pth` checkpoint was loaded and verified:
1. **Parameter Completeness**: All 64 weight/bias tensors of canonical U-Net were verified intact.
2. **Compact Serialized Footprint**: Exactly $124.24$ MB per checkpoint (state_dict only, no optimizer memory bloat).
3. **Execution Time**: The 10-seed experiment completed on Kaggle GPU in approximately 1 hour and 55 minutes (~11.5 minutes per seed).
