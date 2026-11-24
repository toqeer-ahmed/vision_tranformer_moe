# Phase 10: Final Reproducibility & Pipeline Integrity Audit

**Experiment Identifier**: `phase10_unet`  
**Auditor**: Senior Computer Vision Research Engineer & Scientific Auditor  
**Repository**: `D:\Study Material\Research\CV\vision_transformer_research`  
**Date**: 2026-11-25  
**Target Architecture**: Canonical 2D U-Net (Ronneberger et al., 2015)  

---

## 1. Executive Audit Summary

A comprehensive, end-to-end technical audit was conducted on all ten experimental runs of **Phase 10: Canonical Standard U-Net**. Every checkpoint, configuration file, data manifest, seed-generation protocol, and raw confusion matrix was independently inspected and mathematically validated from disk artifacts.

### Key Audit Findings:
1. **Parameter Count Reconciliation**:
   - The exact parameter count of the audited U-Net implementation is **$31,037,763$ trainable parameters** ($100\%$ trainable, zero frozen weights).
   - *Clarification on the earlier $7.76\text{M}$ parameter estimate*: An earlier theoretical calculation assumed a reduced base channel capacity ($C_{\text{base}} = 32$). When $C_{\text{base}} = 32$, the parameter count is exactly $7,763,107$. However, the canonical Ronneberger et al. (2015) specification defines $C_{\text{base}} = 64$ (stage channels: $64 \rightarrow 128 \rightarrow 256 \rightarrow 512 \rightarrow 1024$), which quadruples convolution weights due to quadratic channel scaling ($O(C^2)$), yielding **$31,037,763$ parameters**.
   - *Scientific advantage*: This establishes an exceptionally tight parameter match with **Vanilla Swin-Tiny** ($29,290,365$ parameters, a mere $+5.9\%$ delta), providing a well-controlled baseline to compare 2D convolutional inductive bias against hierarchical self-attention.
2. **Execution & Checkpoint Integrity**:
   - All 10 seeds ($\mathcal{S} = \{42, 123, 314, 777, 999, 1001, 2026, 2027, 2718, 9001\}$) completed training to early stopping without crashes or gradient divergence.
   - All 10 `best_model.pth` files ($124.24$ MB each) are verified present, complete, and fully loadable into PyTorch.
3. **Dataset Separation & Split Consistency**:
   - BUSI dataset ($N=780$ pairs): Partitioned deterministically via `random_split(..., generator=torch.Generator().manual_seed(seed))` into 624 train ($80\%$), 78 validation ($10\%$), and 78 in-domain test ($10\%$). These splits are mathematically identical to Phase 9B.
   - BUS-BRA dataset ($N=1,063$ pairs): Kept strictly out-of-domain. Never seen during training, validation, or checkpoint selection.

---

## 2. Checkpoint & Configuration Verification Matrix

| Seed | Config Path | Output Directory | Checkpoint Verified | Checkpoint Size | Total Batches Tracked | Best Val Epoch |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **42** | `configs/phase10_unet/unet_seed42.yaml` | `outputs/unet_seed42_20261124_100829` | `best_model.pth` | $124.24$ MB | $1,521$ ($39$ ep) | Epoch 29 |
| **123** | `configs/phase10_unet/unet_seed123.yaml` | `outputs/unet_seed123_20261124_102223` | `best_model.pth` | $124.24$ MB | $1,443$ ($37$ ep) | Epoch 27 |
| **314** | `configs/phase10_unet/unet_seed314.yaml` | `outputs/unet_seed314_20261124_103546` | `best_model.pth` | $124.24$ MB | $1,482$ ($38$ ep) | Epoch 28 |
| **777** | `configs/phase10_unet/unet_seed777.yaml` | `outputs/unet_seed777_20261124_104825` | `best_model.pth` | $124.24$ MB | $1,560$ ($40$ ep) | Epoch 30 |
| **999** | `configs/phase10_unet/unet_seed999.yaml` | `outputs/unet_seed999_20261124_110132` | `best_model.pth` | $124.24$ MB | $1,443$ ($37$ ep) | Epoch 27 |
| **1001** | `configs/phase10_unet/unet_seed1001.yaml` | `outputs/unet_seed1001_20261124_111452` | `best_model.pth` | $124.24$ MB | $1,521$ ($39$ ep) | Epoch 29 |
| **2026** | `configs/phase10_unet/unet_seed2026.yaml` | `outputs/unet_seed2026_20261124_112346` | `best_model.pth` | $124.24$ MB | $1,365$ ($35$ ep) | Epoch 25 |
| **2027** | `configs/phase10_unet/unet_seed2027.yaml` | `outputs/unet_seed2027_20261124_113710` | `best_model.pth` | $124.24$ MB | $1,482$ ($38$ ep) | Epoch 28 |
| **2718** | `configs/phase10_unet/unet_seed2718.yaml` | `outputs/unet_seed2718_20261124_115033` | `best_model.pth` | $124.24$ MB | $1,521$ ($39$ ep) | Epoch 29 |
| **9001** | `configs/phase10_unet/unet_seed9001.yaml` | `outputs/unet_seed9001_20261124_120035` | `best_model.pth` | $124.24$ MB | $1,521$ ($39$ ep) | Epoch 29 |

---

## 3. Data Split & Preprocessing Verification

### A. Preprocessing Pipeline
- **Resolution**: $224 \times 224$ pixels via bilinear interpolation.
- **Color Normalization**: Standard ImageNet channel-wise normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
- **Target Masks**: Binary conversion where pixel intensity $> 127$ maps to $1$ (lesion tissue), else $0$ (background tissue).

### B. Training Data Augmentation Suite
Implemented via Albumentations:
- `HorizontalFlip(p=0.5)`
- `VerticalFlip(p=0.5)`
- `RandomRotate90(p=0.5)`
- `Affine(scale=(0.85, 1.15), rotate=(-30, 30), translate_percent=(-0.1, 0.1), p=0.5)`
- `ElasticTransform(alpha=1, sigma=50, p=0.3)`
- `GridDistortion(p=0.3)`
- `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.3)`

Validation and test splits strictly disable augmentations and apply only resizing and normalization.

---

## 4. Mathematical Recalculation from Raw Confusion Matrices

All summary metrics were independently recalculated directly from the raw pixel counts stored in `unet_reproducibility_metrics.json`:

$$\text{IoU}_{\text{fg}} = \frac{\text{TP}_1}{\text{TP}_1 + \text{FP}_1 + \text{FN}_1}, \quad \text{Dice}_{\text{fg}} = \frac{2 \cdot \text{TP}_1}{2 \cdot \text{TP}_1 + \text{FP}_1 + \text{FN}_1}$$
$$\text{Precision}_{\text{fg}} = \frac{\text{TP}_1}{\text{TP}_1 + \text{FP}_1}, \quad \text{Recall}_{\text{fg}} = \frac{\text{TP}_1}{\text{TP}_1 + \text{FN}_1}, \quad \text{Acc} = \frac{\sum \text{TP}}{\sum (\text{TP} + \text{FP})}$$

| Seed | Raw $\text{TP}_1$ | Raw $\text{FP}_1$ | Raw $\text{FN}_1$ | Recalculated Fg IoU | Recalculated Fg Dice | Recalculated Fg Precision | Recalculated Fg Recall |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | $5,491,893$ | $1,778,475$ | $2,654,756$ | $0.553332$ ($55.33\%$) | $0.712446$ ($71.24\%$) | $0.755380$ ($75.54\%$) | $0.674129$ ($67.41\%$) |
| **123** | $4,510,885$ | $988,725$ | $3,635,764$ | $0.493782$ ($49.38\%$) | $0.661117$ ($66.11\%$) | $0.820219$ ($82.02\%$) | $0.553710$ ($55.37\%$) |
| **314** | $5,203,836$ | $1,467,034$ | $2,942,813$ | $0.541295$ ($54.13\%$) | $0.702390$ ($70.24\%$) | $0.780084$ ($78.01\%$) | $0.638770$ ($63.88\%$) |
| **777** | $4,536,897$ | $1,025,124$ | $3,609,752$ | $0.494657$ ($49.47\%$) | $0.661901$ ($66.19\%$) | $0.815674$ ($81.57\%$) | $0.556910$ ($55.69\%$) |
| **999** | $3,678,350$ | $587,495$ | $4,468,299$ | $0.421151$ ($42.12\%$) | $0.592690$ ($59.27\%$) | $0.862290$ ($86.23\%$) | $0.451519$ ($45.15\%$) |
| **1001** | $5,300,757$ | $2,240,111$ | $2,845,892$ | $0.510308$ ($51.03\%$) | $0.675767$ ($67.58\%$) | $0.702880$ ($70.29\%$) | $0.650668$ ($65.07\%$) |
| **2026** | $4,657,171$ | $979,484$ | $3,489,478$ | $0.510338$ ($51.03\%$) | $0.675793$ ($67.58\%$) | $0.826300$ ($82.63\%$) | $0.571666$ ($57.17\%$) |
| **2027** | $4,696,898$ | $1,134,166$ | $3,449,751$ | $0.506118$ ($50.61\%$) | $0.672083$ ($67.21\%$) | $0.805588$ ($80.56\%$) | $0.576537$ ($57.65\%$) |
| **2718** | $5,123,892$ | $1,322,465$ | $3,022,757$ | $0.541145$ ($54.11\%$) | $0.702263$ ($70.23\%$) | $0.794895$ ($79.49\%$) | $0.628968$ ($62.90\%$) |
| **9001** | $4,514,040$ | $763,605$ | $3,632,609$ | $0.506655$ ($50.67\%$) | $0.672556$ ($67.26\%$) | $0.855470$ ($85.55\%$) | $0.554084$ ($55.41\%$) |

**Audit Conclusion**: 100% of reported CSV values match the raw pixel confusion matrices with 0 discrepancy.
