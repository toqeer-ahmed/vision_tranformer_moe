# Phase 11 Final Reproducibility Audit: Attention U-Net (Oktay et al., 2018)

**Document Identifier**: `research_audit/phase11_attention_unet_final/PHASE11_FINAL_REPRODUCIBILITY_AUDIT.md`  
**Workspace**: `D:\Study Material\Research\CV\vision_transformer_research`  
**Auditor**: Senior Computer Vision Research Engineer & Scientific Auditor  
**Date**: 2026-11-26  
**Status**: Formally Audited, Recalculated, and Verified  

---

## 1. Executive Summary

This audit provides complete, independent verification of **Phase 11 (Attention U-Net)** experimental artifacts downloaded from Kaggle GPU execution across **ten matched random seeds** ($\mathcal{S} = \{42, 123, 314, 777, 999, 1001, 2026, 2027, 2718, 9001\}$). 

All raw confusion matrices, checkpoint weights, configuration files, in-domain test logs, and zero-shot BUS-BRA evaluation outputs were subjected to programmatic recalculation. No aggregated values were assumed.

### Core Audit Verdicts:
1. **Experiment Completeness**: All 10 seeds executed to completion, saving exact $125,683,035$-byte checkpoint files (`best_model.pth`), TensorBoard event logs, and in-domain BUSI test metrics.
2. **Architectural Parity**: Verified exact model architecture in [models/attention_unet.py](file:///D:/Study%20Material/Research/CV/vision_transformer_research/models/attention_unet.py) containing **$31,389,295$ trainable parameters** ($+351,532$ parameters / $+1.13\%$ over Standard U-Net's $31,037,763$, and matched to Vanilla Swin-Tiny's $29,290,365$).
3. **Data Integrity & Leakage Controls**:
   - In-Domain BUSI ($N=780$): Strict seed-controlled splits (Train: 624, Val: 78, Test: 78).
   - Out-of-Domain Zero-Shot BUS-BRA ($N=1,063$): Completely unseen during training, validation, early stopping, and hyperparameter tuning. Total ground truth foreground volume: **$8,146,649$ pixels**.
4. **Recalculation Parity**: Programmatic recalculation from raw pixel confusion matrices (`class_tp`, `class_fp`, `class_fn`) matched provided CSV files to machine precision ($\max |\Delta \text{IoU}| = 5.55 \times 10^{-17}$, $\max |\Delta \text{Dice}| = 0.00$).

---

## 2. Experimental Run Verification

Each of the 10 runs was inspected for configuration consistency, training duration, and checkpoint integrity:

| Seed | Run Directory | Checkpoint (`best_model.pth`) | In-Domain BUSI Test IoU (%) | Zero-Shot BUS-BRA Fg IoU (%) | Zero-Shot Volume Ratio | Run Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **42** | `attention_unet_seed42_20261125_205718` | 125,683,035 bytes | $57.50$ | $28.42$ | $0.336\times$ | Complete (Domain Collapse) |
| **123** | `attention_unet_seed123_20261125_210952` | 125,683,035 bytes | $55.61$ | $30.89$ | $0.373\times$ | Complete (Domain Collapse) |
| **314** | `attention_unet_seed314_20261125_211908` | 125,683,035 bytes | $56.20$ | $46.68$ | $0.625\times$ | Complete |
| **777** | `attention_unet_seed777_20261125_213310` | 125,683,035 bytes | $57.74$ | $51.19$ | $0.690\times$ | Complete |
| **999** | `attention_unet_seed999_20261125_214622` | 125,683,035 bytes | $53.21$ | $48.88$ | $0.740\times$ | Complete |
| **1001** | `attention_unet_seed1001_20261125_215702` | 125,683,035 bytes | $53.98$ | $49.76$ | $0.816\times$ | Complete |
| **2026** | `attention_unet_seed2026_20261125_221122` | 125,683,035 bytes | $60.41$ | $51.84$ | $0.789\times$ | Complete |
| **2027** | `attention_unet_seed2027_20261125_222545` | 125,683,035 bytes | $57.85$ | $46.17$ | $0.657\times$ | Complete |
| **2718** | `attention_unet_seed2718_20261125_223947` | 125,683,035 bytes | $56.63$ | $53.36$ | $0.725\times$ | Complete |
| **9001** | `attention_unet_seed9001_20261125_225334` | 125,683,035 bytes | $51.88$ | $46.30$ | $0.625\times$ | Complete |

---

## 3. Training Protocol & Hyperparameter Audit

All runs conformed to the frozen Phase 10 / Phase 11 experimental protocol:
- **Optimizer**: AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $= 0.05$).
- **Base Learning Rate**: $1 \times 10^{-4}$ with `CosineAnnealingLR` decaying to $1 \times 10^{-6}$.
- **Loss Function**: Focal Tversky Loss ($\alpha = 0.7, \beta = 0.3, \gamma = 0.75$).
- **Batch Size**: 16.
- **Maximum Epochs**: 50, with early stopping patience of 10 epochs monitored on in-domain validation loss.
- **Input Resolution**: $224 \times 224 \times 3$, normalized with ImageNet channel statistics.
- **Data Augmentation**: Horizontal flip ($p=0.5$), Vertical flip ($p=0.5$), Random rotate 90 degrees ($p=0.5$).
- **Precision**: PyTorch Automatic Mixed Precision (AMP `fp16`).

---

## 4. Raw Confusion Matrix Recalculations

The zero-shot BUS-BRA evaluation metrics were recalculated directly from the raw pixel counts stored in [attention_unet_reproducibility_metrics.json](file:///D:/Study%20Material/Research/CV/vision_transformer_research/results/phase11_attention_unet/attention_unet_reproducibility_metrics.json):

$$\text{IoU}_{\text{FG}} = \frac{\text{TP}_{\text{FG}}}{\text{TP}_{\text{FG}} + \text{FP}_{\text{FG}} + \text{FN}_{\text{FG}}}, \quad \text{Dice}_{\text{FG}} = \frac{2 \cdot \text{TP}_{\text{FG}}}{2 \cdot \text{TP}_{\text{FG}} + \text{FP}_{\text{FG}} + \text{FN}_{\text{FG}}}$$

$$\text{Precision}_{\text{FG}} = \frac{\text{TP}_{\text{FG}}}{\text{TP}_{\text{FG}} + \text{FP}_{\text{FG}}}, \quad \text{Recall}_{\text{FG}} = \frac{\text{TP}_{\text{FG}}}{\text{TP}_{\text{FG}} + \text{FN}_{\text{FG}}}$$

### Recalculated Per-Seed Zero-Shot Results (BUS-BRA, N=1,063):

| Seed | TP (FG) | FP (FG) | FN (FG) | Fg IoU (%) | Fg Dice (%) | Fg Precision (%) | Fg Recall (%) | Predicted Pixels | Volume Ratio |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | $2,407,829$ | $325,913$ | $5,738,820$ | $28.42$ | $44.26$ | $88.08$ | $29.56$ | $2,733,742$ | $0.336\times$ |
| **123** | $2,640,067$ | $400,847$ | $5,506,582$ | $30.89$ | $47.20$ | $86.82$ | $32.41$ | $3,040,914$ | $0.373\times$ |
| **314** | $4,214,306$ | $881,169$ | $3,932,343$ | $46.68$ | $63.65$ | $82.71$ | $51.73$ | $5,095,475$ | $0.625\times$ |
| **777** | $4,662,393$ | $960,865$ | $3,484,256$ | $51.19$ | $67.72$ | $82.91$ | $57.23$ | $5,623,258$ | $0.690\times$ |
| **999** | $4,655,078$ | $1,376,117$ | $3,491,571$ | $48.88$ | $65.67$ | $77.18$ | $57.14$ | $6,031,195$ | $0.740\times$ |
| **1001** | $4,914,785$ | $1,730,684$ | $3,231,864$ | $49.76$ | $66.45$ | $73.96$ | $60.33$ | $6,645,469$ | $0.816\times$ |
| **2026** | $4,974,992$ | $1,450,388$ | $3,171,657$ | $51.84$ | $68.28$ | $77.43$ | $61.07$ | $6,425,380$ | $0.789\times$ |
| **2027** | $4,264,286$ | $1,090,013$ | $3,882,363$ | $46.17$ | $63.17$ | $79.64$ | $52.34$ | $5,354,299$ | $0.657\times$ |
| **2718** | $4,889,613$ | $1,016,833$ | $3,257,036$ | $53.36$ | $69.59$ | $82.78$ | $60.02$ | $5,906,446$ | $0.725\times$ |
| **9001** | $4,189,957$ | $901,844$ | $3,956,692$ | $46.30$ | $63.30$ | $82.29$ | $51.43$ | $5,091,801$ | $0.625\times$ |

---

## 5. Summary of Audited Datasets and Master Artifacts

All recalculated results and metadata have been written to the following verified files:
- Master CSV: [FOUR_MODEL_BENCHMARK_MASTER.csv](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/phase11_attention_unet_final/FOUR_MODEL_BENCHMARK_MASTER.csv)
- Per-Seed Recalculated: [PHASE11_PER_SEED_RECALCULATED.csv](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/phase11_attention_unet_final/PHASE11_PER_SEED_RECALCULATED.csv)
- In-Domain BUSI Test Metrics: [ATTENTION_UNET_INDOMAIN_TEST_RESULTS.csv](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/phase11_attention_unet_final/ATTENTION_UNET_INDOMAIN_TEST_RESULTS.csv)
- Results Directory Copy: [ATTENTION_UNET_SUMMARY_RESULTS.csv](file:///D:/Study%20Material/Research/CV/vision_transformer_research/results/phase11_attention_unet/ATTENTION_UNET_SUMMARY_RESULTS.csv)

The Phase 11 experimental artifacts are officially verified and ready for downstream statistical testing.
