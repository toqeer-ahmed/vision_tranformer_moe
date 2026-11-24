# Phase 10 Final Audit & Phase 11 Handover: Executive Supervisor Briefing

**Document Identifier**: `research_audit/phase10_unet_final/PHASE10_FINAL_SUPERVISOR_REPORT.md`  
**Auditor & Lead Architect**: Senior Computer Vision Research Engineer & Scientific Auditor  
**Repository**: `D:\Study Material\Research\CV\vision_transformer_research`  
**Date**: 2026-11-25  
**Audited Experiments**: Phase 9B (Vanilla Swin, Swin-MoE) & Phase 10 (Standard U-Net)  

---

## 1. Verified Empirical Results

Across ten matched random seeds ($\mathcal{S} = \{42, 123, 314, 777, 999, 1001, 2026, 2027, 2718, 9001\}$), all experimental artifacts have been audited, recalculated from raw pixel confusion matrices, and verified:

| Architecture | Model Family | Parameters | Pretraining | Zero-Shot Fg IoU (%) | Zero-Shot Fg Dice (%) | Fg Precision (%) | Fg Recall (%) | Volume Ratio vs GT |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vanilla Swin-Tiny** | Hierarchical ViT | $29.29\text{M}$ | ImageNet-1K | $57.50 \pm 4.74$ | $72.91 \pm 3.98$ | $70.44 \pm 6.61$ | $76.51 \pm 6.79$ | $1.10 \times \text{GT}$ |
| **Standard-MLP MoE** | ViT + MoE (Top-2) | $81.15\text{M}$ | Warm-Init | **$57.85 \pm 2.15$** | **$73.28 \pm 1.73$** | $69.92 \pm 4.33$ | **$77.35 \pm 3.73$** | $1.11 \times \text{GT}$ |
| **Standard U-Net** | Canonical 2D CNN | $31.04\text{M}$ | From Scratch | **$50.79 \pm 3.68$** | **$67.29 \pm 3.33$** | **$80.19 \pm 4.73$** | **$58.57 \pm 6.49$** | **$0.74 \times \text{GT}$** |

---

## 2. Corrected Statistical Conclusions

1. **Pretrained Transformers Significantly Outperform From-Scratch CNNs**:
   - At parameter parity ($31.04\text{M}$ vs. $29.29\text{M}$, a $+5.9\%$ delta), Standard U-Net lags behind Vanilla Swin-Tiny by **$-6.71\%$ Foreground IoU**.
   - Under rigorous **Holm-Bonferroni Family-Wise Error Rate correction** across all 14 planned comparisons, the deficit remains statistically significant ($t = -3.467$, Holm-adjusted $p = \mathbf{0.0425} < 0.05$, Cohen's $d_z = -1.10$).
2. **MoE Dominance in 10 out of 10 Seeds**:
   - Compared to Standard-MLP Swin-MoE, U-Net exhibits a **$-7.06\%$ IoU deficit** ($t = -6.342$, Holm-adjusted $p = \mathbf{0.0017}$, Wilcoxon $W = 0.0, p = 0.0273$). In all 10 matched seeds, Swin-MoE outperformed U-Net.
3. **Macro Metrics Borderline under Strict FWER Control**:
   - While raw $p$-values for macro mIoU ($p = 0.0154$) and macro mDice ($p = 0.0137$) suggest significance vs. Vanilla Swin, their Holm-adjusted $p$-values rise to **$p = 0.0549$**. We therefore conservatively report macro metric differences against Vanilla Swin as a non-significant trend.

---

## 3. Scientific Mechanisms & The Pretraining Confound

1. **The Precision-Recall Wedge is Driven by Volume Under-Prediction**:
   - U-Net does not suffer from random noise; it suffers from a **systematic volume under-prediction bias**.
   - On the 1,063 unseen BUS-BRA images, U-Net predicts only **$6.00$ million foreground pixels** against an actual ground truth of **$8.15$ million pixels** (volume ratio $= 0.74$).
   - This conservative core prediction artificially boosts Precision to $80.19\%$, but causes a catastrophic collapse in Recall to $58.57\%$ (missing $41.43\%$ of actual tumor volume).
2. **The Pretraining Confound**:
   - We must emphasize to the research community that this comparison reflects **both architecture and pretraining status**.
   - Swin initialized its weights from 1.28M natural images, providing robust boundary and contrast primitives. U-Net learned all 31.04M parameters from 624 ultrasound images.
   - Future work must evaluate a pretrained CNN (e.g. ResNet-50 U-Net) to definitively decouple the architecture effect from the pretraining effect.
3. **MoE Variance Buffer**:
   - Swin-MoE compressed run-to-run volatility ($s = 2.15\%$ vs $4.74\%$ Vanilla and $3.68\%$ U-Net) and prevented single-seed collapse (worst seed $54.18\%$ vs $42.12\%$ U-Net). We document this as an empirical observation in this 10-seed cohort, pending verification on larger datasets.

---

## 4. Phase 11 Attention U-Net Experiment Protocol

The experiment specification for Phase 11 has been authored and verified:
- **Architecture**: Attention U-Net (Oktay et al., 2018).
- **Exact Parameters**: **$31,389,295$ trainable parameters** (4 Attention Gates add $351,532$ parameters, a $+1.13\%$ increase over Standard U-Net).
- **Experimental Controls**: Identical 10 seeds, identical 80/10/10 BUSI splits, identical $224 \times 224$ resolution, identical AdamW and Focal Tversky Loss.
- **Hypothesis**: Can skip-connection spatial attention gates resolve U-Net's volume under-prediction pathology and lift recall from $58.57\%$ toward $\ge 70.0\%$?

---

## 5. Summary of Saved Artifacts

All files have been saved under `research_audit/phase10_unet_final/`:
- Reports:
  - [PHASE10_FINAL_REPRODUCIBILITY_AUDIT.md](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/phase10_unet_final/PHASE10_FINAL_REPRODUCIBILITY_AUDIT.md)
  - [PHASE10_STATISTICAL_VALIDATION.md](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/phase10_unet_final/PHASE10_STATISTICAL_VALIDATION.md)
  - [PHASE10_PRECISION_RECALL_ANALYSIS.md](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/phase10_unet_final/PHASE10_PRECISION_RECALL_ANALYSIS.md)
  - [PHASE9B_PHASE10_COMPARATIVE_REPORT.md](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/phase10_unet_final/PHASE9B_PHASE10_COMPARATIVE_REPORT.md)
  - [PHASE11_ATTENTION_UNET_EXPERIMENT_PLAN.md](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/phase10_unet_final/PHASE11_ATTENTION_UNET_EXPERIMENT_PLAN.md)
  - [PHASE10_FINAL_SUPERVISOR_REPORT.md](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/phase10_unet_final/PHASE10_FINAL_SUPERVISOR_REPORT.md)
- Datasets & Metrics:
  - [PHASE10_PER_SEED_RECALCULATED.csv](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/phase10_unet_final/PHASE10_PER_SEED_RECALCULATED.csv)
  - [PHASE10_STATISTICAL_METRICS_WITH_HOLM.csv](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/phase10_unet_final/PHASE10_STATISTICAL_METRICS_WITH_HOLM.csv)
  - [PHASE9B_PHASE10_CROSS_BENCHMARK.csv](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/phase10_unet_final/PHASE9B_PHASE10_CROSS_BENCHMARK.csv)
- Publication Figures:
  - [volume_ratio_underprediction.png](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/phase10_unet_final/plots/volume_ratio_underprediction.png)
  - [precision_recall_wedge.png](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/phase10_unet_final/plots/precision_recall_wedge.png)
  - [cross_benchmark_iou_distribution.png](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/phase10_unet_final/plots/cross_benchmark_iou_distribution.png)

---

## 6. Recommended Next Action

With the Phase 10 audit formally concluded, the recommended next step is:
**Proceed to implement the Phase 11 Attention U-Net architecture (`models/attention_unet.py`), unit test suite, and Kaggle deployment notebook.**
*(Note: As instructed, no training will take place until the implementation is approved).*
