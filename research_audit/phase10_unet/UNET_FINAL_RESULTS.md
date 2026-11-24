# Standard U-Net Baseline: Final Empirical Audit & Cross-Architecture Benchmark

**Experiment Identifier**: `phase10_unet`  
**Auditor & Lead Architect**: Senior Computer Vision Research Engineer & Scientific Auditor  
**Repository**: `D:\Study Material\Research\CV\vision_transformer_research`  
**Date**: 2026-11-25  
**Status**: Completed, 100% Empirically Audited and Statistically Verified  

---

## 1. Executive Summary & Core Scientific Findings

The canonical 2D U-Net baseline (Ronneberger et al., 2015; $31,037,763$ trainable parameters) has completed full empirical training and zero-shot generalization testing across the **10 matched random seeds** ($\mathcal{S} = \{42, 123, 314, 777, 999, 1001, 2026, 2027, 2718, 9001\}$).

### Key Empirical Findings:

1. **Pretrained Vision Transformers Significantly Outperform From-Scratch CNNs**:
   - Standard U-Net achieves an external zero-shot BUS-BRA Foreground IoU of **$50.79\% \pm 3.68\%$** and Foreground Dice of **$67.29\% \pm 3.33\%$**.
   - Compared to **Vanilla Swin-Tiny** ($57.50\% \pm 4.74\%$), U-Net exhibits a **statistically significant deficit of $-6.71\%$ IoU** (paired $t = -3.467$, $p = 0.0071$, Wilcoxon $W = 4.0, p = 0.0137$, Cohen's $d_z = -1.10$).
   - Compared to **Standard-MLP Swin-MoE** ($57.85\% \pm 2.15\%$), U-Net is **strictly inferior across all 10 seeds** (paired $\Delta = -7.06\%$ IoU, $t = -6.342$, $p = 0.0001$, Wilcoxon $W = 0.0, p = 0.0020$, Cohen's $d_z = -2.01$).

2. **The Precision-Recall Pathology in Cross-Dataset Generalization**:
   - Standard U-Net exhibits remarkably **high precision ($80.19\% \pm 4.73\%$)**, outperforming Vanilla Swin ($70.44\%$, $+9.75\%$, $p = 0.0033$) and Swin-MoE ($69.92\%$, $+10.27\%$, $p = 0.0017$).
   - However, U-Net suffers from **catastrophically degraded recall ($58.57\% \pm 6.49\%$)**, trailing Vanilla Swin by **$-17.94\%$** ($p = 0.0002$) and Swin-MoE by **$-18.78\%$** ($p < 0.00005$).
   - *Clinical interpretation*: Convolutional inductive bias without pretraining produces an overly conservative model that segments only high-confidence tumor cores and systematically misses lesion boundaries and faint acoustic shadows.

3. **In-Domain vs. Out-of-Domain Generalization Gap**:
   - On the held-in BUSI test split ($N=78$), U-Net achieves **$56.06\% \pm 3.88\%$** Foreground IoU and **$71.77\% \pm 3.22\%$** Foreground Dice.
   - On the external BUS-BRA dataset ($N=1,063$), U-Net drops to **$50.79\%$ IoU** (a domain-shift degradation penalty of $-5.27\%$ IoU and $-9.84\%$ Recall).

---

## 2. Comprehensive 10-Seed Empirical Summary

The table below provides the full cross-architecture benchmark across all 10 matched seeds on the external BUS-BRA dataset ($N = 1,063$):

| Architecture | Model Family | Parameters | Pretraining | Fg IoU (%) | Fg Dice (%) | Macro mIoU (%) | Macro mDice (%) | Fg Precision (%) | Fg Recall (%) | Pixel Acc (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vanilla Swin-Tiny** | Hierarchical ViT | $29.29\text{M}$ | ImageNet-1K | $57.50 \pm 4.74$ | $72.91 \pm 3.98$ | $76.10 \pm 2.82$ | $85.09 \pm 2.23$ | $70.44 \pm 6.61$ | $76.51 \pm 6.79$ | $95.06 \pm 0.98$ |
| **Standard-MLP MoE** | ViT + MoE (Top-2) | $81.15\text{M}$ | Warm-Init | $57.85 \pm 2.15$ | $73.28 \pm 1.73$ | $76.30 \pm 1.31$ | $85.29 \pm 0.99$ | $69.92 \pm 4.33$ | $77.35 \pm 3.73$ | $95.11 \pm 0.50$ |
| **Standard U-Net** | Canonical 2D CNN | $31.04\text{M}$ | From Scratch | **$50.79 \pm 3.68$** | **$67.29 \pm 3.33$** | **$72.82 \pm 1.93$** | **$82.32 \pm 1.72$** | **$80.19 \pm 4.73$** | **$58.57 \pm 6.49$** | **$95.11 \pm 0.28$** |

*Note: All summary statistics are reported as Mean $\pm$ Sample Standard Deviation ($N=10$).*

---

## 3. Per-Seed Paired Performance Breakdown

The table below details the performance of all three models across every matched seed:

| Seed | Vanilla Swin IoU (%) | Swin-MoE IoU (%) | Standard U-Net IoU (%) | $\Delta(\text{U-Net} - \text{Vanilla})$ | $\Delta(\text{U-Net} - \text{MoE})$ | U-Net Precision (%) | U-Net Recall (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | $56.78\%$ | $59.24\%$ | $55.33\%$ | $-1.45\%$ | $-3.91\%$ | $75.54\%$ | $67.41\%$ |
| **123** | $47.33\%$ | $54.18\%$ | $49.38\%$ | $+2.05\%$ | $-4.80\%$ | $82.02\%$ | $55.37\%$ |
| **314** | $60.61\%$ | $56.12\%$ | $54.13\%$ | $-6.48\%$ | $-1.99\%$ | $78.01\%$ | $63.88\%$ |
| **777** | $60.16\%$ | $57.93\%$ | $49.47\%$ | $-10.69\%$ | $-8.46\%$ | $81.57\%$ | $55.69\%$ |
| **999** | $60.98\%$ | $56.57\%$ | $42.12\%$ | $-18.86\%$ | $-14.45\%$ | $86.23\%$ | $45.15\%$ |
| **1001** | $59.08\%$ | $59.52\%$ | $51.03\%$ | $-8.05\%$ | $-8.48\%$ | $70.29\%$ | $65.07\%$ |
| **2026** | $50.69\%$ | $59.25\%$ | $51.03\%$ | $+0.34\%$ | $-8.22\%$ | $82.63\%$ | $57.17\%$ |
| **2027** | $58.24\%$ | $59.88\%$ | $50.61\%$ | $-7.63\%$ | $-9.27\%$ | $80.56\%$ | $57.65\%$ |
| **2718** | $59.78\%$ | $60.42\%$ | $54.11\%$ | $-5.67\%$ | $-6.31\%$ | $79.49\%$ | $62.90\%$ |
| **9001** | $61.33\%$ | $55.41\%$ | $50.67\%$ | $-10.66\%$ | $-4.74\%$ | $85.55\%$ | $55.41\%$ |
| **Mean** | **$57.50\%$** | **$57.85\%$** | **$50.79\%$** | **$-6.71\%$** | **$-7.06\%$** | **$80.19\%$** | **$58.57\%$** |

---

## 4. Formal Hypothesis Testing & Statistical Significance

### A. Standard U-Net vs. Vanilla Swin-Tiny (Parameter Parity: 31.04M vs 29.29M)
- **Null Hypothesis $H_0$**: $\mu_{\text{UNet}} = \mu_{\text{Vanilla}}$
- **Mean Paired Difference**: $\Delta = -6.71\%$ IoU ($95\%$ CI: $[-11.09\%, -2.33\%]$)
- **Paired $t$-test**: $t(9) = -3.467$, $p = 0.0071$ (**Statistically significant, $p < 0.01$**)
- **Wilcoxon Signed-Rank Test**: $W = 4.0$, $p = 0.0137$ (**Statistically significant, $p < 0.05$**)
- **Effect Size**: Cohen's $d_z = -1.10$ (Large negative effect)

### B. Standard U-Net vs. Standard-MLP Swin-MoE
- **Null Hypothesis $H_0$**: $\mu_{\text{UNet}} = \mu_{\text{MoE}}$
- **Mean Paired Difference**: $\Delta = -7.06\%$ IoU ($95\%$ CI: $[-9.58\%, -4.54\%]$)
- **Paired $t$-test**: $t(9) = -6.342$, $p = 0.0001$ (**Extremely significant, $p < 0.001$**)
- **Wilcoxon Signed-Rank Test**: $W = 0.0$, $p = 0.0020$ (**U-Net lost in 10/10 seeds**)
- **Effect Size**: Cohen's $d_z = -2.01$ (Massive negative effect)

---

## 5. BUSI In-Domain Test vs. BUS-BRA Zero-Shot Transfer

To isolate domain shift from general model learning capacity, Standard U-Net was evaluated on both the BUSI in-domain test split ($N=78$) and external BUS-BRA ($N=1,063$):

| Evaluation Set | Metric | Mean | Std | Min (Worst Seed) | Max (Best Seed) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **BUSI In-Domain ($N=78$)** | Fg IoU | $56.06\%$ | $3.88\%$ | $48.70\%$ (Seed 2027) | $62.04\%$ (Seed 314) |
| | Fg Dice | $71.77\%$ | $3.22\%$ | $65.50\%$ (Seed 2027) | $76.57\%$ (Seed 314) |
| | Fg Precision | $76.03\%$ | $4.67\%$ | $67.84\%$ (Seed 1001) | $83.66\%$ (Seed 314) |
| | Fg Recall | $68.41\%$ | $6.12\%$ | $60.35\%$ (Seed 2027) | $79.73\%$ (Seed 1001) |
| | Macro mIoU | $83.85\%$ | $1.45\%$ | $81.03\%$ (Seed 2027) | $86.17\%$ (Seed 314) |
| **BUS-BRA Zero-Shot ($N=1,063$)** | Fg IoU | $50.79\%$ | $3.68\%$ | $42.12\%$ (Seed 999) | $55.33\%$ (Seed 42) |
| | Fg Dice | $67.29\%$ | $3.33\%$ | $59.27\%$ (Seed 999) | $71.24\%$ (Seed 42) |
| | Fg Precision | $80.19\%$ | $4.73\%$ | $70.29\%$ (Seed 1001) | $86.23\%$ (Seed 999) |
| | Fg Recall | $58.57\%$ | $6.49\%$ | $45.15\%$ (Seed 999) | $67.41\%$ (Seed 42) |
| | Macro mIoU | $72.82\%$ | $1.93\%$ | $68.26\%$ (Seed 999) | $75.16\%$ (Seed 42) |
| **Domain Shift Impact ($\Delta$)** | **Fg IoU** | **$-5.27\%$** | — | — | — |
| | **Fg Recall** | **$-9.84\%$** | — | — | — |
| | **Fg Precision** | **$+4.16\%$** | — | — | — |

---

## 6. Research Artifacts Generated

All data artifacts have been recalculated and archived in the repository:
- Per-seed Zero-Shot Results: [UNET_PER_SEED_RESULTS.csv](file:///D:/Study%20Material/Research/CV/vision_transformer_research/results/phase10_unet/UNET_PER_SEED_RESULTS.csv)
- Per-seed In-Domain Test Results: [UNET_INDOMAIN_TEST_RESULTS.csv](file:///D:/Study%20Material/Research/CV/vision_transformer_research/results/phase10_unet/UNET_INDOMAIN_TEST_RESULTS.csv)
- Summary Results: [UNET_SUMMARY_RESULTS.csv](file:///D:/Study%20Material/Research/CV/vision_transformer_research/results/phase10_unet/UNET_SUMMARY_RESULTS.csv)
- Paired Comparative Dataset: [RECALCULATED_PHASE10_PER_SEED_PAIRED.csv](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/phase10_unet/RECALCULATED_PHASE10_PER_SEED_PAIRED.csv)
- Formal Statistical Metrics: [RECALCULATED_PHASE10_STATISTICAL_METRICS.csv](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/phase10_unet/RECALCULATED_PHASE10_STATISTICAL_METRICS.csv)
- Raw Confusion Matrices: [unet_reproducibility_metrics.json](file:///D:/Study%20Material/Research/CV/vision_transformer_research/results/phase10_unet/unet_reproducibility_metrics.json)
- Publication Figures:
  - Benchmark Distribution: [cross_architecture_comparison_iou.png](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/phase10_unet/plots/cross_architecture_comparison_iou.png)
  - Precision-Recall Trade-off: [precision_recall_tradeoff.png](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/phase10_unet/plots/precision_recall_tradeoff.png)
  - Paired Delta per Seed: [paired_difference_unet_vs_swin.png](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/phase10_unet/plots/paired_difference_unet_vs_swin.png)
