# Phase 10: Statistical Validation & Multiple-Comparison Corrected Analysis

**Experiment Identifier**: `phase10_unet`  
**Auditor**: Senior Computer Vision Research Engineer & Scientific Auditor  
**Repository**: `D:\Study Material\Research\CV\vision_transformer_research`  
**Date**: 2026-11-25  

---

## 1. Statistical Methodology & Assumptions

To rigorously evaluate whether Standard U-Net differs significantly from Phase 9B baselines (**Vanilla Swin-Tiny** and **Standard-MLP Swin-MoE**), we employ paired statistical tests across the 10 matched random seeds:

1. **Paired Differences ($\Delta_i = y_{i, \text{UNet}} - y_{i, \text{Baseline}}$)**:
   - Measures intra-seed performance changes controlling for split variation.
2. **Normality Testing (Shapiro-Wilk)**:
   - Evaluates whether paired differences $\Delta$ meet the normality assumption required by Student's paired $t$-test.
3. **Parametric & Non-Parametric Hypothesis Testing**:
   - Two-tailed paired $t$-test ($t$-statistic, degrees of freedom $= 9$).
   - Two-tailed Wilcoxon signed-rank test ($W$-statistic) for distribution-free verification.
4. **Effect Size (Cohen's $d_z$)**:
   $$d_z = \frac{\bar{\Delta}}{s_{\Delta}}$$
5. **Multiple-Comparison Correction (Holm-Bonferroni Step-Down)**:
   - Controls the Family-Wise Error Rate (FWER) at $\alpha = 0.05$ across all 14 planned pairwise tests:
     $$p_{\text{adj}(i)} = \min\left(1.0, \max_{j \le i} \left[(m - j + 1) \cdot p_{(j)}\right]\right)$$

---

## 2. Normality Assessment of Paired Differences

The Shapiro-Wilk test was conducted on all paired difference series ($N=10$):

| Metric | $\Delta(\text{U-Net} - \text{Vanilla})$ Shapiro $W$ | $\Delta(\text{U-Net} - \text{Vanilla})$ Shapiro $p$ | $\Delta(\text{U-Net} - \text{MoE})$ Shapiro $W$ | $\Delta(\text{U-Net} - \text{MoE})$ Shapiro $p$ | Normality Status ($\alpha = 0.05$) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Foreground IoU** | $0.9549$ | $0.7266$ | $0.9405$ | $0.5584$ | **Normal** (Assumption satisfied) |
| **Foreground Dice** | $0.9503$ | $0.6721$ | $0.9197$ | $0.3546$ | **Normal** (Assumption satisfied) |
| **Macro mIoU** | $0.9681$ | $0.8726$ | $0.9535$ | $0.7104$ | **Normal** (Assumption satisfied) |
| **Macro mDice** | $0.9616$ | $0.8042$ | $0.9372$ | $0.5219$ | **Normal** (Assumption satisfied) |
| **Foreground Precision** | $0.8870$ | $0.1567$ | $0.9486$ | $0.6523$ | **Normal** (Assumption satisfied) |
| **Foreground Recall** | $0.9589$ | $0.7734$ | $0.8797$ | $0.1296$ | **Normal** (Assumption satisfied) |
| **Pixel Accuracy** | $0.7832$ | **$0.0091$** | $0.9631$ | $0.8207$ | **Non-Normal** for Vanilla difference |

*Statistical Note*: Because paired differences for lesion segmentation metrics do not deviate significantly from normality ($p > 0.12$), the parametric paired $t$-test is valid. For Pixel Accuracy against Vanilla Swin, non-parametric Wilcoxon is reported as the primary test.

---

## 3. Comprehensive Pairwise Statistical Test Matrix

### A. Standard U-Net vs. Vanilla Swin-Tiny (Parameter Parity: 31.04M vs. 29.29M)

| Metric | Mean $\Delta$ | $95\%$ CI for $\Delta$ | Paired $t$ | Raw $p_t$ | **Holm $p_t$** | Wilcoxon $W$ | Raw $p_W$ | **Holm $p_W$** | Cohen's $d_z$ | Statistical Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Foreground IoU** | **$-6.71\%$** | $[-11.09\%, -2.33\%]$ | $-3.467$ | $0.0071$ | **$0.0425$** | $4.0$ | $0.0137$ | $0.0820$ | $-1.10$ | **Significantly Inferior ($p < 0.05$)** |
| **Foreground Dice** | **$-5.62\%$** | $[-9.41\%, -1.83\%]$ | $-3.352$ | $0.0085$ | **$0.0425$** | $4.0$ | $0.0137$ | $0.0820$ | $-1.06$ | **Significantly Inferior ($p < 0.05$)** |
| **Macro mIoU** | **$-3.29\%$** | $[-5.78\%, -0.79\%]$ | $-2.982$ | $0.0154$ | **$0.0549$** | $4.0$ | $0.0137$ | $0.0820$ | $-0.94$ | *Borderline trend ($p > 0.05$)* |
| **Macro mDice** | **$-2.77\%$** | $[-4.83\%, -0.72\%]$ | $-3.053$ | $0.0137$ | **$0.0549$** | $4.0$ | $0.0137$ | $0.0820$ | $-0.97$ | *Borderline trend ($p > 0.05$)* |
| **Foreground Precision** | **$+9.75\%$** | $[+4.18\%, +15.32\%]$ | $+3.958$ | $0.0033$ | **$0.0232$** | $1.0$ | $0.0039$ | **$0.0312$** | $+1.25$ | **Significantly Superior ($p < 0.05$)** |
| **Foreground Recall** | **$-17.94\%$** | $[-24.47\%, -11.42\%]$ | $-6.218$ | $0.0002$ | **$0.0019$** | $0.0$ | $0.0020$ | **$0.0273$** | $-1.97$ | **Significantly Inferior ($p < 0.01$)** |
| **Pixel Accuracy** | $+0.04\%$ | $[-0.69\%, +0.78\%]$ | $+0.137$ | $0.8939$ | $1.0000$ | $20.0$ | $0.4922$ | $0.9844$ | $+0.04$ | Not Significant |

### B. Standard U-Net vs. Standard-MLP Swin-MoE

| Metric | Mean $\Delta$ | $95\%$ CI for $\Delta$ | Paired $t$ | Raw $p_t$ | **Holm $p_t$** | Wilcoxon $W$ | Raw $p_W$ | **Holm $p_W$** | Cohen's $d_z$ | Statistical Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Foreground IoU** | **$-7.06\%$** | $[-9.58\%, -4.54\%]$ | $-6.342$ | $0.0001$ | **$0.0017$** | $0.0$ | $0.0020$ | **$0.0273$** | $-2.01$ | **Extremely Significant ($p < 0.01$)** |
| **Foreground Dice** | **$-5.99\%$** | $[-8.25\%, -3.73\%]$ | $-5.991$ | $0.0002$ | **$0.0023$** | $0.0$ | $0.0020$ | **$0.0273$** | $-1.89$ | **Extremely Significant ($p < 0.01$)** |
| **Macro mIoU** | **$-3.49\%$** | $[-4.89\%, -2.08\%]$ | $-5.598$ | $0.0003$ | **$0.0034$** | $0.0$ | $0.0020$ | **$0.0273$** | $-1.77$ | **Extremely Significant ($p < 0.01$)** |
| **Macro mDice** | **$-2.97\%$** | $[-4.17\%, -1.77\%]$ | $-5.593$ | $0.0003$ | **$0.0034$** | $0.0$ | $0.0020$ | **$0.0273$** | $-1.77$ | **Extremely Significant ($p < 0.01$)** |
| **Foreground Precision** | **$+10.27\%$** | $[+5.02\%, +15.52\%]$ | $+4.429$ | $0.0017$ | **$0.0132$** | $2.0$ | $0.0059$ | **$0.0410$** | $+1.40$ | **Significantly Superior ($p < 0.05$)** |
| **Foreground Recall** | **$-18.78\%$** | $[-23.57\%, -13.99\%]$ | $-8.869$ | $0.0000$ | **$0.0001$** | $0.0$ | $0.0020$ | **$0.0273$** | $-2.80$ | **Extremely Significant ($p < 0.001$)** |
| **Pixel Accuracy** | $+0.00\%$ | $[-0.43\%, +0.43\%]$ | $+0.005$ | $0.9958$ | $1.0000$ | $26.0$ | $0.9219$ | $0.9844$ | $+0.00$ | Not Significant |

---

## 4. Methodological Distinction: Data-Split Seeds vs. Model Initialization Seeds

A critical statistical nuance must be documented when comparing Phase 9B and Phase 10:

1. **Pretrained Vision Transformers (Swin-Tiny and Swin-MoE)**:
   - The encoder backbone is initialized from deterministic, pre-trained ImageNet-1K weights (`microsoft/swin-tiny-patch4-window7-224`).
   - Consequently, across all 10 seeds, the backbone parameter initialization is **identical**.
   - The random seed strictly controls: (1) head classification projection weights, (2) random sample shuffling in the data loader, and (3) pseudo-random Albumentations transforms.
2. **From-Scratch CNNs (Standard U-Net)**:
   - Contains zero pre-trained weights.
   - The random seed governs the **random initialization of the entire 31,037,763 parameter tensor space** via Kaiming/He normal distributions, in addition to controlling the data loader and augmentations.
3. **Statistical Implication**:
   - The "random seed" introduces substantially greater variance into from-scratch U-Net optimization than into pre-trained Vision Transformers.
   - Therefore, the comparison reflects both an **architectural contrast** (convolutions vs. windowed self-attention) and an **initialization manifold contrast** (convex pre-trained basin vs. raw non-convex initialization from scratch).

---

## 5. Statistical Caveats & Limitations

- **Sample Size ($N=10$)**: While 10 seeds exceed common medical imaging literature standards (which frequently report single-seed runs), power remains limited for detecting subtle variance ratio differences.
- **Family-Wise Error Rate Impact**: Under strict Holm-Bonferroni correction, U-Net's deficit in Foreground IoU ($p = 0.0425$) and Foreground Dice ($p = 0.0425$) retains statistical significance, but macro mIoU ($p = 0.0549$) and macro mDice ($p = 0.0549$) cross into the non-significant zone. We therefore do not claim statistical significance for macro metrics against Vanilla Swin under FWER control.
