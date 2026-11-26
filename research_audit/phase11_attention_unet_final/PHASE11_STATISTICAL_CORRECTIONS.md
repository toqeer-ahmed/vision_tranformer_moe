# Phase 11 Statistical Corrections: Hypothesis Testing, Multiplicity Families, and Volume Ratio Formulations

**Document Identifier**: `research_audit/phase11_attention_unet_final/PHASE11_STATISTICAL_CORRECTIONS.md`  
**Workspace**: `D:\Study Material\Research\CV\vision_transformer_research`  
**Auditor**: Senior Computer Vision Research Engineer & Statistical Auditor  
**Date**: 2026-11-26  
**Status**: Formal Statistical Revision & Multiplicity Audit  

---

## 1. Correction of Hypothesis-Testing Interpretation

In initial reporting, the phrase *"The Attention Gate hypothesis is decisively falsified"* was used to describe the $-7.24\%$ decline in Foreground Recall and $-5.44\%$ decline in Foreground IoU. 

From the perspective of formal statistical inference, this language is imprecise and overstates what a non-significant paired test establishes.

### Formal Distinctions:
1. **Evidence of Improvement ($H_1: \mu_{\Delta} > 0$)**:
   - The original research hypothesis postulated that Attention Gates would increase Foreground Recall under cross-dataset domain shift ($\mu_{\Delta_{\text{Recall}}} > 0$).
   - The sample mean difference was negative ($\bar{\Delta} = -7.24\%$), and the upper bound of the 95% confidence interval is $+2.93\%$.
   - **Conclusion**: The empirical data **failed to establish evidence of improvement**. The hypothesis of superior recall is not supported by the data.
2. **Evidence of Deterioration ($H_1: \mu_{\Delta} < 0$)**:
   - For Foreground IoU: $t(9) = -1.706, p = 0.1225$, Holm $p = 0.7168$, Wilcoxon $W = 15.0, p = 0.1602$.
   - For Foreground Recall: $t(9) = -1.614, p = 0.1416$, Holm $p = 0.7168$, Wilcoxon $W = 13.0, p = 0.1055$.
   - Because all two-tailed $p$-values exceed $\alpha = 0.05$ and both 95% confidence intervals include zero (IoU CI: $[-12.66\%, +1.78\%]$; Recall CI: $[-17.41\%, +2.93\%]$), the test **fails to reject the null hypothesis of no difference**.
   - **Conclusion**: We **cannot claim statistically significant deterioration**. Describing the result as "decisively falsified" or "proven performance collapse" conflates a negative sample point estimate with statistical rejection.
3. **Statistical Power Limitation**:
   - Across $N = 10$ matched seeds, the sample standard deviation of paired differences is substantial ($s_{\Delta_{\text{IoU}}} = 10.09\%$, $s_{\Delta_{\text{Recall}}} = 14.20\%$).
   - Given this variance, the study had approximately $35\%$ statistical power to detect a true difference of $5\%$.
   - With wide confidence intervals spanning from $-12.66\%$ to $+1.78\%$, the data cannot rule out either a small positive effect ($< +1.8\%$) or a moderate-to-large negative effect (down to $-12.7\%$).
   - **Correct Scientific Summary**: Attention Gates did not provide a measurable benefit over Standard U-Net, but statistical power in this 10-seed cohort is insufficient to definitively assert a population-level degradation.

---

## 2. Multiplicity Correction: Defining Holm-Bonferroni Families

To ensure rigorous Family-Wise Error Rate (FWER) control, we audit the multiplicity structure across comparisons and metrics.

### Family Definitions:
1. **Primary Segmentation Quality Family (7 Metrics)**:
   - Metrics: Foreground IoU, Foreground Dice, Macro mIoU, Macro mDice, Foreground Precision, Foreground Recall, Pixel Accuracy.
   - These 7 metrics evaluate pixel-level spatial overlap, class balance, and accuracy.
2. **Diagnostic Calibration Metric (Volume Ratio)**:
   - Volume Ratio evaluates macroscopic tumor volume calibration. Because it is a global diagnostic ratio rather than a bounded spatial overlap score, it is conventionally treated as a secondary diagnostic endpoint.
3. **Omnibus Family (8 Metrics)**:
   - All 8 metrics evaluated together in a single global family.

### Audited Multiplicity Table for Attention U-Net vs Standard U-Net:

| Metric | Raw Paired $t$ | Raw $p$-value | Scheme A: Primary Family (7 Metrics) Holm $p$ | Scheme B: Omnibus Family (8 Metrics) Holm $p$ | Wilcoxon $W$ | Wilcoxon Raw $p$ | Wilcoxon Holm $p$ (Scheme A) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pixel Accuracy** | $-1.884$ | $0.0913$ | **$0.6392$** | **$0.7305$** | $14.0$ | $0.1309$ | $0.7852$ |
| **Macro mIoU** | $-1.722$ | $0.1195$ | **$0.7168$** | **$0.8363$** | $14.0$ | $0.1309$ | $0.7852$ |
| **Foreground IoU** | $-1.706$ | $0.1225$ | **$0.7168$** | **$0.8363$** | $15.0$ | $0.1602$ | $0.8008$ |
| **Macro mDice** | $-1.691$ | $0.1259$ | **$0.7168$** | **$0.8363$** | $15.0$ | $0.1602$ | $0.8008$ |
| **Foreground Dice** | $-1.681$ | $0.1277$ | **$0.7168$** | **$0.8363$** | $15.0$ | $0.1602$ | $0.8008$ |
| **Foreground Recall** | $-1.614$ | $0.1416$ | **$0.7168$** | **$0.8363$** | $13.0$ | $0.1055$ | $0.7383$ |
| **Volume Ratio** | $-1.452$ | $0.1808$ | *(Secondary)* | **$0.8363$** | $16.0$ | $0.1934$ | *(Secondary)* |
| **Foreground Precision** | $+0.618$ | $0.5518$ | **$0.7168$** | **$0.8363$** | $23.0$ | $0.5566$ | $0.8008$ |

*Multiplicity Conclusion*: Under both Scheme A and Scheme B, all adjusted $p$-values for Attention U-Net vs Standard U-Net remain $> 0.63$. The difference between the two architectures is statistically indistinguishable from zero.

---

## 3. Mathematical Verification of Volume Ratio Formulations

In medical image segmentation literature, "volume ratio" can be computed under multiple distinct mathematical formulations. Here we audit and clarify each definition:

### Definition 1: Dataset-Level Aggregate Ratio per Seed (Used in Master Benchmark)
For a given seed $s$ evaluated across all $N = 1,063$ images in BUS-BRA:
$$\text{VolRatio}_{s} = \frac{\sum_{i=1}^N \text{PredVol}_{s, i}}{\sum_{i=1}^N \text{GTVol}_{s, i}} = \frac{\text{TP}_{s} + \text{FP}_{s}}{\text{TP}_{s} + \text{FN}_{s}}$$

In our benchmark:
- Ground-Truth Foreground Volume is constant across all seeds: $\sum_{i=1}^N \text{GTVol}_{i} = 8,146,649$ pixels ($8.15\text{ MP}$).
- For Seed 42: $\text{PredVol}_{42} = 2,733,742 \implies \text{VolRatio}_{42} = 0.3356$.
- For Seed 2718: $\text{PredVol}_{2718} = 5,906,446 \implies \text{VolRatio}_{2718} = 0.7250$.

### Definition 2: Mean of Per-Seed Aggregate Volume Ratios (Used in Master Benchmark)
The summary volume ratio reported in the master table is the sample mean of $\text{VolRatio}_s$ across the 10 seeds:
$$\overline{\text{VolRatio}} = \frac{1}{S} \sum_{s=1}^S \text{VolRatio}_{s} = \frac{1}{10} \sum_{s=1}^{10} \frac{\text{PredVol}_s}{8,146,649}$$

Because the denominator is strictly constant, this is mathematically identical to the global aggregate ratio:
$$\overline{\text{VolRatio}} = \frac{\sum_{s=1}^{10} \text{PredVol}_s}{10 \times 8,146,649} = \frac{51,947,979}{81,466,490} = 0.6377 \approx 0.638\times$$

### Definition 3: Mean of Per-Image Volume Ratios (Why It Is Avoided)
$$\text{VolRatio}_{\text{per-image}} = \frac{1}{N} \sum_{i=1}^N \frac{\text{PredVol}_i}{\text{GTVol}_i}$$
- **Instability**: If an image contains no ground truth lesion ($\text{GTVol}_i = 0$), the ratio is undefined ($\infty$).
- **Severe Outlier Distortion**: For very small lesions (e.g., $\text{GTVol}_i = 200$ pixels), a false-positive cluster of $1,000$ pixels yields a ratio of $5.0\times$, heavily skewing the arithmetic mean.
- **Auditor Decision**: Definition 1 and Definition 2 are the standard, robust formulations in medical segmentation benchmarks and are maintained throughout this audit.
