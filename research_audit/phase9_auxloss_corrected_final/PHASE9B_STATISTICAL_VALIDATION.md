# Phase 9B Statistical Validation & Hypothesis Testing

**Experiment Identifier**: `phase9_auxloss_corrected`  
**Auditor**: Senior Computer Vision Research Engineer & Scientific Auditor  
**Repository**: `D:\Study Material\Research\CV\vision_transformer_research`  
**Sample Size**: $N = 10$ Matched Random Seeds ($df = 9$)  
**Benchmark Target**: Zero-Shot Cross-Dataset Transfer on BUS-BRA ($N = 1,063$)  

---

## 1. Methodology: Independent Metric Recalculation

All metrics were recalculated directly from the raw confusion matrix tensors (`class_tp`, `class_fp`, `class_fn`) and test predictions in `reproducibility_metrics.json`. No numbers were transcribed from previous summary documents without independent mathematical verification.

### Mathematical Definitions:
- **Foreground IoU**: $\frac{\text{TP}_1}{\text{TP}_1 + \text{FP}_1 + \text{FN}_1}$
- **Foreground Dice**: $\frac{2 \cdot \text{TP}_1}{2 \cdot \text{TP}_1 + \text{FP}_1 + \text{FN}_1}$
- **Foreground Precision**: $\frac{\text{TP}_1}{\text{TP}_1 + \text{FP}_1}$
- **Foreground Recall**: $\frac{\text{TP}_1}{\text{TP}_1 + \text{FN}_1}$
- **Macro mIoU**: $\frac{1}{2} (\text{IoU}_0 + \text{IoU}_1)$
- **Macro mDice**: $\frac{1}{2} (\text{Dice}_0 + \text{Dice}_1)$
- **Pixel Accuracy**: $\frac{\sum \text{TP}_c}{\text{Total Pixels}}$

---

## 2. Recalculated Statistical Summary Across 10 Matched Seeds

| Metric | Vanilla Swin-Tiny (29.29M) | Standard-MLP Swin-MoE (81.15M) | Mean Paired Difference ($\bar{D}$) | 95% Confidence Interval | Paired $t$-test ($t$, $p$-value) | Wilcoxon Signed-Rank ($W$, $p$-value) | Effect Size (Cohen's $d_z$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Foreground IoU** | $57.50\% \pm 4.74\%$ | $57.85\% \pm 2.15\%$ | **$+0.35\%$** | $[-3.09\%, +3.79\%]$ | $t = 0.2319, \; p = 0.8218$ | $W = 25.0, \; p = 0.8457$ | $0.0733$ (Negligible) |
| **Foreground Dice** | $72.91\% \pm 3.98\%$ | $73.28\% \pm 1.73\%$ | **$+0.37\%$** | $[-2.48\%, +3.22\%]$ | $t = 0.2942, \; p = 0.7753$ | $W = 25.0, \; p = 0.8457$ | $0.0930$ (Negligible) |
| **Macro mIoU** | $76.10\% \pm 2.82\%$ | $76.30\% \pm 1.31\%$ | **$+0.20\%$** | $[-1.87\%, +2.26\%]$ | $t = 0.2164, \; p = 0.8335$ | $W = 25.0, \; p = 0.8457$ | $0.0684$ (Negligible) |
| **Macro mDice** | $85.09\% \pm 2.23\%$ | $85.29\% \pm 0.99\%$ | **$+0.20\%$** | $[-1.41\%, +1.80\%]$ | $t = 0.2779, \; p = 0.7874$ | $W = 25.0, \; p = 0.8457$ | $0.0879$ (Negligible) |
| **Foreground Precision** | $70.44\% \pm 6.61\%$ | $69.92\% \pm 4.33\%$ | **$-0.52\%$** | $[-5.32\%, +4.27\%]$ | $t = -0.2459, \; p = 0.8113$ | $W = 24.0, \; p = 0.7695$ | $-0.0778$ (Negligible) |
| **Foreground Recall** | $76.51\% \pm 6.79\%$ | $77.35\% \pm 3.73\%$ | **$+0.83\%$** | $[-3.57\%, +5.24\%]$ | $t = 0.4283, \; p = 0.6785$ | $W = 25.0, \; p = 0.8457$ | $0.1354$ (Negligible) |
| **Pixel Accuracy** | $95.06\% \pm 0.98\%$ | $95.11\% \pm 0.50\%$ | **$+0.04\%$** | $[-0.68\%, +0.77\%]$ | $t = 0.1364, \; p = 0.8945$ | $W = 25.0, \; p = 0.8457$ | $0.0431$ (Negligible) |

---

## 3. Complete Per-Seed Matched Pairs Table ($N=10$)

The table below reports the verified per-seed scores and paired deltas ($D_s = \text{MoE}_s - \text{Vanilla}_s$):

| Seed | Vanilla FG IoU | MoE FG IoU | $\Delta$ FG IoU | Vanilla FG Dice | MoE FG Dice | $\Delta$ FG Dice | Vanilla mIoU | MoE mIoU | Higher Performer |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | $56.78\%$ | $59.24\%$ | **$+2.46\%$** | $72.44\%$ | $74.40\%$ | $+1.97\%$ | $75.58\%$ | $77.16\%$ | **MoE** |
| **123** | $47.33\%$ | $54.18\%$ | **$+6.85\%$** | $64.25\%$ | $70.28\%$ | $+6.03\%$ | $69.59\%$ | $74.19\%$ | **MoE** |
| **314** | $60.61\%$ | $56.12\%$ | **$-4.49\%$** | $75.47\%$ | $71.89\%$ | $-3.58\%$ | $77.96\%$ | $74.97\%$ | **Vanilla** |
| **777** | $60.16\%$ | $57.93\%$ | **$-2.23\%$** | $75.12\%$ | $73.36\%$ | $-1.76\%$ | $77.51\%$ | $76.41\%$ | **Vanilla** |
| **999** | $60.98\%$ | $56.57\%$ | **$-4.41\%$** | $75.76\%$ | $72.26\%$ | $-3.50\%$ | $78.16\%$ | $75.49\%$ | **Vanilla** |
| **1001** | $59.08\%$ | $59.52\%$ | **$+0.43\%$** | $74.28\%$ | $74.62\%$ | $+0.34\%$ | $77.04\%$ | $77.39\%$ | **MoE** |
| **2026** | $50.69\%$ | $59.25\%$ | **$+8.56\%$** | $67.28\%$ | $74.41\%$ | $+7.14\%$ | $72.73\%$ | $77.35\%$ | **MoE** |
| **2027** | $58.24\%$ | $59.88\%$ | **$+1.64\%$** | $73.61\%$ | $74.91\%$ | $+1.30\%$ | $76.74\%$ | $77.53\%$ | **MoE** |
| **2718** | $59.78\%$ | $60.42\%$ | **$+0.64\%$** | $74.83\%$ | $75.33\%$ | $+0.50\%$ | $77.41\%$ | $77.70\%$ | **MoE** |
| **9001** | $61.33\%$ | $55.41\%$ | **$-5.92\%$** | $76.03\%$ | $71.30\%$ | $-4.73\%$ | $78.32\%$ | $74.84\%$ | **Vanilla** |

---

## 4. Rigorous Hypothesis Testing Decisions

### Null Hypothesis 1: Mean Generalization Equivalence
- **$H_0$**: $\mu_{\text{MoE}} - \mu_{\text{Vanilla}} = 0$
- **Findings**:
  - Across all seven evaluated metrics, the two-tailed paired $t$-test yields $p$-values ranging from $0.6785$ to $0.8945$.
  - The non-parametric Wilcoxon signed-rank test yields $p = 0.8457$ across all primary overlap metrics ($W = 25.0$).
  - For Foreground IoU, $\bar{D} = +0.35\%$, with a $95\%$ Confidence Interval of $[-3.09\%, +3.79\%]$. Because the interval cleanly brackets zero, neither architecture can be declared superior in mean performance.
  - The standardized effect size (Cohen's $d_z = 0.0733$) falls well below the threshold for even a "small" effect ($|d_z| < 0.20$).
- **Formal Scientific Decision**: **Fail to reject $H_0$**. Standard-MLP Swin-MoE does not provide a statistically significant improvement in mean zero-shot cross-dataset segmentation performance over Vanilla Swin-Tiny.
