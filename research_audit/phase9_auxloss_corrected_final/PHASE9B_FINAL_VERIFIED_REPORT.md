# Phase 9B Final Verified Scientific Report

**Project**: Vision Transformer Mixture-of-Experts for Breast Ultrasound Segmentation  
**Auditor**: Senior Computer Vision Research Engineer & Scientific Auditor  
**Repository**: `D:\Study Material\Research\CV\vision_transformer_research`  
**Date of Audit**: 2026-11-24  
**Status**: Concluded & Formally Verified  

---

## 1. Executive Summary

This report delivers the final verified scientific findings of the Phase 9B audit (`phase9_auxloss_corrected`). All conclusions are derived exclusively from raw experimental artifacts, recalculation from confusion matrices, and rigorous hypothesis testing across 10 matched random seeds (`42, 123, 314, 777, 999, 1001, 2026, 2027, 2718, 9001`).

---

## 2. Summary of Verified vs. Failed Claims

| Research Claim | Initial Reported Hypothesis | Recalculated Empirical Evidence | Statistical Decision | Final Scientific Status |
| :--- | :--- | :--- | :---: | :---: |
| **Mean Generalization Advantage** | MoE significantly outperforms Vanilla Swin by $+2\%$ to $+6\%$ | $\Delta = +0.35\%$, $95\%$ CI: $[-3.09\%, +3.79\%]$ | Paired $t$: $p = 0.8218$<br>Wilcoxon: $p = 0.8457$ | **FAILED VERIFICATION**<br>(No significant mean difference) |
| **Variance Reduction** | MoE provides a statistically significant $79.5\%$ reduction in variance | Classical $F = 4.87, p_{\text{uncorr}} = 0.0273$<br>Brown–Forsythe: $p = 0.3552$ | Bonferroni: $p = 0.1913$<br>Holm: $p = 0.1367$ | **FAILED VERIFICATION**<br>(Does not survive correction; normality violated) |
| **Guaranteed Performance Floor** | MoE establishes a guaranteed reliability floor of $54.18\%$ IoU | Vanilla min: $47.33\%$ (Seed 123)<br>MoE min: $54.18\%$ (Seed 123) | Extreme Value Theory: Sample min $\neq$ Support Infimum | **FAILED VERIFICATION**<br>(Sample minimum, not guaranteed floor) |
| **Causal Isolation of Routing** | Conditional routing acts as a structural regularizer | MoE: $81.15\text{M}$ params<br>Vanilla: $29.29\text{M}$ params | Capacity ratio: $2.77\times$<br>Confounded by parameter count | **FAILED VERIFICATION**<br>(Severely confounded by model capacity) |
| **Pipeline & Auxiliary Loss Repair** | Auxiliary loss is active and backpropagates through routers | All 12 MoE layers register $\mathcal{L}_{\text{aux}} \approx 0.24$<br>$\nabla_{W_g} \mathcal{L}_{\text{total}} \neq 0$ | Diagnostic suite passed ($4/4$ tests) | **VERIFIED CORRECT**<br>(Complete autograd connectivity) |
| **Artifact Completeness** | All 20 runs completed without missing data or disk overflow | 20 runs complete with checkpoints, event logs, and test metrics | $20/20$ complete | **VERIFIED COMPLETE**<br>(Disk footprint $<4$ GB) |

---

## 3. Verified Statistical Findings

### A. Mean Zero-Shot Cross-Dataset Performance (BUSI $\to$ BUS-BRA)
Across $N=10$ matched random seeds evaluated on $N=1,063$ external images:
- **Foreground IoU**: Vanilla Swin $57.50\% \pm 4.74\%$ vs. Standard-MLP MoE $57.85\% \pm 2.15\%$
  - Mean Paired Difference: $\bar{D} = +0.35\%$ ($95\%$ CI: $[-3.09\%, +3.79\%]$)
  - Paired $t$-test: $t = 0.2319, \; p = 0.8218$
  - Wilcoxon signed-rank test: $W = 25.0, \; p = 0.8457$
  - Standardized Effect Size: Cohen's $d_z = 0.0733$ (negligible)
- **Foreground Dice**: Vanilla Swin $72.91\% \pm 3.98\%$ vs. Standard-MLP MoE $73.28\% \pm 1.73\%$
  - Mean Paired Difference: $\bar{D} = +0.37\%$ ($95\%$ CI: $[-2.48\%, +3.22\%]$)
  - Paired $t$-test: $t = 0.2942, \; p = 0.7753$
  - Wilcoxon signed-rank test: $W = 25.0, \; p = 0.8457$
  - Cohen's $d_z = 0.0930$ (negligible)
- **Win / Loss Ratio**: Standard-MLP MoE achieved higher Foreground IoU on **6 seeds**; Vanilla Swin achieved higher Foreground IoU on **4 seeds**.

### B. Variance Analysis & Multiple Testing Correction
- **Classical $F$-Test**:
  - Sample variance: $s_{\text{Vanilla}}^2 = 2.246 \times 10^{-3}$ vs. $s_{\text{MoE}}^2 = 4.613 \times 10^{-4}$
  - Variance ratio: $F(9, 9) = 4.8702, \; p_{\text{uncorrected}} = 0.0273$.
- **Assumption Violation**:
  - The Shapiro–Wilk test on Vanilla Swin strongly rejects normality ($p = 0.0068$ for FG IoU, $p < 0.006$ across all overlap metrics). The low $F$-test $p$-value is driven by two catastrophic collapse outliers (Seeds 123 and 2026), violating the fundamental requirement of the $F$-test.
- **Robust Brown–Forsythe Test**:
  - Median-centered homoscedasticity test: $p = 0.3552$ (Foreground IoU), $p = 0.3517$ (Foreground Dice). **Statistically non-significant**.
- **Family-Wise Error Rate Control**:
  - Bonferroni-adjusted $p$-value across the 7 evaluated metrics: $p_{\text{adj}} = \min(1.0, 7 \times 0.0273) = \mathbf{0.1913}$.
  - Holm-Bonferroni adjusted $p$-value: $p_{\text{adj}} = \mathbf{0.1367}$.
  - Benjamini–Hochberg False Discovery Rate ($q=0.05$): Zero metrics significant.

---

## 4. The Model Capacity Confound

Direct measurement of model state dicts revealed:
- **Vanilla Swin-Tiny**: $29,290,365$ trainable parameters.
- **Standard-MLP Swin-MoE**: $81,148,989$ trainable parameters ($2.77\times$ capacity).
- **Confound Analysis**:
  Because MoE introduces $+51.86$ million additional parameters, any empirical stability observed across the 10 seeds cannot be causally attributed to conditional routing. It is plausible that an $81\text{M}$-parameter dense model would exhibit identical stabilization due to overparameterization.
- **Remedy**: A parameter-matched dense control (Dense Swin-Tiny Wide-FFN with expansion ratio $\times 16$, matching $81.14\text{M}$ parameters) is required to decouple capacity from routing.

---

## 5. Most Defensible Scientific Conclusions

When submitting to peer-reviewed venues (e.g., MICCAI, IEEE TMI):

1. **On Mean Generalization**:
   Standard-MLP Swin-MoE does not provide a statistically significant improvement in mean zero-shot cross-dataset generalization over Vanilla Swin-Tiny on breast ultrasound segmentation ($\Delta = +0.35\%, p = 0.8218$).
2. **On Optimization Stability**:
   Across the 10 tested seeds, Standard-MLP MoE demonstrated empirical stability (all seeds within $54.18\% \dots 60.42\%$), whereas Vanilla Swin suffered severe initialization collapse on 2 out of 10 seeds ($47.33\%$ and $50.69\%$).
3. **On Scientific Rigor**:
   This stability difference is not statistically significant after multiple-comparison correction ($p_{\text{adj}} = 0.1913$, Brown–Forsythe $p = 0.3552$) and is confounded by a $2.77\times$ increase in parameter capacity.
4. **On Sample Size Lessons**:
   The apparent $+5.96\%$ MoE lead observed during the 3-seed Phase 9 audit was an artifact of small-sample selection bias. Increasing the sample size to 10 matched seeds revealed the true parity of mean performance.
