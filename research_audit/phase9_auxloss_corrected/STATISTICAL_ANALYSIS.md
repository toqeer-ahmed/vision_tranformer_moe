# Statistical Hypothesis Testing & Power Analysis: 10-Seed MoE Validation

**Auditor & Lead Statistician**: Senior Computer Vision Research Engineer & Scientific Auditor  
**Repository**: `D:\Study Material\Research\CV\vision_transformer_research`  
**Target Milestone**: Phase 9 Corrected Multi-Seed Validation  
**Sample Size**: $N = 10$ Matched Random Seeds ($df = 9$)  
**Benchmark Target**: Zero-Shot Generalization on BUS-BRA ($N = 1,063$)  

---

## 1. Methodology: Matched-Pair Statistical Framework

In medical imaging experiments, comparing two architectures using independent two-sample tests is invalid because both architectures are evaluated on the exact same dataset splits, random seeds, and patient cases. The appropriate statistical framework is a **Matched-Pair Test**.

Let $Y_{\text{MoE}, s}$ and $Y_{\text{Vanilla}, s}$ denote the zero-shot evaluation metric (Foreground IoU, Foreground Dice, etc.) for random seed $s \in \mathcal{S} = \{42, 123, 314, 777, 999, 1001, 2026, 2027, 2718, 9001\}$.

The paired difference for each seed is defined as:
$$D_s = Y_{\text{MoE}, s} - Y_{\text{Vanilla}, s}$$

For $N = 10$ seeds:
- Sample mean paired difference: $\bar{D} = \frac{1}{10} \sum_{s=1}^{10} D_s = +0.0035$ ($+0.35\%$)
- Sample standard deviation of differences: $s_D = \sqrt{\frac{1}{9} \sum_{s=1}^{10} (D_s - \bar{D})^2} = 0.0483$ ($4.83\%$)
- Standard Error of the Mean: $\text{SE}(\bar{D}) = \frac{s_D}{\sqrt{10}} = 0.0153$ ($1.53\%$)

---

## 2. Statistical Hypothesis Test Results

### A. Two-Tailed Paired Student's $t$-Test
- **Null Hypothesis ($H_0$)**: $\mu_D = 0$ (Conditional MoE routing has no effect on zero-shot segmentation performance).
- **Alternative Hypothesis ($H_1$)**: $\mu_D \neq 0$ (Conditional MoE routing produces a significant difference in segmentation performance).
- **Test Statistic**:
  $$t = \frac{\bar{D}}{\text{SE}(\bar{D})} = \frac{0.0035}{0.0153} = 0.2319$$
- **Degrees of Freedom**: $df = 9$.
- **Critical Threshold ($\alpha = 0.05$)**: $t_{\text{crit}} = \pm 2.262$.
- **$p$-value**: $p = 0.8218$.
- **Decision**: $|t| = 0.2319 < 2.262$. **Fail to reject $H_0$**. There is no statistically significant difference in mean performance.

---

### B. Wilcoxon Signed-Rank Test (Non-Parametric)
Because ultrasound segmentation metrics can exhibit skewness across random seeds:
- Test statistic $W = 25.0$.
- Two-sided $p$-value: $p = 0.8457$.
- **Decision**: $p = 0.8457 \gg 0.05$. **Fail to reject $H_0$**. Non-parametric verification confirms that the median performance difference is indistinguishable from zero.

---

### C. Standardized Effect Size: Cohen's $d_z$
$$d_z = \frac{\bar{D}}{s_D} = \frac{0.0035}{0.0483} = 0.0733$$
- **Standard Interpretation**:
  - $|d_z| < 0.20$: **Negligible effect**
  - $0.20 \le |d_z| < 0.50$: Small effect
  - $0.50 \le |d_z| < 0.80$: Medium effect
  - $|d_z| \ge 0.80$: Large effect
- **Finding**: $d_z = 0.0733$ confirms that the observed mean difference is negligible in magnitude.

---

### D. 95% Confidence Interval
$$\text{CI}_{95\%} = \left[ \bar{D} - t_{\text{crit}} \cdot \text{SE}(\bar{D}), \quad \bar{D} + t_{\text{crit}} \cdot \text{SE}(\bar{D}) \right]$$
$$\text{CI}_{95\%} = \left[ +0.35\% - (2.262 \times 1.53\%), \quad +0.35\% + (2.262 \times 1.53\%) \right] = [-3.09\%, +3.79\%]$$
Because the 95% confidence interval cleanly spans zero, an advantage for Vanilla Swin or MoE cannot be claimed.

---

## 3. The Real Scientific Discovery: Variance Ratio $F$-Test

While the difference in *means* is not statistically significant, the difference in *variances* is highly significant:

### Variance Ratio Test on Cross-Dataset Performance:
- Vanilla Swin sample variance: $s_{\text{vanilla}}^2 = (0.0474)^2 = 2.246 \times 10^{-3}$
- Standard-MLP MoE sample variance: $s_{\text{moe}}^2 = (0.0215)^2 = 4.613 \times 10^{-4}$
- Variance Ratio:
  $$F = \frac{s_{\text{vanilla}}^2}{s_{\text{moe}}^2} = \frac{2.246 \times 10^{-3}}{4.613 \times 10^{-4}} = 4.8702$$
- Degrees of freedom: $df_1 = 9, \; df_2 = 9$.
- Critical value ($\alpha = 0.05$, two-tailed): $F_{\text{crit}} = 4.026$.
- Two-tailed $p$-value: **$p = 0.0273$**.
- **Decision**: $F = 4.8702 > 4.026$, $p = 0.0273 < 0.05$. **Reject $H_0$ of equal variances**.

### Multi-Metric Variance Test Summary:
| Metric | Vanilla Variance ($s_V^2$) | MoE Variance ($s_{\text{MoE}}^2$) | Variance Ratio $F$ | Two-Tailed $p$-value | Significant at $\alpha=0.05$? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Foreground IoU** | $2.246 \times 10^{-3}$ | $4.613 \times 10^{-4}$ | **$4.8702$** | **$0.0273$** | **YES** |
| **Foreground Dice** | $1.584 \times 10^{-3}$ | $3.010 \times 10^{-4}$ | **$5.2623$** | **$0.0212$** | **YES** |
| **Macro mIoU** | $7.952 \times 10^{-4}$ | $1.726 \times 10^{-4}$ | **$4.6067$** | **$0.0327$** | **YES** |
| **Macro mDice** | $4.973 \times 10^{-4}$ | $9.868 \times 10^{-5}$ | **$5.0393$** | **$0.0244$** | **YES** |

---

## 4. Why the Original Phase 9 Audit Saw a Fake "+5.96%" Gain

The original Phase 9 audit evaluated only 3 seeds: `42`, `123`, and `2026`.
Looking at the 10-seed table:
- **Seed 123**: Vanilla collapsed to $47.33\%$ IoU; MoE achieved $54.18\%$ ($\Delta = +6.85\%$).
- **Seed 2026**: Vanilla collapsed to $50.69\%$ IoU; MoE achieved $59.25\%$ ($\Delta = +8.56\%$).
- **Seed 42**: Vanilla achieved $56.78\%$ IoU; MoE achieved $59.24\%$ ($\Delta = +2.46\%$).

Average delta across these 3 seeds: **$+5.96\%$**.
This demonstrates why evaluating only 3 seeds is scientifically dangerous:
Vanilla Swin happened to suffer initialization failures on 2 out of those 3 seeds, creating the false illusion of a massive $+5.96\%$ MoE architectural advantage.
When tested across 10 seeds, Vanilla Swin succeeded on seeds 314, 777, 999, and 9001, outperforming MoE on 4 out of 10 seeds and bringing the true mean difference down to $+0.35\%$.

---

## 5. Clinical & Academic Formulation for Publication

When writing the paper for submission to MICCAI / IEEE TMI:

> **Recommended Paper Framing**:  
> *"Mixture-of-Experts with auxiliary load-balancing loss does not act as a capacity amplifier for raw out-of-distribution segmentation accuracy ($\Delta = +0.35\%$, $p = 0.8218$). Rather, it functions as a structural regularizer against catastrophic initialization collapse. While standard Vision Transformers suffer severe seed sensitivity on small medical cohorts (variance $\sigma^2 = 2.25 \times 10^{-3}$, minimum IoU $47.33\%$), conditional MoE routing bounds the worst-case performance floor to $54.18\%$ and provides a statistically significant $79.5\%$ reduction in cross-dataset variance ($F = 4.87, p = 0.0273$)."*
