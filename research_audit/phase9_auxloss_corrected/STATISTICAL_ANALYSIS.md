# Statistical Analysis Protocol for Multi-Seed MoE Evaluation

**Auditor**: Senior Computer Vision Research Engineer & Scientific Auditor  
**Repository**: `D:\Study Material\Research\CV\vision_transformer_research`  
**Target Milestone**: Phase 9 Corrected Multi-Seed Validation  

---

## 1. Methodology: Matched-Pair Statistical Framework

In machine learning experiments, comparing two architectures using independent two-sample t-tests is invalid because both architectures are evaluated on the exact same dataset splits and random seeds. The correct statistical framework is a **Matched-Pair Test**.

Let $Y_{\text{MoE}, s}$ and $Y_{\text{Vanilla}, s}$ denote the evaluation metric (e.g., Foreground IoU) for seed $s \in \mathcal{S}$.  
The paired difference is defined as:
$$D_s = Y_{\text{MoE}, s} - Y_{\text{Vanilla}, s}$$

For a set of $N$ seeds, the sample mean paired difference is:
$$\bar{D} = \frac{1}{N} \sum_{s=1}^N D_s$$

The sample standard deviation of paired differences is:
$$s_D = \sqrt{\frac{1}{N-1} \sum_{s=1}^N (D_s - \bar{D})^2}$$

The standard error of the mean difference is:
$$\text{SE}(\bar{D}) = \frac{s_D}{\sqrt{N}}$$

---

## 2. Hypothesis Testing Formulations

### A. Two-Tailed Paired Student's t-Test
- **Null Hypothesis ($H_0$)**: $\mu_D = 0$ (Conditional MoE routing has no effect on zero-shot segmentation performance).
- **Alternative Hypothesis ($H_1$)**: $\mu_D \neq 0$ (Conditional MoE routing produces a significant difference in segmentation performance).
- **Test Statistic**:
  $$t = \frac{\bar{D}}{\text{SE}(\bar{D})} = \frac{\bar{D} \sqrt{N}}{s_D}$$
- **Degrees of Freedom**: $df = N - 1$.
- **Decision Rule**: Reject $H_0$ at significance level $\alpha = 0.05$ if $|t| > t_{\text{crit}, df}$.

### B. Wilcoxon Signed-Rank Test (Non-Parametric Alternative)
Because ultrasound segmentation metrics on small sample sizes may exhibit non-Gaussian distributions, the **Wilcoxon signed-rank test** must be computed in parallel:
1. Rank the absolute differences $|D_s|$ in ascending order.
2. Sum the ranks corresponding to positive differences ($W^+$) and negative differences ($W^-$).
3. The test statistic is $W = \min(W^+, W^-)$.
4. Reject $H_0$ if $W \le W_{\text{crit}}$ for sample size $N$.

---

## 3. Power Analysis & Sample Size Constraints

The table below illustrates the critical threshold requirements as the seed count increases:

| Seed Count ($N$) | Degrees of Freedom ($df$) | Critical Value $t_{\text{crit}}$ ($\alpha=0.05$) | Minimum Detectable Cohen's $d_z$ ($80\%$ Power) | Statistical Verdict |
| :---: | :---: | :---: | :---: | :--- |
| **$N = 3$** (Original Phase 9) | 2 | **4.303** | $\sim 2.50$ | **INSUFFICIENT**. Extremely underpowered; can only detect massive, unrealistic effects. |
| **$N = 5$** (Primary Corrected Tier) | 4 | **2.776** | $\sim 1.50$ | **ACCEPTABLE EXPLORATORY**. Adequate for detecting strong medium-to-large effects. |
| **$N = 10$** (Extended Corrected Tier)| 9 | **2.262** | $\sim 0.95$ | **RECOMMENDED PUBLICATION STANDARD**. Robust power to detect real $+1.5\% - +2.5\%$ IoU gains. |

> **Audit Insight**:  
> With $N=3$, $|t|$ must exceed $4.303$ to achieve $p < 0.05$. In the original Phase 9 run, the mean gain in Foreground IoU was $+2.03\%$, but because $N=3$, claiming formal statistical confirmation was mathematically invalid. Expanding to $N=5$ or $N=10$ provides the necessary degrees of freedom.

---

## 4. Effect Size & Confidence Intervals

In addition to $p$-values, research publications must report standardized effect sizes and confidence intervals:

### A. Cohen's $d_z$ for Paired Samples:
$$d_z = \frac{\bar{D}}{s_D}$$
- **Interpretation**:
  - $|d_z| \ge 0.2$: Small effect
  - $|d_z| \ge 0.5$: Medium effect
  - $|d_z| \ge 0.8$: Large effect
  - $|d_z| \ge 1.2$: Very large effect

### B. Two-Sided $95\%$ Confidence Interval:
$$\text{CI}_{95\%} = \left[ \bar{D} - t_{\text{crit}, df} \cdot \frac{s_D}{\sqrt{N}}, \quad \bar{D} + t_{\text{crit}, df} \cdot \frac{s_D}{\sqrt{N}} \right]$$
If the $95\%$ CI does not contain $0.0$, the improvement is statistically significant at $\alpha = 0.05$.

---

## 5. Distinction: Statistical Significance vs. Clinical Usefulness

A statistically significant result ($p < 0.05$) does not automatically translate to clinical utility:
1. **Statistical Significance**: Demonstrates that the $+2.03\%$ Foreground IoU improvement is unlikely to have occurred purely by random initialization chance.
2. **Clinical Utility**: In ultrasound diagnostic workflows, a Foreground IoU of $58.11\%$ indicates that the model correctly outlines the bulk of the lesion, but boundary margins still deviate from expert sonographer annotations by $>40\%$. The model is an effective proof-of-concept for cross-hardware robustness, but cannot be deployed as an autonomous diagnostic instrument.

