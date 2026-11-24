# Critical Variance & Distribution Analysis: Phase 9B

**Experiment Identifier**: `phase9_auxloss_corrected`  
**Auditor**: Senior Computer Vision Research Engineer & Scientific Auditor  
**Repository**: `D:\Study Material\Research\CV\vision_transformer_research`  
**Sample Size**: $N = 10$ Matched Random Seeds  
**Benchmark Target**: Zero-Shot Generalization on BUS-BRA ($N = 1,063$)  

---

## 1. Executive Summary of Variance Findings

In preliminary audits, Standard-MLP Swin-MoE was reported to achieve a "statistically significant $79.5\%$ reduction in cross-dataset performance variance" based on a classical two-sample variance ratio $F$-test ($F = 4.87, p = 0.0273$).

This comprehensive audit investigated:
1. The **normality assumptions** underlying the classical $F$-test.
2. Robust, distribution-free alternatives (**Brown–Forsythe and Levene tests**).
3. **Multiple-comparison corrections** across all seven evaluated metrics.
4. The statistical validity of the **"guaranteed reliability floor"** claim.

### Definitive Statistical Verdict:
> **The variance-reduction claim DOES NOT survive multiple-comparison correction.**  
> Furthermore, because Vanilla Swin violates the normality assumption ($p = 0.0068$), the classical $F$-test is mathematically invalid. When robust, median-centered homoscedasticity tests (Brown–Forsythe) are applied, the variance difference is **statistically non-significant ($p = 0.3552$)**.

---

## 2. Multi-Metric Variance Ratios & Classical $F$-Tests

The sample variance of metric $Y$ across $N=10$ seeds is computed as:
$$s^2 = \frac{1}{N-1} \sum_{s=1}^N (Y_s - \bar{Y})^2$$

The classical variance ratio statistic is:
$$F = \frac{s_{\text{Vanilla}}^2}{s_{\text{MoE}}^2}, \quad df_1 = 9, \; df_2 = 9$$

| Metric | Vanilla Variance ($s_V^2$) | MoE Variance ($s_M^2$) | Variance Ratio ($F$) | Uncorrected $p$-value ($F$) | Bonferroni Adjusted $p$ | Holm Adjusted $p$ | FDR $q=0.05$ Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Foreground IoU** | $2.246 \times 10^{-3}$ | $4.613 \times 10^{-4}$ | **$4.8702$** | $0.0273$ | **$0.1913$** | **$0.1367$** | **NO** |
| **Foreground Dice** | $1.584 \times 10^{-3}$ | $3.010 \times 10^{-4}$ | **$5.2623$** | $0.0212$ | **$0.1481$** | **$0.1481$** | **NO** |
| **Macro mIoU** | $7.952 \times 10^{-4}$ | $1.726 \times 10^{-4}$ | **$4.6067$** | $0.0327$ | **$0.2291$** | **$0.1309$** | **NO** |
| **Macro mDice** | $4.973 \times 10^{-4}$ | $9.868 \times 10^{-5}$ | **$5.0393$** | $0.0244$ | **$0.1710$** | **$0.1466$** | **NO** |
| **Foreground Precision** | $4.364 \times 10^{-3}$ | $1.876 \times 10^{-3}$ | **$2.3262$** | $0.2245$ | **$1.0000$** | **$0.2245$** | **NO** |
| **Foreground Recall** | $4.608 \times 10^{-3}$ | $1.391 \times 10^{-3}$ | **$3.3132$** | $0.0890$ | **$0.6229$** | **$0.1780$** | **NO** |
| **Pixel Accuracy** | $9.605 \times 10^{-5}$ | $2.481 \times 10^{-5}$ | **$3.8722$** | $0.0563$ | **$0.3941$** | **$0.1689$** | **NO** |

### Multiple Comparisons Analysis:
- When testing $m = 7$ metrics simultaneously at family-wise error rate $\alpha = 0.05$:
  - **Bonferroni Critical Threshold**: $\alpha / 7 = 0.00714$. None of the uncorrected $p$-values ($0.0212 \dots 0.2245$) fall below $0.00714$.
  - **Holm-Bonferroni Step-Down**: The lowest $p$-value ($0.0212$ for FG Dice) requires $p < 0.05 / 7 = 0.00714$ to initiate rejection; because $0.0212 > 0.00714$, the testing terminates immediately with zero rejections.
  - **Benjamini-Hochberg False Discovery Rate**: No metric achieves significance under $\text{FDR} \le 0.05$.
- **Conclusion**: Under standard statistical standards for multi-metric reporting, the variance reduction is **not statistically significant**.

---

## 3. Assumption Checking: Non-Normality & Test Invalidation

The classical $F$-test strictly requires that both samples are drawn from normal parent distributions:
$$Y_{\text{Vanilla}} \sim \mathcal{N}(\mu_V, \sigma_V^2), \quad Y_{\text{MoE}} \sim \mathcal{N}(\mu_M, \sigma_M^2)$$

It is a well-known theorem in mathematical statistics (Box, 1953) that the two-sample $F$-test is **extremely sensitive to non-normality**. Moderate kurtosis or skewness causes severe inflation of Type I error rates.

### Shapiro–Wilk Normality Tests:
| Metric | Vanilla Shapiro $p$-value | MoE Shapiro $p$-value | Paired Delta Shapiro $p$-value | Normality Assumption Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **Foreground IoU** | **$0.0068$** | $0.4619$ | $0.0754$ | **REJECTED for Vanilla ($p < 0.01$)** |
| **Foreground Dice** | **$0.0049$** | $0.5117$ | $0.0821$ | **REJECTED for Vanilla ($p < 0.01$)** |
| **Macro mIoU** | **$0.0058$** | $0.4578$ | $0.0760$ | **REJECTED for Vanilla ($p < 0.01$)** |
| **Macro mDice** | **$0.0046$** | $0.5401$ | $0.0822$ | **REJECTED for Vanilla ($p < 0.01$)** |
| **Pixel Accuracy** | **$0.0001$** | $0.5983$ | $0.0034$ | **REJECTED for Vanilla ($p < 0.001$)** |

### Why Did Vanilla Swin Fail Normality?
Vanilla Swin exhibits an **empirical bimodal / left-skewed distribution**:
- $8$ out of $10$ seeds converged normally with high Foreground IoU ($56.78\% \dots 61.33\%$).
- $2$ out of $10$ seeds suffered severe initialization collapse: Seed 123 ($47.33\%$) and Seed 2026 ($50.69\%$).
- These two catastrophic failures produce an inflated sample variance ($s_V^2 = 2.25 \times 10^{-3}$) and heavy negative skewness, causing the classical $F$-test to return a misleadingly low uncorrected $p$-value ($0.0273$).

---

## 4. Robust Homoscedasticity Tests: Brown–Forsythe

Because the normality assumption is rejected, we must apply the **Brown–Forsythe test**, which centers observations on group **medians** rather than means, providing robust protection against skewness and heavy tails.

### Brown–Forsythe Test Results:
| Metric | Brown–Forsythe Statistic | $p$-value | Significant at $\alpha=0.05$? |
| :--- | :---: | :---: | :---: |
| **Foreground IoU** | $0.8981$ | **$0.3552$** | **NO** |
| **Foreground Dice** | $0.9130$ | **$0.3517$** | **NO** |
| **Macro mIoU** | $0.6409$ | **$0.4335$** | **NO** |
| **Macro mDice** | $0.7351$ | **$0.4021$** | **NO** |
| **Foreground Precision** | $0.1729$ | **$0.6826$** | **NO** |
| **Foreground Recall** | $0.2392$ | **$0.6306$** | **NO** |
| **Pixel Accuracy** | $0.1221$ | **$0.7307$** | **NO** |

**Conclusion**: When tested using a statistically sound, distribution-free test, **there is no statistically significant difference in variance between Vanilla Swin and Standard-MLP MoE across any metric**.

---

## 5. Investigation of the "Reliability Floor" Claim

In preliminary discussions, it was suggested that MoE establishes a "guaranteed reliability floor" by raising the minimum observed Foreground IoU from $47.33\%$ to $54.18\%$.

### Observed Extremes:
- **Vanilla Swin Scores**:
  `[47.33%, 50.69%, 56.78%, 58.24%, 59.08%, 59.78%, 60.16%, 60.61%, 60.98%, 61.33%]`
  - $\min = 47.33\%$ (Seed 123)
  - $\max = 61.33\%$ (Seed 9001)
  - Range = $14.00\%$
- **Standard-MLP MoE Scores**:
  `[54.18%, 55.41%, 56.12%, 56.57%, 57.93%, 59.24%, 59.25%, 59.52%, 59.88%, 60.42%]`
  - $\min = 54.18\%$ (Seed 123)
  - $\max = 60.42\%$ (Seed 2718)
  - Range = $6.24\%$
- **Difference Between Observed Minima**: $54.18\% - 47.33\% = \mathbf{+6.85\%}$.

### Why the Observed Minimum is NOT a Guaranteed Future Floor:
From the perspective of **Extreme Value Theory (EVT)** and mathematical statistics:
1. **Sample Minimum vs. Distribution Infimum**:
   The sample minimum $X_{(1)} = \min(X_1, \dots, X_{10})$ is a random variable, not a fixed parameter. It represents only the lowest draw from a sample of size $10$.
2. **Confidence Intervals on Extremes**:
   With only $N=10$ observations, the non-parametric $95\%$ confidence interval for the true population minimum cannot be bounded above $0.0$.
3. **Probability of Future Runs Falling Below $54.18\%$**:
   Under a normal approximation $\mathcal{N}(57.85\%, (2.15\%)^2)$, the probability of an MoE seed falling below $54.18\%$ is:
   $$P(Z < \frac{54.18 - 57.85}{2.15}) = P(Z < -1.707) \approx 0.0439 \; (4.4\%)$$
   In a production deployment of 50 models, the expected number of MoE models falling below $54.18\%$ is $50 \times 0.0439 \approx 2.2$ runs.
4. **Scientific Verdict**:
   Claiming $54.18\%$ is a "guaranteed performance floor" is a statistical error (conflating an empirical sample minimum with a bounded support parameter). It must be described strictly as the **"observed minimum across the 10 tested seeds"**.

---

## 6. Generated Visualizations

The following verified plots were generated from raw data and saved in `research_audit/phase9_auxloss_corrected_final/plots/`:

1. **Per-Seed Paired Difference in Foreground IoU**:
   - File: `plots/paired_difference_foreground_iou.png`
   - Shows the positive deltas for seeds 42, 123, 1001, 2026, 2027, 2718 alongside the negative deltas for seeds 314, 777, 999, 9001.
2. **Distribution & Order Statistics Comparison**:
   - File: `plots/distribution_comparison_foreground_iou.png`
   - Overlays boxplots, matched pairwise connection lines, and annotated minima.
