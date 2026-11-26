# Phase 11 Statistical Validation: Attention U-Net Cross-Model Evaluation

**Document Identifier**: `research_audit/phase11_attention_unet_final/PHASE11_STATISTICAL_VALIDATION.md`  
**Workspace**: `D:\Study Material\Research\CV\vision_transformer_research`  
**Auditor**: Senior Computer Vision Research Engineer & Scientific Auditor  
**Date**: 2026-11-26  
**Cohort**: 10 Matched Seeds ($\mathcal{S} = \{42, 123, 314, 777, 999, 1001, 2026, 2027, 2718, 9001\}$)  
**Evaluation Target**: BUS-BRA Zero-Shot Binary Segmentation ($N = 1,063$)  

---

## 1. Master Metric Summary (Mean ± Sample Standard Deviation)

The table below summarizes the empirical distributions of all four audited architectures evaluated zero-shot on BUS-BRA across 10 matched seeds:

| Metric | Attention U-Net ($31.39\text{M}$) | Standard U-Net ($31.04\text{M}$) | Vanilla Swin-Tiny ($29.29\text{M}$) | Standard-MLP Swin-MoE ($81.15\text{M}$) |
| :--- | :---: | :---: | :---: | :---: |
| **Foreground IoU** | $45.35\% \pm 8.64\%$ | $50.79\% \pm 3.68\%$ | $57.50\% \pm 4.74\%$ | **$57.85\% \pm 2.15\%$** |
| **Foreground Dice** | $61.93\% \pm 8.84\%$ | $67.29\% \pm 3.33\%$ | $72.91\% \pm 3.98\%$ | **$73.28\% \pm 1.73\%$** |
| **Macro mIoU** | $69.91\% \pm 4.60\%$ | $72.82\% \pm 1.93\%$ | $76.10\% \pm 2.82\%$ | **$76.30\% \pm 1.31\%$** |
| **Macro mDice** | $79.54\% \pm 4.57\%$ | $82.32\% \pm 1.72\%$ | $85.09\% \pm 2.23\%$ | **$85.29\% \pm 0.99\%$** |
| **Foreground Precision** | **$81.38\% \pm 4.38\%$** | $80.19\% \pm 4.73\%$ | $70.44\% \pm 6.61\%$ | $69.92\% \pm 4.33\%$ |
| **Foreground Recall** | $51.33\% \pm 11.32\%$ | $58.57\% \pm 6.49\%$ | $76.51\% \pm 6.79\%$ | **$77.35\% \pm 3.73\%$** |
| **Pixel Accuracy** | $94.71\% \pm 0.61\%$ | **$95.11\% \pm 0.28\%$** | $95.06\% \pm 0.98\%$ | **$95.11\% \pm 0.50\%$** |
| **Tumor Volume Ratio** | $0.638\times \pm 0.162\times$ | $0.736\times \pm 0.121\times$ | $1.101\times \pm 0.181\times$ | **$1.112\times \pm 0.114\times$** |

---

## 2. Planned Comparison 1: Attention U-Net vs Standard U-Net (Ablation of Attention Gates)

This comparison evaluates the specific architectural contribution of adding 4 Additive Attention Gates ($\alpha = \sigma(\psi^T \text{ReLU}(W_g^T g + W_x^T x))$) to the canonical U-Net architecture.

$$\Delta = \text{Attention U-Net} - \text{Standard U-Net}$$

| Metric | Mean $\Delta$ | $95\%$ Confidence Interval | Paired $t$-stat | Raw $p$ ($t$-test) | Holm-Adjusted $p$ | Wilcoxon $W$ | Wilcoxon $p$ | Cohen's $d_z$ | Shapiro-Wilk $p$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Foreground IoU** | $-5.44\%$ | $[-12.66\%, +1.78\%]$ | $-1.706$ | $0.1225$ | $0.7168$ | $15.0$ | $0.1602$ | $-0.54$ | $0.115$ |
| **Foreground Dice** | $-5.36\%$ | $[-12.59\%, +1.87\%]$ | $-1.681$ | $0.1277$ | $0.7168$ | $15.0$ | $0.1602$ | $-0.53$ | $0.060$ |
| **Macro mIoU** | $-2.91\%$ | $[-6.73\%, +0.92\%]$ | $-1.722$ | $0.1195$ | $0.7168$ | $14.0$ | $0.1309$ | $-0.54$ | $0.106$ |
| **Macro mDice** | $-2.78\%$ | $[-6.51\%, +0.95\%]$ | $-1.691$ | $0.1259$ | $0.7168$ | $15.0$ | $0.1602$ | $-0.53$ | $0.058$ |
| **Foreground Precision** | $+1.19\%$ | $[-3.17\%, +5.55\%]$ | $+0.618$ | $0.5518$ | $0.7168$ | $23.0$ | $0.5566$ | $+0.20$ | $0.895$ |
| **Foreground Recall** | $-7.24\%$ | $[-17.41\%, +2.93\%]$ | $-1.614$ | $0.1416$ | $0.7168$ | $13.0$ | $0.1055$ | $-0.51$ | $0.284$ |
| **Pixel Accuracy** | $-0.40\%$ | $[-0.88\%, +0.08\%]$ | $-1.884$ | $0.0913$ | $0.6392$ | $14.0$ | $0.1309$ | $-0.60$ | $0.041$ |
| **Volume Ratio** | $-0.099\times$ | $[-0.253, +0.055]$ | $-1.452$ | $0.1808$ | — | $16.0$ | $0.1934$ | $-0.46$ | $0.637$ |

### Findings & Hypothesis Verification:
- **Falsification of Recall Hypothesis**: The central hypothesis that Attention Gates improve foreground recall under domain shift is **falsified**. Instead of increasing, Foreground Recall declined from $58.57\%$ to $51.33\%$ ($\Delta = -7.24\%$).
- **Variance Inflation**: Attention U-Net exhibited dramatically higher variance than Standard U-Net ($s = 8.64\%$ vs $3.68\%$, variance ratio $F = 5.50, p = 0.0163$ uncorrected).
- **Non-Significance after Correction**: After Holm-Bonferroni correction, the mean drop of $-5.44\%$ IoU is not statistically significant ($p_{\text{Holm}} = 0.7168$) due to large standard error ($s_{\Delta} = 10.09\%$) induced by severe collapses on seeds 42 and 123.

---

## 3. Planned Comparison 2: Attention U-Net vs Vanilla Swin-Tiny (Parameter Parity ~31M vs ~29M)

This comparison contrasts a Convolutional Attention network against a Hierarchical Vision Transformer at near-identical parameter count ($31.39\text{M}$ vs $29.29\text{M}$, a $+7.2\%$ parameter difference).

$$\Delta = \text{Attention U-Net} - \text{Vanilla Swin-Tiny}$$

| Metric | Mean $\Delta$ | $95\%$ Confidence Interval | Paired $t$-stat | Raw $p$ ($t$-test) | Holm-Adjusted $p$ | Wilcoxon $W$ | Wilcoxon $p$ | Cohen's $d_z$ | Shapiro-Wilk $p$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Foreground IoU** | **$-12.15\%$** | $[-17.59\%, -6.71\%]$ | $-5.060$ | $\mathbf{0.00069}$ | $\mathbf{0.0041}$ | $2.0$ | $\mathbf{0.0039}$ | $-1.60$ | $0.527$ |
| **Foreground Dice** | **$-10.98\%$** | $[-16.50\%, -5.46\%]$ | $-4.498$ | $\mathbf{0.00150}$ | $\mathbf{0.0059}$ | $2.0$ | $\mathbf{0.0039}$ | $-1.42$ | $0.394$ |
| **Macro mIoU** | **$-6.20\%$** | $[-9.00\%, -3.40\%]$ | $-5.011$ | $\mathbf{0.00073}$ | $\mathbf{0.0041}$ | $2.0$ | $\mathbf{0.0039}$ | $-1.58$ | $0.345$ |
| **Macro mDice** | **$-5.55\%$** | $[-8.34\%, -2.77\%]$ | $-4.521$ | $\mathbf{0.00147}$ | $\mathbf{0.0059}$ | $2.0$ | $\mathbf{0.0039}$ | $-1.43$ | $0.358$ |
| **Foreground Precision** | **$+10.94\%$** | $[+3.78\%, +18.10\%]$ | $+3.435$ | $\mathbf{0.00720}$ | $\mathbf{0.0144}$ | $2.0$ | $\mathbf{0.0039}$ | $+1.09$ | $0.272$ |
| **Foreground Recall** | **$-25.19\%$** | $[-35.76\%, -14.62\%]$ | $-5.385$ | $\mathbf{0.00044}$ | $\mathbf{0.0031}$ | $2.0$ | $\mathbf{0.0039}$ | $-1.70$ | $0.300$ |
| **Pixel Accuracy** | $-0.35\%$ | $[-0.88\%, +0.17\%]$ | $-1.520$ | $0.1628$ | $0.1628$ | $18.0$ | $0.2754$ | $-0.48$ | $0.067$ |
| **Volume Ratio** | **$-0.463\times$** | $[-0.686, -0.239]$ | $-4.681$ | $\mathbf{0.00114}$ | — | $2.0$ | $\mathbf{0.0039}$ | $-1.48$ | $0.388$ |

### Findings:
- **Statistically Significant Transformer Superiority**: Vanilla Swin-Tiny significantly outperforms Attention U-Net across Foreground IoU ($-12.15\%, p_{\text{Holm}} = 0.0041$), Foreground Dice ($-10.98\%, p_{\text{Holm}} = 0.0059$), Macro mIoU ($-6.20\%, p_{\text{Holm}} = 0.0041$), Macro mDice ($-5.55\%, p_{\text{Holm}} = 0.0059$), and Recall ($-25.19\%, p_{\text{Holm}} = 0.0031$).
- **Large Effect Sizes**: Cohen's $|d_z| \ge 1.42$ across all segmentation overlap metrics, confirming substantial, practically meaningful deficits.

---

## 4. Planned Comparison 3: Attention U-Net vs Standard-MLP Swin-MoE

This comparison contrasts Attention U-Net against the Top-2 Mixture-of-Experts Vision Transformer.

$$\Delta = \text{Attention U-Net} - \text{Standard-MLP Swin-MoE}$$

| Metric | Mean $\Delta$ | $95\%$ Confidence Interval | Paired $t$-stat | Raw $p$ ($t$-test) | Holm-Adjusted $p$ | Wilcoxon $W$ | Wilcoxon $p$ | Cohen's $d_z$ | Shapiro-Wilk $p$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Foreground IoU** | **$-12.50\%$** | $[-18.31\%, -6.69\%]$ | $-4.869$ | $\mathbf{0.00089}$ | $\mathbf{0.0046}$ | $1.0$ | $\mathbf{0.00195}$ | $-1.54$ | $0.002$* |
| **Foreground Dice** | **$-11.35\%$** | $[-17.36\%, -5.34\%]$ | $-4.270$ | $\mathbf{0.00208}$ | $\mathbf{0.0062}$ | $1.0$ | $\mathbf{0.00195}$ | $-1.35$ | $0.001$* |
| **Macro mIoU** | **$-6.39\%$** | $[-9.50\%, -3.29\%]$ | $-4.654$ | $\mathbf{0.00119}$ | $\mathbf{0.0048}$ | $1.0$ | $\mathbf{0.00195}$ | $-1.47$ | $0.002$* |
| **Macro mDice** | **$-5.75\%$** | $[-8.86\%, -2.64\%]$ | $-4.180$ | $\mathbf{0.00237}$ | $\mathbf{0.0062}$ | $1.0$ | $\mathbf{0.00195}$ | $-1.32$ | $0.001$* |
| **Foreground Precision** | **$+11.46\%$** | $[+6.25\%, +16.68\%]$ | $+4.958$ | $\mathbf{0.00077}$ | $\mathbf{0.0046}$ | $3.0$ | $\mathbf{0.00586}$ | $+1.57$ | $0.301$ |
| **Foreground Recall** | **$-26.02\%$** | $[-34.30\%, -17.74\%]$ | $-7.119$ | $\mathbf{0.00006}$ | $\mathbf{0.00039}$ | $1.0$ | $\mathbf{0.00195}$ | $-2.25$ | $0.519$ |
| **Pixel Accuracy** | $-0.40\%$ | $[-0.92\%, +0.12\%]$ | $-1.722$ | $0.1185$ | $0.1185$ | $17.0$ | $0.2324$ | $-0.55$ | $0.355$ |
| **Volume Ratio** | **$-0.475\times$** | $[-0.627, -0.322]$ | $-7.060$ | $\mathbf{0.00006}$ | — | $1.0$ | $\mathbf{0.00195}$ | $-2.23$ | $0.314$ |

*\*Note: Non-normal distribution detected via Shapiro-Wilk test for IoU/Dice differences ($p \le 0.002$); the non-parametric Wilcoxon signed-rank test rigorously confirms significance ($p = 0.00195$).*

### Findings:
- In 9 out of 10 matched seeds, Swin-MoE decisively beat Attention U-Net (and in the 1 remaining seed, seed 2026, Swin-MoE IoU was $59.25\%$ vs Attention U-Net's $51.84\%$).
- Attention U-Net suffers a massive **$-26.02\%$ deficit in Foreground Recall** ($p_{\text{Holm}} = 0.00039$, $d_z = -2.25$), underscoring the inability of spatial attention gates to capture faint cross-domain lesion margins.

---

## 5. Homoscedasticity & Variance Ratio Audit

Variance ratios and Levene's tests were conducted to test whether Attention U-Net stabilized or destabilized training:

| Baseline Compared | Metric | $\text{Var}(\text{Attn\_UNet})$ | $\text{Var}(\text{Baseline})$ | Variance Ratio $F$ | Uncorrected $p$ ($F$-test) | Levene / Brown-Forsythe $p$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Standard U-Net** | Fg IoU | $0.00746$ | $0.00136$ | **$5.50$** | $\mathbf{0.0163}$ | $0.0894$ |
| **Standard U-Net** | Fg Dice | $0.00781$ | $0.00111$ | **$7.02$** | $\mathbf{0.0069}$ | $0.0607$ |
| **Vanilla Swin** | Fg IoU | $0.00746$ | $0.00225$ | **$3.32$** | $0.0886$ | $0.2875$ |
| **Swin-MoE** | Fg IoU | $0.00746$ | $0.00046$ | **$16.18$** | $\mathbf{0.0002}$ | $\mathbf{0.0097}$ |

### Takeaway on Variance:
Adding attention gates to U-Net did **not** act as a variance buffer; on the contrary, it **multiplied empirical variance by $5.5\times$** compared to Standard U-Net ($p = 0.0163$) and **$16.2\times$** compared to Swin-MoE ($p = 0.0002$). Attention gates introduce routing/gating sensitivities that are prone to pathological suppression under domain shift.
