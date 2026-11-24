# Corrected Phase 9 Results & Evaluation Protocol

**Experiment Identifier**: `phase9_auxloss_corrected`  
**Auditor**: Senior Computer Vision Research Engineer & Scientific Auditor  
**Repository**: `D:\Study Material\Research\CV\vision_transformer_research`  
**Execution Environment**: Kaggle GPU Dual T4 / P100  
**Sample Size**: $N = 10$ Matched Random Seeds (`42`, `123`, `314`, `777`, `999`, `1001`, `2026`, `2027`, `2718`, `9001`)  
**External Evaluation Benchmark**: BUS-BRA Full Dataset ($N = 1,063$ images)  

---

## 1. Evaluation Protocol & Metric Definitions

To ensure complete scientific transparency and prevent metric conflation, all evaluation is partitioned into **Foreground (Clinical Lesion)** metrics and **Macro-Averaged (Background + Lesion)** metrics.

### Evaluated Benchmark Datasets:
1. **Training & In-Domain Validation**: BUSI dataset ($N=780$ ultrasound images, 90/10 train/val split).
2. **Zero-Shot Out-of-Distribution (OOD) Benchmark**: The complete external BUS-BRA dataset ($N=1,063$ images) collected from different ultrasound hardware and clinical institutions.

### Metric Separation:
- **Foreground IoU** ($\text{class\_iou}[1]$): The true boundary intersection-over-union on breast tumors.
- **Foreground Dice** ($\text{class\_dice}[1]$): F1 overlap on clinical lesions.
- **Foreground Precision** ($\text{class\_precision}[1]$): Resistance to false positive artifacts on unfamiliar ultrasound machines.
- **Foreground Recall** ($\text{class\_recall}[1]$): Tumor sensitivity / detection completeness.
- **Background IoU** ($\text{class\_iou}[0]$): Agreement on non-tumor parenchymal tissue ($>94\%$).
- **Macro mIoU**: Arithmetic mean of Background IoU and Foreground IoU.
- **Macro mDice**: Arithmetic mean of Background Dice and Foreground Dice.
- **Pixel Accuracy**: Total correct pixels divided by total evaluated pixels.

---

## 2. Final Empirical Results: 10-Seed Matched Pairs

All 10 seeds were executed across matched pairs for Vanilla Swin-Tiny (29.29M parameters) and Standard-MLP Swin-MoE (81.15M parameters, Top-2 gating across 4 experts with active auxiliary load-balancing loss $\lambda = 0.01$).

### Aggregate Performance Summary (Mean $\pm$ Standard Deviation across 10 Seeds)

| Metric | Vanilla Swin-Tiny (29.29M) | Standard-MLP Swin-MoE (81.15M) | Paired Difference ($\Delta$) | 95% Confidence Interval | Paired $t$-test ($p$-value) | Wilcoxon ($p$-value) | Variance Ratio $F$ ($p$-value) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Foreground IoU** | $57.50\% \pm 4.74\%$ | $57.85\% \pm 2.15\%$ | **$+0.35\%$** | $[-3.09\%, +3.79\%]$ | $t = 0.2319 \; (p = 0.8218)$ | $W = 25.0 \; (p = 0.8457)$ | **$F = 4.87 \; (p = 0.0273)$** |
| **Foreground Dice** | $72.91\% \pm 3.98\%$ | $73.28\% \pm 1.73\%$ | **$+0.37\%$** | $[-2.48\%, +3.22\%]$ | $t = 0.2942 \; (p = 0.7753)$ | $W = 25.0 \; (p = 0.8457)$ | **$F = 5.26 \; (p = 0.0212)$** |
| **Macro mIoU** | $76.10\% \pm 2.82\%$ | $76.30\% \pm 1.31\%$ | **$+0.20\%$** | $[-1.87\%, +2.26\%]$ | $t = 0.2164 \; (p = 0.8335)$ | $W = 25.0 \; (p = 0.8457)$ | **$F = 4.61 \; (p = 0.0327)$** |
| **Macro mDice** | $85.09\% \pm 2.23\%$ | $85.29\% \pm 0.99\%$ | **$+0.20\%$** | $[-1.41\%, +1.80\%]$ | $t = 0.2779 \; (p = 0.7874)$ | $W = 25.0 \; (p = 0.8457)$ | **$F = 5.04 \; (p = 0.0244)$** |
| **Foreground Precision** | $70.44\% \pm 6.61\%$ | $69.92\% \pm 4.33\%$ | **$-0.52\%$** | $[-5.32\%, +4.27\%]$ | $t = -0.2459 \; (p = 0.8113)$ | $W = 24.0 \; (p = 0.7695)$ | $F = 2.33 \; (p = 0.2245)$ |
| **Foreground Recall** | $76.51\% \pm 6.79\%$ | $77.35\% \pm 3.73\%$ | **$+0.83\%$** | $[-3.57\%, +5.24\%]$ | $t = 0.4283 \; (p = 0.6785)$ | $W = 25.0 \; (p = 0.8457)$ | $F = 3.31 \; (p = 0.0890)$ |
| **Pixel Accuracy** | $95.06\% \pm 0.98\%$ | $95.11\% \pm 0.50\%$ | **$+0.04\%$** | $[-0.68\%, +0.77\%]$ | $t = 0.1364 \; (p = 0.8945)$ | $W = 25.0 \; (p = 0.8457)$ | $F = 3.87 \; (p = 0.0563)$ |

---

## 3. Complete Per-Seed Matched Breakdown ($N=10$)

| Seed | Vanilla FG IoU | MoE FG IoU | $\Delta$ FG IoU | Vanilla FG Dice | MoE FG Dice | $\Delta$ FG Dice | Vanilla mIoU | MoE mIoU | Superior Model |
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

- **Win / Loss Ratio**: Standard-MLP MoE outperformed Vanilla Swin on **6 out of 10 seeds**; Vanilla Swin outperformed MoE on **4 out of 10 seeds**.
- **Minimum Performance Floor**:
  - Vanilla Swin worst seed: **$47.33\%$** IoU (Seed 123)
  - Standard-MLP MoE worst seed: **$54.18\%$** IoU (Seed 123) — a **$+6.85\%$ higher worst-case floor**.

---

## 4. Key Scientific Insights

1. **Rejection of the Null Hypothesis for Mean Gain**:
   - The observed mean Foreground IoU advantage is $+0.35\%$ ($p = 0.8218$). The 95% Confidence Interval is $[-3.09\%, +3.79\%]$, which spans zero. Cohen's $d_z = 0.0733$ confirms a negligible effect size.
   - **Conclusion**: There is no statistically significant improvement in mean zero-shot generalization when replacing dense FFNs with Standard-MLP MoE layers on this dataset regime.

2. **Confirmation of Statistically Significant Variance Reduction**:
   - Vanilla Swin Foreground IoU standard deviation: **$4.74\%$** (Variance: $2.246 \times 10^{-3}$)
   - Standard-MLP MoE Foreground IoU standard deviation: **$2.15\%$** (Variance: $4.613 \times 10^{-4}$)
   - Variance ratio $F(9, 9) = 4.8702, \; p = 0.0273$ (**Statistically significant at $\alpha = 0.05$**).
   - **Conclusion**: MoE with auxiliary load balancing acts as an **effective architectural regularizer against catastrophic initialization collapse**, reducing cross-dataset performance variance by **$79.5\%$**.
