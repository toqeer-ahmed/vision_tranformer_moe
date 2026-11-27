# Final Manuscript Numbers and Master Benchmark Tables

**Audit Document:** Definitive Numerical Reference for Five-Model Segmentation Study  
**Lead Auditor & Biostatistician:** Independent ML Research Auditor & Biostatistician  
**Workspace:** `D:\Study Material\Research\CV\vision_transformer_research`  
**Associated Canonical Artifacts:**
- [FINAL_VERIFIED_PER_SEED_VALUES.csv](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/final_cross_phase_integrity_audit/FINAL_VERIFIED_PER_SEED_VALUES.csv)
- [FINAL_VERIFIED_INDOMAIN_RESULTS.csv](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/final_cross_phase_integrity_audit/FINAL_VERIFIED_INDOMAIN_RESULTS.csv)
- [FINAL_VERIFIED_PAIRED_TESTS.csv](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/final_cross_phase_integrity_audit/FINAL_VERIFIED_PAIRED_TESTS.csv)
- [FINAL_MULTIPLICITY_FAMILIES.md](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/final_cross_phase_integrity_audit/FINAL_MULTIPLICITY_FAMILIES.md)
- [FINAL_SOURCE_LEVEL_RECONCILIATION.md](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/final_cross_phase_integrity_audit/FINAL_SOURCE_LEVEL_RECONCILIATION.md)

---

## 1. Auditor Certification of Canonical Numbers

This document provides the definitive, frozen numerical results for publication in scientific manuscripts, conference presentations, and supplementary materials. Every number in this document:
1. Has been derived strictly from raw, per-seed experimental runs across the exact **10 matched seeds** (`[42, 123, 314, 777, 999, 1001, 2026, 2027, 2718, 9001]`).
2. Uses **sample standard deviation** with $ddof=1$.
3. Has been verified against raw confusion matrices, image counts, and JSON execution logs.
4. Distinguishes foreground-specific metrics from whole-image macro averages.
5. Expresses differences strictly in **percentage points (pp)**.

---

## 2. Model Architecture Complexity & Parameter Scaling

| Architecture | Model Family | Encoder Pretraining | Total Parameters | Active Parameters (Inference) | Gating Mechanism | Training Epochs / Protocol |
|---|---|---|:---:|:---:|---|---|
| **Vanilla Swin-Tiny** | Hierarchical ViT | ImageNet-1K | **$29,294,008$ ($29.29\text{M}$)** | **$29,294,008$ ($29.29\text{M}$)** | Dense Attention | 100 Epochs, Cosine Annealing, AdamW |
| **Standard-MLP Swin-MoE** | Sparsely-Gated ViT | ImageNet-1K | **$81,148,989$ ($81.15\text{M}$)** | **$46,644,429$ ($46.64\text{M}$)** | Top-2 Gating ($4$ Experts $\times$ $12$ Blocks) | 100 Epochs, Load-Balancing Aux Loss |
| **Standard U-Net** | Canonical CNN | Scratch | **$31,037,634$ ($31.04\text{M}$)** | **$31,037,634$ ($31.04\text{M}$)** | Plain Skip Connections | 100 Epochs, Cosine Annealing, AdamW |
| **Attention U-Net** | Attn-Gated CNN | Scratch | **$31,388,487$ ($31.39\text{M}$)** | **$31,388,487$ ($31.39\text{M}$)** | Oktay et al. Additive Skip Gates | 100 Epochs, Cosine Annealing, AdamW |
| **ResNet-50 U-Net** | Residual CNN | ImageNet-1K | **$32,520,386$ ($32.52\text{M}$)** | **$32,520,386$ ($32.52\text{M}$)** | 5-Stage ResNet-50 Decoder Skips | 100 Epochs, Cosine Annealing, AdamW |

> [!NOTE] Swin-MoE Parameter Resolution
> The total parameter count of $81.15\text{M}$ reflects the full model containing $4$ experts in the MLP layer of all $12$ Swin Transformer blocks ($48$ total expert MLPs). During any given forward pass, Top-2 gating routes each token through exactly $2$ of the $4$ experts, activating $46.64\text{M}$ parameters ($57.48\%$ of total capacity).

---

## 3. Master Table 1: In-Domain Evaluation (Held-Out BUSI Test Set)

- **Dataset:** BUSI Test Cohort ($N = 78$ images, $3,913,728$ pixels per seed).
- **Protocol:** Models trained on BUSI training splits ($N = 546$ images) and validated on BUSI validation splits ($N = 156$ images).
- **Aggregation:** Sample mean $\pm$ sample SD across 10 matched seeds ($ddof=1$).

| Architecture | In-Domain FG IoU (%) | In-Domain FG Dice (%) | In-Domain FG Recall (%) | In-Domain FG Precision (%) | In-Domain Pixel Accuracy (%) | Best Validation Dice (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Vanilla Swin-Tiny** | $60.24 \pm 6.11$ | $74.97 \pm 4.40$ | $79.34 \pm 6.94$ | $72.05 \pm 5.49$ | $95.58 \pm 0.81$ | $69.76 \pm 3.96$ |
| **Standard-MLP Swin-MoE** | $60.58 \pm 5.28$ | $75.26 \pm 3.96$ | $79.03 \pm 6.31$ | $72.64 \pm 4.54$ | $95.69 \pm 0.70$ | $70.21 \pm 3.32$ |
| **Standard U-Net** | $56.06 \pm 3.88$ | $71.77 \pm 3.17$ | $68.41 \pm 7.47$ | **$76.03 \pm 4.09$** | $95.53 \pm 0.38$ | $63.85 \pm 3.71$ |
| **Attention U-Net** | $56.10 \pm 2.53$ | $71.88 \pm 2.06$ | $68.32 \pm 6.39$ | **$76.62 \pm 4.88$** | $95.51 \pm 0.40$ | $63.12 \pm 3.45$ |
| **ResNet-50 U-Net** | **$59.26 \pm 4.35$** | **$74.34 \pm 3.36$** | **$76.06 \pm 5.86$** | $73.03 \pm 3.93$ | **$95.81 \pm 0.51$** | **$68.22 \pm 3.21$** |

---

## 4. Master Table 2: Zero-Shot External Generalization (BUS-BRA Cohort)

- **Dataset:** External BUS-BRA Cohort ($N = 1,875$ images across $1,063$ patients, $94,080,000$ pixels evaluated per seed).
- **Protocol:** Strict zero-shot transfer without fine-tuning, domain adaptation, calibration, or threshold tuning.
- **Aggregation:** Sample mean $\pm$ sample SD across 10 matched seeds ($ddof=1$).

| Architecture | FG IoU (%) | FG Dice (%) | Macro mIoU (%) | Macro mDice (%) | FG Recall (%) | FG Precision (%) | Pixel Accuracy (%) | 2D Pixel Ratio ($\frac{\text{Pred}}{\text{GT}}$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Vanilla Swin-Tiny** | **$57.50 \pm 4.74$** | **$72.91 \pm 3.98$** | **$76.10 \pm 2.82$** | **$84.58 \pm 2.09$** | **$76.51 \pm 6.79$** | $70.44 \pm 6.61$ | $95.06 \pm 0.98$ | $1.1005 \pm 0.1805$ |
| **Standard-MLP Swin-MoE** | **$57.85 \pm 2.15$** | **$73.28 \pm 1.73$** | **$76.30 \pm 1.31$** | **$84.77 \pm 0.93$** | **$77.35 \pm 3.73$** | $69.92 \pm 4.33$ | $95.11 \pm 0.50$ | $1.1122 \pm 0.1138$ |
| **Standard U-Net** | $50.79 \pm 3.68$ | $67.29 \pm 3.33$ | $72.82 \pm 1.93$ | $82.02 \pm 1.72$ | $58.57 \pm 6.49$ | **$80.19 \pm 4.73$** | **$95.11 \pm 0.28$** | $0.7365 \pm 0.1212$ |
| **Attention U-Net** | $45.35 \pm 8.64$ | $61.93 \pm 8.84$ | $69.91 \pm 4.60$ | $79.13 \pm 4.79$ | $51.33 \pm 11.32$ | **$81.38 \pm 4.38$** | $94.71 \pm 0.61$ | $0.6377 \pm 0.1624$ |
| **ResNet-50 U-Net** | $50.36 \pm 6.98$ | $66.71 \pm 6.48$ | $72.35 \pm 3.84$ | $81.39 \pm 3.54$ | $62.84 \pm 10.34$ | $72.62 \pm 7.13$ | $94.65 \pm 0.82$ | $0.8746 \pm 0.1728$ |

---

## 5. Master Table 3: Cross-Domain Distribution Shift & Paired Transfer Penalty

- **Definition:** $\Delta_{\text{transfer}} = \text{BUS-BRA Zero-Shot FG IoU} - \text{BUSI In-Domain FG IoU}$.
- A **negative** value represents performance degradation under scanner and patient cohort shift.

| Architecture | In-Domain BUSI FG IoU (%) | Zero-Shot BUS-BRA FG IoU (%) | Mean Transfer Penalty ($\Delta_{\text{transfer}}$) | SD of Paired Difference (pp) | Paired 95% Confidence Interval |
|---|:---:|:---:|:---:|:---:|:---:|
| **Vanilla Swin-Tiny** | $60.24 \pm 6.11$ | $57.50 \pm 4.74$ | **$-2.74\text{ pp}$** | $4.17\text{ pp}$ | $[-5.72\text{ pp}, +0.24\text{ pp}]$ |
| **Standard-MLP Swin-MoE** | $60.58 \pm 5.28$ | $57.85 \pm 2.15$ | **$-2.73\text{ pp}$** | $4.89\text{ pp}$ | $[-6.23\text{ pp}, +0.76\text{ pp}]$ |
| **Standard U-Net** | $56.06 \pm 3.88$ | $50.79 \pm 3.68$ | **$-5.27\text{ pp}$** | $5.28\text{ pp}$ | $[-9.05\text{ pp}, -1.49\text{ pp}]$ |
| **ResNet-50 U-Net** | $59.26 \pm 4.35$ | $50.36 \pm 6.98$ | **$-8.91\text{ pp}$** | $8.00\text{ pp}$ | $[-14.63\text{ pp}, -3.18\text{ pp}]$ |
| **Attention U-Net** | $56.10 \pm 2.53$ | $45.35 \pm 8.64$ | **$-10.75\text{ pp}$** | $8.96\text{ pp}$ | $[-17.16\text{ pp}, -4.34\text{ pp}]$ |

---

## 6. Master Table 4: Definitive Pairwise Hypothesis Testing (Zero-Shot BUS-BRA)

All statistical comparisons are paired across the exact 10 matched random seeds ($df=9$, two-tailed).

| Contrast Comparison | Evaluated Metric | Mean Diff (pp) | SD Diff (pp) | SEM (pp) | 95% Confidence Interval | Paired $t$ | Raw $p_t$ | Wilcoxon $W$ | Raw $p_W$ | Cohen's $d_z$ | Holm $p_{\text{adj}}$ (Family B/C, $m=6$) | Holm $p_{\text{adj}}$ (Family A, $m=7$) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Swin-Tiny vs. Attn U-Net** | Foreground IoU | $+12.15$ | $7.61$ | $2.40$ | $[+6.71, +17.59]$ | $+5.052$ | **$0.000688$** | $1.0$ | $0.0039$ | $+1.60$ | **$0.004131$** | **$0.004131$** |
| **Swin-Tiny vs. Std U-Net** | Foreground IoU | $+6.71$ | $6.12$ | $1.94$ | $[+2.33, +11.09]$ | $+3.467$ | **$0.007077$** | $4.0$ | $0.0137$ | $+1.10$ | **$0.035385$** | **$0.035385$** |
| **Swin-Tiny vs. ResNet-50** | Foreground IoU | $+7.14$ | $9.39$ | $2.97$ | $[+0.42, +13.86]$ | $+2.405$ | **$0.039569$** | $8.0$ | $0.0488$ | $+0.76$ | $0.158274$ | $0.237411$ |
| **Std U-Net vs. Attn U-Net** | Foreground IoU | $+5.44$ | $10.09$ | $3.19$ | $[-1.78, +12.66]$ | $+1.704$ | $0.122497$ | $13.0$ | $0.1602$ | $+0.54$ | $0.367492$ | $0.716800$ |
| **Swin-MoE vs. Swin-Tiny** | Foreground IoU | $+0.35$ | $4.81$ | $1.52$ | $[-3.09, +3.79]$ | $+0.232$ | $0.821822$ | $25.0$ | $0.8457$ | $+0.07$ | $1.000000$ | $1.000000$ |
| **Std U-Net vs. ResNet-50** | Foreground IoU | $+0.43$ | $8.06$ | $2.55$ | $[-5.33, +6.19]$ | $+0.169$ | $0.869227$ | $27.0$ | $1.0000$ | $+0.05$ | $1.000000$ | $1.000000$ |
| **Swin-Tiny vs. Std U-Net** | Foreground Recall | $+17.94$ | $9.12$ | $2.88$ | $[+11.42, +24.46]$ | $+6.218$ | **$0.000155$** | $0.0$ | $0.0020$ | $+1.97$ | **$0.000932$** | **$0.001088$** |
| **Swin-Tiny vs. Attn U-Net** | Foreground Recall | $+25.19$ | $14.77$ | $4.67$ | $[+14.62, +35.75]$ | $+5.391$ | **$0.000438$** | $0.0$ | $0.0020$ | $+1.70$ | **$0.002192$** | **$0.003069$** |
| **Swin-Tiny vs. ResNet-50** | Foreground Recall | $+13.68$ | $13.16$ | $4.16$ | $[+4.27, +23.09]$ | $+3.287$ | **$0.009425$** | $6.0$ | $0.0273$ | $+1.04$ | **$0.037700$** | $0.065975$ |
| **Std U-Net vs. Swin-Tiny** | Foreground Precision | $+9.75$ | $7.79$ | $2.46$ | $[+4.18, +15.32]$ | $+3.958$ | **$0.003315$** | $6.0$ | $0.0273$ | $+1.25$ | **$0.019888$** | **$0.023204$** |
| **Std U-Net vs. ResNet-50** | Foreground Precision | $+7.57$ | $6.16$ | $1.95$ | $[+3.16, +11.98]$ | $+3.884$ | **$0.003716$** | $4.0$ | $0.0137$ | $+1.23$ | **$0.019888$** | **$0.026011$** |

---

## 7. Master Table 5: In-Domain Pretraining Test (BUSI Held-Out Test Set)

- **Comparison:** ResNet-50 U-Net (ImageNet-1K Pretrained) vs. Standard U-Net (Initialized from Scratch).
- **Cohort:** BUSI Held-Out Test Set ($N = 78$ images, 10 matched seeds).

| Endpoint Metric | Pretrained ResNet-50 U-Net | Scratch Standard U-Net | Mean Difference (pp) | Paired 95% Confidence Interval | Paired $t$ ($df=9$) | Two-Tailed Raw $p_t$ | Wilcoxon $W$ | Wilcoxon Raw $p_W$ | Holm $p_{\text{adj}}$ ($m=3$) | Biostatistical Decision |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Foreground IoU** | $59.26 \pm 4.35\%$ | $56.06 \pm 3.88\%$ | **$+3.20\text{ pp}$** | $[-1.96\text{ pp}, +8.37\text{ pp}]$ | $+1.403$ | $0.194148$ | $16.0$ | $0.2754$ | $0.388296$ | **Not Statistically Significant ($p > 0.05$)** |
| **Foreground Dice** | $74.34 \pm 3.36\%$ | $71.77 \pm 3.17\%$ | **$+2.57\text{ pp}$** | $[-1.44\text{ pp}, +6.58\text{ pp}]$ | $+1.439$ | $0.184139$ | $17.0$ | $0.3223$ | $0.388296$ | **Not Statistically Significant ($p > 0.05$)** |
| **Foreground Recall** | $76.06 \pm 5.86\%$ | $68.41 \pm 7.47\%$ | **$+7.65\text{ pp}$** | $[+1.22\text{ pp}, +14.07\text{ pp}]$ | $+2.671$ | **$0.025624$** | $8.0$ | $0.0488$ | $0.076872$ | **Significant Unadjusted, Trend Adjusted ($p < 0.10$)** |
| **Foreground Precision**| $73.03 \pm 3.93\%$ | $76.03 \pm 4.09\%$ | **$-3.00\text{ pp}$** | $[-6.42\text{ pp}, +0.42\text{ pp}]$ | $-1.974$ | $0.079744$ | $11.0$ | $0.1055$ | $0.239232$ | **Non-Significant Trend ($p < 0.10$)** |
| **Pixel Accuracy** | $95.81 \pm 0.51\%$ | $95.53 \pm 0.38\%$ | **$+0.28\text{ pp}$** | $[-0.07\text{ pp}, +0.63\text{ pp}]$ | $+1.801$ | $0.105260$ | $11.0$ | $0.1055$ | $0.239232$ | **Not Statistically Significant ($p > 0.05$)** |

---

## 8. LaTeX Manuscript Source Code

The following standard LaTeX `booktabs` tables are fully formatted and ready to be pasted directly into LaTeX manuscripts or Overleaf:

### LaTeX: Zero-Shot Generalization Benchmark (Master Table)
```latex
\begin{table*}[t]
\centering
\small
\caption{Zero-shot generalization benchmark on the external BUS-BRA ultrasound cohort ($N=1,875$ images, $1,063$ patients). All models were trained exclusively on the BUSI dataset across 10 matched random seeds. Values represent sample mean $\pm$ sample standard deviation ($ddof=1$). Parameter counts indicate total and active parameters during inference. The 2D pixel ratio indicates total predicted lesion pixels divided by ground-truth lesion pixels.}
\label{tab:zero_shot_busbra}
\begin{tabular}{llccccccc}
\toprule
\textbf{Architecture} & \textbf{Pretraining} & \textbf{Parameters (Tot/Act)} & \textbf{FG IoU (\%)} & \textbf{FG Dice (\%)} & \textbf{Recall (\%)} & \textbf{Precision (\%)} & \textbf{Pixel Acc (\%)} & \textbf{Pixel Ratio} \\
\midrule
Vanilla Swin-Tiny & ImageNet-1K & 29.29M / 29.29M & \textbf{57.50} $\pm$ 4.74 & \textbf{72.91} $\pm$ 3.98 & \textbf{76.51} $\pm$ 6.79 & 70.44 $\pm$ 6.61 & 95.06 $\pm$ 0.98 & 1.10 $\pm$ 0.18 \\
Standard-MLP Swin-MoE & ImageNet-1K & 81.15M / 46.64M & \textbf{57.85} $\pm$ 2.15 & \textbf{73.28} $\pm$ 1.73 & \textbf{77.35} $\pm$ 3.73 & 69.92 $\pm$ 4.33 & 95.11 $\pm$ 0.50 & 1.11 $\pm$ 0.11 \\
Standard U-Net & Scratch & 31.04M / 31.04M & 50.79 $\pm$ 3.68 & 67.29 $\pm$ 3.33 & 58.57 $\pm$ 6.49 & \textbf{80.19} $\pm$ 4.73 & \textbf{95.11} $\pm$ 0.28 & 0.74 $\pm$ 0.12 \\
ResNet-50 U-Net & ImageNet-1K & 32.52M / 32.52M & 50.36 $\pm$ 6.98 & 66.71 $\pm$ 6.48 & 62.84 $\pm$ 10.34 & 72.62 $\pm$ 7.13 & 94.65 $\pm$ 0.82 & 0.87 $\pm$ 0.17 \\
Attention U-Net & Scratch & 31.39M / 31.39M & 45.35 $\pm$ 8.64 & 61.93 $\pm$ 8.84 & 51.33 $\pm$ 11.32 & \textbf{81.38} $\pm$ 4.38 & 94.71 $\pm$ 0.61 & 0.64 $\pm$ 0.16 \\
\bottomrule
\end{tabular}
\end{table*}
```

### LaTeX: In-Domain vs. Zero-Shot Transfer Drop
```latex
\begin{table}[h]
\centering
\small
\caption{Cross-domain transfer performance and paired distribution shift penalties ($\Delta_{\text{transfer}} = \text{BUS-BRA Zero-Shot} - \text{BUSI In-Domain}$). Evaluated across 10 matched seeds with sample SD ($ddof=1$).}
\label{tab:transfer_penalty}
\begin{tabular}{lcccc}
\toprule
\textbf{Architecture} & \textbf{In-Domain IoU (\%)} & \textbf{Zero-Shot IoU (\%)} & \textbf{Mean Drop (pp)} & \textbf{95\% CI (pp)} \\
\midrule
Vanilla Swin-Tiny & 60.24 $\pm$ 6.11 & 57.50 $\pm$ 4.74 & \textbf{-2.74} $\pm$ 4.17 & [-5.72, +0.24] \\
Standard-MLP Swin-MoE & 60.58 $\pm$ 5.28 & 57.85 $\pm$ 2.15 & \textbf{-2.73} $\pm$ 4.89 & [-6.23, +0.76] \\
Standard U-Net & 56.06 $\pm$ 3.88 & 50.79 $\pm$ 3.68 & -5.27 $\pm$ 5.28 & [-9.05, -1.49] \\
ResNet-50 U-Net & 59.26 $\pm$ 4.35 & 50.36 $\pm$ 6.98 & -8.91 $\pm$ 8.00 & [-14.63, -3.18] \\
Attention U-Net & 56.10 $\pm$ 2.53 & 45.35 $\pm$ 8.64 & -10.75 $\pm$ 8.96 & [-17.16, -4.34] \\
\bottomrule
\end{tabular}
\end{table}
```

---
*End of Final Manuscript Numbers Reference Document.*
