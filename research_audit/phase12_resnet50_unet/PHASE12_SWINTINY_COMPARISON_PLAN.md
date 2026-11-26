# Phase 12 Swin-Tiny vs. ResNet-50 U-Net Comparison and Statistical Audit Plan

## 1. Scientific Context and Primary Research Question

In Phases 9B through 11, we observed a substantial zero-shot performance gap between Vision Transformers and Convolutional Neural Networks on external breast ultrasound lesion segmentation (BUS-BRA, $N = 1,063$):

| Architecture Family | Pretraining Status | Zero-Shot BUS-BRA FG IoU | Foreground Recall | Foreground Precision | Volume Ratio |
|---|---|---|---|---|---|
| **Vanilla Swin-Tiny** | ImageNet-1K Pretrained | **$57.50\% \pm 4.74\%$** | $60.67\% \pm 5.17\%$ | $81.76\% \pm 4.67\%$ | $0.97 \pm 0.17$ |
| **Standard-MLP Swin-MoE** | ImageNet-1K Pretrained | **$57.85\% \pm 2.15\%$** | $60.52\% \pm 2.50\%$ | $82.72\% \pm 4.02\%$ | $0.98 \pm 0.16$ |
| **Standard U-Net** | Random Initialization | **$50.79\% \pm 3.68\%$** | $58.57\% \pm 4.05\%$ | $80.19\% \pm 4.54\%$ | $0.74 \pm 0.08$ |
| **Attention U-Net** | Random Initialization | **$45.35\% \pm 10.98\%$** | $50.75\% \pm 13.91\%$ | $81.99\% \pm 4.15\%$ | $0.67 \pm 0.22$ |

### The Critical Confound: Pretraining vs. Architecture
In the above comparison, Swin-Tiny benefited from **ImageNet-1K pretraining**, whereas Standard U-Net and Attention U-Net were initialized **from scratch**. This conflates two distinct hypotheses:
1. **Architectural Inductive Bias Hypothesis**: Vision Transformers generalize better out-of-distribution because self-attention captures long-range dependencies and dynamic spatial interactions that fixed $3\times 3$ convolutions cannot represent.
2. **Pretraining Confound Hypothesis**: The generalization advantage is entirely driven by ImageNet pretraining, which provides rich invariant multi-scale feature detectors that prevent overfitting to the acoustic speckle patterns of BUSI.

**Phase 12 directly decouples these hypotheses.** By providing ResNet-50 U-Net with identical ImageNet-1K pretraining and matching parameter capacity ($32.52\text{M}$ vs $29.29\text{M}$), we isolate architecture as the sole explanatory variable.

---

## 2. Competing Hypotheses and Outcome Scenarios

$$\begin{aligned}
H_0 &: \mu_{\text{Swin-Tiny}} - \mu_{\text{ResNet50-UNet}} = 0 \quad (\text{Pretraining explains the gap; architectures perform equally}) \\
H_1 &: \mu_{\text{Swin-Tiny}} - \mu_{\text{ResNet50-UNet}} > 0 \quad (\text{Transformer inductive bias confers true architectural superiority}) \\
H_2 &: \mu_{\text{ResNet50-UNet}} - \mu_{\text{Swin-Tiny}} > 0 \quad (\text{Pretrained CNN outperforms Transformer under domain shift})
\end{aligned}$$

### Diagnostic Outcome Matrix:
| Scenario | Observed ResNet-50 FG IoU | Scientific Interpretation |
|---|---|---|
| **Scenario A (Confound Confirmed)** | $\approx 57\% - 58\%$ ($p > 0.05$ vs. Swin) | Pretraining was the decisive factor. Pretrained CNN matches pretrained Transformer. The earlier 6.7% gap was pretraining-driven. |
| **Scenario B (Transformer Superiority Confirmed)** | $\approx 51\% - 53\%$ ($p < 0.05$, Swin > ResNet) | Pretraining alone does not resolve the CNN domain-shift penalty. Transformer self-attention provides genuine cross-dataset robustness. |
| **Scenario C (CNN Superiority)** | $> 59\%$ ($p < 0.05$, ResNet > Swin) | Pretrained residual convolutions generalize better than window self-attention on ultrasound morphology. |

---

## 3. Statistical Testing and Multiplicity Protocol

### A. Paired Experimental Design
Because all 10 runs use identical seed controls (`[42, 123, 314, 777, 999, 1001, 2026, 2027, 2718, 9001]`), each ResNet-50 U-Net run is paired directly with the corresponding Swin-Tiny run trained on the exact same data split:
$$D_i = \text{Metric}_{\text{Swin-Tiny}}^{(i)} - \text{Metric}_{\text{ResNet50-UNet}}^{(i)}, \quad i = 1, \dots, 10$$

### B. Statistical Tests
1. **Paired Student's t-test**:
   $$t = \frac{\bar{D}}{s_D / \sqrt{n}}, \quad df = 9$$
2. **Wilcoxon Signed-Rank Test**:
   Non-parametric test on paired ranks to protect against non-normality in small samples ($N = 10$).
3. **Effect Size (Cohen's $d_z$)**:
   $$d_z = \frac{\bar{D}}{s_D}$$
4. **95% Confidence Interval for Mean Difference**:
   $$\bar{D} \pm t_{0.975, 9} \times \frac{s_D}{\sqrt{10}}$$

### C. Multiplicity Correction: Holm-Bonferroni Procedure
To prevent family-wise type I error inflation across the evaluation metrics:
- **Primary 7-Metric Family**: Foreground IoU, Foreground Dice, Foreground Precision, Foreground Recall, Mean IoU, Mean Dice, Pixel Accuracy.
- **Secondary Morphological Metric**: Volume Ratio ($\frac{\text{Predicted Foreground}}{\text{Ground Truth Foreground}}$) evaluated separately to examine the volume under-prediction pathology.
- Unadjusted $p$-values are sorted: $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$.
- Compared against adjusted significance thresholds $\alpha_k = \frac{0.05}{m - k + 1}$.

---

## 4. Volume Ratio and Boundary Analysis

A critical signature identified in Phase 10 and Phase 11 was the **severe volume under-prediction** of randomly initialized CNNs:
- Swin-Tiny: Volume ratio $0.97 \pm 0.17$ (almost exact foreground calibration).
- Standard U-Net: Volume ratio $0.74 \pm 0.08$ (under-segments lesions by $26\%$).
- Attention U-Net: Volume ratio $0.67 \pm 0.22$ (under-segments lesions by $33\%$).

In Phase 12, we will evaluate whether ImageNet pretraining restores the volume ratio of ResNet-50 U-Net toward $1.0\times$, or whether convolutional decoders inherently suffer from acoustic boundary attenuation.
