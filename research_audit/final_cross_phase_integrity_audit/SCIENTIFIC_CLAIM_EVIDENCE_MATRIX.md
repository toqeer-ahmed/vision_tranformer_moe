# Scientific Claim Evidence Matrix: Pre-Manuscript Integrity Review

## 1. Executive Summary

This document evaluates the six core scientific hypotheses and claims of the five-model breast ultrasound segmentation study against the verified statistical evidence across all 10 matched seeds.

Each claim is formally classified under one of four standards of evidence:
1. **Supported by verified evidence** (Empirically verified with statistical significance at family-wise $\alpha = 0.05$ or direct experimental control).
2. **Descriptively observed but not statistically established** (Observed in sample means/effect sizes, but non-significant after planned multiplicity correction).
3. **Plausible hypothesis requiring further experiments** (Theoretically sound mechanism consistent with data, but not directly tested by experimental design).
4. **Unsupported or contradicted** (Empirically refuted by experimental data).

---

## 2. Evaluation Matrix of Scientific Claims

| Identifier | Scientific Claim | Audit Classification | Statistical / Empirical Evidence | Manuscript Writing Guidance |
|---|---|---|---|---|
| **Claim A** | *ImageNet pretraining alone does not explain the observed Swin-Tiny versus CNN zero-shot performance gap.* | **Supported by verified evidence** | ResNet-50 U-Net was provided with identical ImageNet-1K pretraining and matched capacity ($32.52\text{M}$ vs $29.29\text{M}$). Its zero-shot FG IoU ($50.36\% \pm 6.98\%$) remained $7.14\%$ below Swin-Tiny ($57.50\% \pm 4.74\%$) and was statistically identical to from-scratch Standard U-Net ($50.79\% \pm 3.68\%$, $p = 0.869$). Pretraining failed to elevate the CNN out-of-distribution. | **Approved for manuscript**. State clearly that pretraining alone does not resolve the cross-dataset performance deficit of CNNs. |
| **Claim B** | *Swin-Tiny significantly outperforms pretrained ResNet-50 U-Net on BUS-BRA foreground IoU.* | **Descriptively observed but not statistically established** | Descriptively, Swin-Tiny exceeds ResNet-50 by $+7.14\%$ in mean FG IoU (Cohen's $d_z = 0.761$) with raw $p = 0.0396$. However, after planned Holm-Bonferroni correction across the 7-metric primary family, **$p_{\text{adj}} = 0.2374 > 0.05$**. | **Correction required**. Do NOT claim statistical significance. Describe as a moderate-to-large descriptive difference ($+7.14\%$, $d_z = 0.76$) that did not achieve significance after multiplicity correction ($p_{\text{adj}} = 0.24$). |
| **Claim C** | *Swin-Tiny significantly outperforms pretrained ResNet-50 U-Net on recall.* | **Descriptively observed but not statistically established (Strong Trend)** | Swin-Tiny achieves a $+13.68\%$ higher mean recall ($76.51\%$ vs $62.84\%$, $d_z = 1.039$) with raw $p = 0.0094$. Under Holm-Bonferroni correction, **$p_{\text{adj}} = 0.0660$**, falling just short of the $\alpha = 0.05$ threshold. | **Correction required**. Characterize as a large descriptive recall difference ($+13.68\%$, $d_z = 1.04$) that represents a strong statistical trend ($p_{\text{adj}} = 0.066$) requiring larger sample cohorts to confirm. |
| **Claim D** | *ImageNet pretraining significantly improves in-domain ResNet-50 U-Net test FG IoU.* | **Not statistically significant in this experiment** | The observed in-domain FG IoU improvement associated with the pretrained ResNet-50 U-Net was not statistically significant in this experiment ($59.26\% \pm 4.35\%$ vs $56.06\% \pm 3.88\%$, $+3.20\text{ pp}$, paired $t = 1.403, p = 0.1941, 95\%\text{ CI}: [-1.96\text{ pp}, +8.37\text{ pp}]$, Holm $p_{\text{adj}} = 0.388$). Pretrained ResNet-50 U-Net showed higher best validation Dice ($68.22\%$ vs $63.85\%$) and higher test recall ($76.06\%$ vs $68.41\%$, raw $p = 0.025$), but the overall test FG IoU difference does not reject the null hypothesis. Non-significance does not prove that pretraining provides no benefit. | **Correction required**. State that the observed in-domain FG IoU improvement associated with the pretrained ResNet-50 U-Net was not statistically significant in this experiment. |
| **Claim E** | *ImageNet pretraining causes a precision reduction relative to scratch Standard U-Net.* | **Descriptive association; does not isolate causality** | Standard U-Net precision is $80.19\% \pm 4.73\%$, whereas ResNet-50 U-Net is $72.62\% \pm 7.13\%$ (paired difference $-7.57\text{ pp}$, $t = 3.884$, raw $p = 0.0037$, Holm $p_{\text{adj}} = 0.0260$). While this precision difference is statistically significant, the comparison conflates encoder architecture (residual vs canonical convolutional) with pretraining initialization, and thus does not isolate the causal effect of pretraining. | **Correction required**. Report that pretrained ResNet-50 U-Net achieved lower precision than scratch Standard U-Net, but explicitly acknowledge that architectural differences prevent claiming a causal effect of pretraining. |
| **Claim F** | *CNN convolutional filters misinterpret acoustic shadowing, while Swin attention suppresses speckle.* | **Plausible hypothesis requiring further experiments** | This mechanical explanation is consistent with observed false positive shifts and volume ratios, but was not directly tested via feature-attribution maps (e.g., Grad-CAM, attention rollout) or acoustic noise perturbation studies. | **Qualify in manuscript**. Present as an interpretative architectural hypothesis in the Discussion section rather than an experimentally established empirical finding. |

---

## 3. Strict Pre-Publication Writing Rules

1. **Multiplicity Discipline**: Never cite raw $p$-values as evidence of significance when the Holm-Bonferroni adjusted $p$-value exceeds $0.05$.
2. **Capacity Confound Discipline**: Always state that Standard-MLP Swin-MoE has $81.15\text{M}$ total parameters ($2.77\times$ Vanilla Swin). The cross-seed variance reduction ($s = 2.15\%$ vs $4.74\%$) cannot be separated from capacity scaling without a parameter-matched dense control.
3. **Volume Ratio Nomenclature**: Always refer to volume ratios as **2D pixel-count ratios** ($\frac{\text{Predicted Pixels}}{\text{Ground Truth Pixels}}$), never as clinical 3D volumetric metrics.
4. **Cohort Specification**: Always specify that the external evaluation cohort consists of **$1,875$ images from the BUS-BRA clinical repository** ($94,080,000$ pixels evaluated per seed).
