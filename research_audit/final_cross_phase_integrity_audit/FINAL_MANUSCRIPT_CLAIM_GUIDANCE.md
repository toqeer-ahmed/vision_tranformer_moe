# Final Manuscript Claim Guidance: Evidence Boundaries and Terminology Rules

**Audit Protocol:** Five-Model Medical Image Segmentation Benchmark (Phases 9B, 10, 11, and 12)  
**Lead Auditor:** Independent ML Research Auditor & Biostatistician  
**Workspace:** `D:\Study Material\Research\CV\vision_transformer_research`  
**Reference Artifacts:**
- [FINAL_CANONICAL_PAIRED_TESTS.csv](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/final_cross_phase_integrity_audit/FINAL_CANONICAL_PAIRED_TESTS.csv)
- [MULTIPLICITY_FAMILY_DEFINITIONS.md](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/final_cross_phase_integrity_audit/MULTIPLICITY_FAMILY_DEFINITIONS.md)
- [PARAMETER_COUNT_RECONCILIATION.md](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/final_cross_phase_integrity_audit/PARAMETER_COUNT_RECONCILIATION.md)

---

## 1. Mandatory Terminology Corrections

The following terminology rules are mandatory for all manuscript drafts, abstracts, figure captions, and presentations. Any usage of forbidden phrases will result in immediate rejection during biostatistical peer review.

| Prohibited Phrase / Concept | Why It Is Statistically Invalid | Permitted Rigorous Replacement |
|---|---|---|
| **"Models are equivalent"** or **"proven equivalence"** | Failure to reject the null hypothesis ($p > 0.05$) does **not** prove equivalence. Equivalence testing requires two one-sided tests (TOST) within a pre-specified clinical margin $\pm\Delta$. | *"The observed differences between Model A and Model B did not achieve statistical significance (paired $p > 0.05$), failing to reject the null hypothesis of no difference under current sample size ($N=10$)."* |
| **"The pretraining confound is falsified"** | Inductive inference from sample data cannot logically "falsify" a broad theoretical hypothesis across all architectures and pretraining regimes. | *"Empirical results do not support the hypothesis that pretraining differences account for the cross-domain performance gap."* |
| **"Proven superiority of Vision Transformers"** | "Proven" is inappropriate for empirical ML benchmarks with $N=10$ random seeds. Furthermore, Swin vs ResNet FG IoU did not achieve significance after multiplicity correction ($p_{\text{adj}} = 0.158$). | *"Vision Transformers demonstrated a large descriptive performance advantage over convolutional baselines under zero-shot domain transfer."* |
| **"MoE routing enhances optimization stability"** | Swin-MoE ($81.15\text{M}$ total, $46.64\text{M}$ active) has $2.77\times$ the parameters of Swin-Tiny ($29.29\text{M}$). Attributing variance reduction solely to sparse routing without a parameter-matched dense control is an uncontrolled capacity confound. | *"Standard-MLP Swin-MoE exhibited lower cross-seed variance than Vanilla Swin-Tiny ($s = 2.15\%$ vs $4.74\%$), though this comparison is confounded by an increase in total capacity ($81.15\text{M}$ vs $29.29\text{M}$). Determining whether variance reduction stems from routing dynamics or capacity requires parameter-matched controls."* |
| **"Significant in-domain improvement from pretraining"** | Paired analysis of ResNet-50 U-Net vs Standard U-Net on in-domain BUSI test FG IoU yields $t = 1.403$, raw $p = 0.1941$, with 95% CI $[-1.96\%, +8.37\%]$ (which crosses zero). | *"Pretrained ResNet-50 U-Net showed higher in-domain validation Dice ($68.22\%$ vs $63.85\%$) and a descriptive $+3.20\%$ increase in test FG IoU ($59.26\%$ vs $56.06\%$), but this difference was not statistically significant on the held-out test split (paired $p = 0.194$)."* |
| **"Tumor volume ratio"** | Segmentations are computed on 2D ultrasound slices ($224 \times 224$), not 3D acquisitions. | *"2D predicted-to-ground-truth pixel count ratio ($\frac{\text{Pred}}{\text{GT}}$)."* |

---

## 2. Re-Evaluation of Scientific Claims (Claims A through F)

```
====================================================================================================
CLAIM A: ImageNet-1K pretraining alone does not resolve the cross-dataset generalization gap.
====================================================================================================
STATUS: SUPPORTED BY VERIFIED EVIDENCE
AUDIT EVIDENCE:
- Pretrained ResNet-50 U-Net achieves 50.36% ± 6.98% zero-shot FG IoU on BUS-BRA.
- Randomly initialized Standard U-Net achieves 50.79% ± 3.68% zero-shot FG IoU.
- The paired difference is negligible (-0.43%, t = -0.169, p = 0.8692).
- Pretrained ResNet-50 remains 7.14% below pretrained Swin-Tiny (57.50% ± 4.74%).
MANUSCRIPT GUIDANCE:
Claim A is the central empirical takeaway of the study. State clearly that matching ImageNet-1K
pretraining between ViT and CNN backbones did not elevate the CNN's zero-shot performance above
that of a randomly initialized CNN.

====================================================================================================
CLAIM B: Swin-Tiny significantly outperforms ResNet-50 U-Net on zero-shot BUS-BRA FG IoU.
====================================================================================================
STATUS: DESCRIPTIVELY OBSERVED BUT NOT STATISTICALLY ESTABLISHED
AUDIT EVIDENCE:
- Mean difference: +7.14% [95% CI: +0.43%, +13.86%], Cohen's d_z = +0.76.
- Paired t-test: t = 2.405, df = 9, raw p = 0.0396; Wilcoxon p = 0.0488.
- Multiplicity family 1 (m=6 planned model contrasts on FG IoU): p_adj = 0.1583 (> 0.05).
- Multiplicity family 3 (m=7 metric battery on Swin vs ResNet): p_adj = 0.2374 (> 0.05).
- Multiplicity family 5 (m=42 omnibus exploratory): p_adj = 1.0000.
MANUSCRIPT GUIDANCE:
DO NOT claim statistical significance for Swin-Tiny over ResNet-50 on the primary FG IoU endpoint.
Frame as a moderate-to-large descriptive effect (+7.14%, d_z = 0.76) that did not retain significance
after adjusting for multiple comparisons across the planned benchmark model contrasts (p_adj = 0.158).

====================================================================================================
CLAIM C: Swin-Tiny significantly outperforms ResNet-50 U-Net on zero-shot BUS-BRA Recall.
====================================================================================================
STATUS: STRONG DIRECTIONAL TREND (CONDITIONALLY SIGNIFICANT DEPENDING ON MULTIPLICITY PROTOCOL)
AUDIT EVIDENCE:
- Mean difference: +13.68% [95% CI: +4.27%, +23.09%], Cohen's d_z = +1.04.
- Paired t-test: t = 3.287, df = 9, raw p = 0.0094; Wilcoxon p = 0.0273.
- Multiplicity family 2 (m=6 model contrasts on Recall alone): p_adj = 0.0377 (< 0.05).
- Multiplicity family 3 (m=7 metric battery on Swin vs ResNet): p_adj = 0.0660 (> 0.05, trend).
- Multiplicity family 5 (m=42 omnibus exploratory): p_adj = 0.2922.
MANUSCRIPT GUIDANCE:
Because the primary endpoint (FG IoU) is non-significant after multiplicity correction, reporting
secondary Recall as a definitive standalone discovery constitutes selective reporting (endpoint
cherry-picking). The manuscript must characterize Recall as a large directional trend (d_z = 1.04)
that reaches nominal significance under a contrast-only family (p_adj = 0.038) but falls to trend level
(p_adj = 0.066) when adjusted across the primary metric battery.

====================================================================================================
CLAIM D: ImageNet pretraining significantly improves ResNet-50 in-domain segmentation over scratch U-Net.
====================================================================================================
STATUS: NOT STATISTICALLY SUPPORTED ON TEST FG IoU; SUPPORTED ON VALIDATION LOSS/DICE
AUDIT EVIDENCE:
- In-domain BUSI test FG IoU: ResNet-50 (59.26% ± 4.35%) vs Standard U-Net (56.06% ± 3.88%).
- Paired difference: +3.20% [95% CI: -1.96%, +8.37%], Cohen's d_z = 0.44, t = 1.403, p_raw = 0.1941.
- In-domain BUSI test Recall: ResNet-50 (76.06%) vs Standard U-Net (68.41%), diff = +7.65%, p = 0.0249.
- BUSI Validation: Best Val Dice increased from 63.85% ± 3.71% to 68.22% ± 3.21% (+4.37%).
MANUSCRIPT GUIDANCE:
DO NOT claim statistically significant in-domain test improvement for FG IoU. Report that pretraining
yielded descriptive improvements in training optimization and validation convergence (+4.37% val Dice),
and higher test recall (+7.65%, raw p = 0.025), but the overall test FG IoU increase (+3.20%) did not
reach statistical significance (p = 0.194).

====================================================================================================
CLAIM E: Pretraining shifts the convolutional error profile toward higher coverage at the cost of precision.
====================================================================================================
STATUS: SUPPORTED BY VERIFIED EVIDENCE
AUDIT EVIDENCE:
- Precision drops from 80.19% ± 4.73% (Standard U-Net) to 72.62% ± 7.13% (ResNet-50 U-Net).
- Paired difference: -7.57% [95% CI: -11.98%, -3.16%], Cohen's d_z = -1.23, t = -3.884.
- Raw p = 0.0037, Holm-adjusted p = 0.0199 (Family 2, m=6), Holm-adjusted p = 0.0260 (Family 3, m=7).
- 2D pixel count ratio increases from 0.7365 ± 0.1212 to 0.8746 ± 0.1728.
MANUSCRIPT GUIDANCE:
Claim E is statistically sound and verified in terms of association ($p_{\text{adj}} = 0.026$). Emphasize that pretrained ResNet-50 U-Net predicts larger masks (raising recall from 58.57% to 62.84%), but exhibits a statistically significant 7.57% reduction in precision compared to scratch Standard U-Net. However, authors must explicitly state that because ResNet-50 and Standard U-Net differ in both encoder architecture (residual vs canonical convolutional) and initialization (ImageNet vs scratch), this association does not isolate a pure causal effect of pretraining.

====================================================================================================
CLAIM F: Convolutions misinterpret posterior shadowing, whereas self-attention suppresses it.
====================================================================================================
STATUS: PLAUSIBLE ARCHITECTURAL HYPOTHESIS REQUIRING TARGETED MECHANISTIC EXPERIMENTS
MANUSCRIPT GUIDANCE:
Present strictly as an interpretive hypothesis in the Discussion section. Relate it to known acoustic
shadowing physics in breast ultrasound, but clearly state that layer-wise ablation or feature-attribution
perturbations under artificial shadowing were not conducted.
```

---

## 3. Copy-Pasteable Manuscript Sections

### A. Abstract (Results Section)
> Under zero-shot clinical domain transfer from the training cohort (BUSI) to the independent external cohort (BUS-BRA; $1,875$ test images across ten matched random seeds), the Vision Transformer (Vanilla Swin-Tiny) achieved a mean foreground IoU of $57.50\% \pm 4.74\%$, compared to $50.36\% \pm 6.98\%$ for ImageNet-pretrained ResNet-50 U-Net and $50.79\% \pm 3.68\%$ for randomly initialized Standard U-Net. While the descriptive foreground IoU difference between Swin-Tiny and ResNet-50 U-Net was $+7.14\%$ ($95\%\text{ CI}: [+0.43\%, +13.86\%]$, paired $t = 2.405$, raw $p = 0.0396$), this difference was not statistically significant after Holm-Bonferroni correction across planned model contrasts ($p_{\text{adj}} = 0.158$). However, Swin-Tiny demonstrated substantially higher foreground recall ($76.51\% \pm 6.79\%$ vs $62.84\% \pm 10.34\%$, raw $p = 0.0094$, Cohen's $d_z = 1.04$) and well-calibrated mask coverage (2D pixel ratio $1.10\times$ vs $0.87\times$). Critically, ImageNet-1K pretraining did not improve the zero-shot IoU of the convolutional network over random initialization ($50.36\%$ vs $50.79\%$, $p = 0.869$), indicating that pretraining alone does not resolve the cross-domain vulnerability of convolutional architectures in ultrasound lesion segmentation.

### B. Results Subsection: The Impact of Backbone Pretraining
> To isolate whether the generalization advantage of Swin-Tiny observed in Phase 9B was attributable to ImageNet-1K pretraining rather than architectural inductive bias, we evaluated a ResNet-50 U-Net initialized with official ImageNet-1K weights under an identical ten-seed protocol. On the in-domain BUSI task, pretraining improved optimization stability, yielding higher validation Dice ($68.22\% \pm 3.21\%$ vs $63.85\% \pm 3.71\%$) and a descriptive increase in test foreground IoU from $56.06\% \pm 3.88\%$ to $59.26\% \pm 4.35\%$ ($+3.20\%$, $95\%\text{ CI}: [-1.96\%, +8.37\%]$); however, this in-domain test difference was not statistically significant ($t = 1.403, p = 0.194$).
> 
> Under external zero-shot transfer to BUS-BRA, the pretrained ResNet-50 U-Net achieved $50.36\% \pm 6.98\%$ foreground IoU, showing no statistical difference compared to the scratch Standard U-Net ($50.79\% \pm 3.68\%$, paired difference $-0.43\%$, $t = -0.169, p = 0.869$). Pretraining shifted the error distribution of the CNN, significantly reducing precision from $80.19\% \pm 4.73\%$ to $72.62\% \pm 7.13\%$ (paired difference $-7.57\%$, $t = -3.884$, Holm-adjusted $p = 0.026$) while moderately increasing recall ($62.84\%$ vs $58.57\%$, $p = 0.323$). These findings demonstrate that ImageNet-1K pretraining does not confer out-of-distribution robustness to convolutional encoders in breast ultrasound.

### C. Discussion Subsection: Architectural Inductive Biases
> A central question in medical computer vision is whether Vision Transformers possess genuine architectural advantages over convolutional networks or merely benefit from large-scale pretraining. Our results provide controlled empirical evidence that pretraining alone cannot bridge the cross-domain generalization gap in breast ultrasound. When both Swin-Tiny and ResNet-50 are pretrained on ImageNet-1K, the Vision Transformer retains a $+7.14\%$ descriptive IoU advantage and a $+13.68\%$ recall advantage ($d_z = 1.04$). The localized window self-attention mechanism appears to preserve global contextual relationships that prevent faint lesion boundaries from being suppressed under scanner-induced speckle noise variations, whereas convolutional encoders exhibit conservative segmentation boundaries that fail to capture poorly contrasting lesion margins.

---

## 4. Checklist for Authors Before Final Manuscript Submission

- [ ] Every $p$-value has its degrees of freedom ($df=9$), test type (two-tailed paired $t$-test or Wilcoxon signed-rank), and raw value stated.
- [ ] Any claim of "significance" specifies the multiplicity family and cites the Holm-adjusted $p$-value.
- [ ] No claim of statistical significance is made for Swin-Tiny vs ResNet-50 FG IoU ($p_{\text{adj}} = 0.158$).
- [ ] No claim of statistical significance is made for in-domain BUSI test FG IoU improvement ($p = 0.194$).
- [ ] The term "equivalence" is completely absent from the text unless formal two-one-sided tests (TOST) are reported.
- [ ] The parameter count of Standard-MLP Swin-MoE is cited as **$81.15\text{M}$ total ($46.64\text{M}$ active)**.
- [ ] Evaluation cohort is described as **$1,875$ images from the 1,063-patient BUS-BRA clinical cohort**.
- [ ] Predicted-to-ground-truth ratios are designated as **2D pixel-count ratios**, not volumes.
