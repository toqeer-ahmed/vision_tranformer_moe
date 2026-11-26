# Final Cross-Phase Integrity Audit: Five-Model Breast Ultrasound Segmentation Benchmark (Phases 9B, 10, 11, and 12)

**Audit Scope**: Phases 9B (Vanilla Swin-Tiny, Standard-MLP Swin-MoE), 10 (Standard U-Net), 11 (Attention U-Net), and 12 (ImageNet-Pretrained ResNet-50 U-Net)  
**Lead Auditor**: Independent ML Research Auditor & Biostatistician  
**Workspace**: `D:\Study Material\Research\CV\vision_transformer_research`  
**Audit Status**: Complete, Independently Verified, Canonical Data Harmonized  

---

## 1. Executive Summary

This independent cross-phase integrity audit evaluates the five-model breast ultrasound segmentation study prior to manuscript preparation. The primary research goal across these phases was to evaluate whether Vision Transformers (Swin-Tiny and Swin-MoE) possess genuine cross-dataset zero-shot generalization advantages over convolutional architectures (Standard U-Net, Attention U-Net, and ResNet-50 U-Net) when transferring from the held-in **BUSI** dataset to the external **BUS-BRA** dataset.

All five models were trained across **ten identical matched seeds** (`[42, 123, 314, 777, 999, 1001, 2026, 2027, 2718, 9001]`) with identical AdamW optimization, learning rate ($1e-4$), weight decay ($0.05$), Cosine Annealing, and Focal Tversky Loss.

### Primary Audit Conclusions:
1. **Source Data Integrity Confirmed**: All five models share the exact same ten random seeds, identical 80/10/10 BUSI training/validation/test partitions, and an identical external evaluation cohort ($1,875$ paired images; $94,080,000$ pixels evaluated per seed; $8,146,649$ ground truth foreground pixels). Zero-shot data separation was strictly maintained: BUS-BRA was never accessed for training, validation, checkpoint selection, or threshold tuning.
2. **Swin-MoE Parameter Count Reconciled**: Standard-MLP Swin-MoE contains **$81.15\text{ M}$ total parameters** ($81,148,989$ trainable) and **$46.64\text{ M}$ active parameters per forward pass**. The $33.64\text{ M}$ figure cited in Phase 12 documentation was an erroneous transcription of a single-stage hypothetical model.
3. **Pretraining Confound Falsified**: Pretrained ResNet-50 U-Net achieves **$50.36\% \pm 6.98\%$** zero-shot Foreground IoU, which remains **$7.14\%$ below** Vanilla Swin-Tiny ($57.50\% \pm 4.74\%$) and is statistically indistinguishable from randomly initialized Standard U-Net ($50.79\% \pm 3.68\%$, $p = 0.869$). Pretraining alone does not resolve the convolutional domain-shift penalty.
4. **Standard Deviations Reconciled**: All standard deviation discrepancies in draft reports have been traced and corrected to their true sample standard deviations ($N-1=9$ ddof): Swin-MoE Recall SD is strictly **$3.73\%$**, Precision SD is **$4.33\%$**; Standard U-Net Recall SD is strictly **$6.49\%$**, Precision SD is **$4.73\%$**.
5. **Multiplicity Discipline Established**: While Swin-Tiny demonstrates large descriptive advantages over ResNet-50 U-Net in Foreground Recall ($+13.68\%$, Cohen's $d_z = 1.04$) and FG IoU ($+7.14\%$, $d_z = 0.76$), these comparisons yield Holm-adjusted $p$-values of **$p_{\text{adj}} = 0.0660$** (trend) and **$p_{\text{adj}} = 0.2374$** (non-significant) across the 7-metric primary family. They must not be claimed as statistically established at $\alpha = 0.05$.

---

## 2. Canonical Five-Model Benchmark Master Table

The table below presents the verified benchmark results calculated directly from raw per-seed confusion matrices and in-domain evaluation records:

| Model Architecture | Model Family | Backbone Pretraining | Total Parameters (Active) | In-Domain BUSI FG IoU (%) | Zero-Shot BUS-BRA FG IoU (%) | Zero-Shot FG Dice (%) | Zero-Shot FG Recall (%) | Zero-Shot FG Precision (%) | Pixel Accuracy (%) | 2D Volume Ratio ($\frac{\text{Pred}}{\text{GT}}$) |
|---|---|---|---|---|---|---|---|---|---|---|
| **Standard-MLP Swin-MoE** | Sparsely-Gated ViT | ImageNet-1K | 81.15M (46.64M) | $62.0 \pm 2.8$ | **$57.85 \pm 2.15$** | $73.28 \pm 1.73$ | **$77.35 \pm 3.73$** | $69.92 \pm 4.33$ | $95.11 \pm 0.50$ | $1.1122 \pm 0.1138$ |
| **Vanilla Swin-Tiny** | Hierarchical ViT | ImageNet-1K | 29.29M (29.29M) | $61.5 \pm 3.5$ | **$57.50 \pm 4.74$** | $72.91 \pm 3.98$ | **$76.51 \pm 6.79$** | $70.44 \pm 6.61$ | $95.06 \pm 0.98$ | $1.1005 \pm 0.1805$ |
| **Standard U-Net** | Canonical CNN | Random Init | 31.04M (31.04M) | $56.06 \pm 3.88$ | **$50.79 \pm 3.68$** | $67.29 \pm 3.33$ | $58.57 \pm 6.49$ | **$80.19 \pm 4.73$** | $95.11 \pm 0.28$ | $0.7365 \pm 0.1212$ |
| **ResNet-50 U-Net** | Residual CNN | ImageNet-1K | 32.52M (32.52M) | **$59.26 \pm 4.35$** | **$50.36 \pm 6.98$** | $66.71 \pm 6.48$ | $62.84 \pm 10.34$ | $72.62 \pm 7.13$ | $94.65 \pm 0.82$ | $0.8746 \pm 0.1728$ |
| **Attention U-Net** | Attention-Gated CNN | Random Init | 31.39M (31.39M) | $56.10 \pm 2.53$ | **$45.35 \pm 8.64$** | $61.93 \pm 8.84$ | $51.33 \pm 11.32$ | **$81.38 \pm 4.38$** | $94.71 \pm 0.61$ | $0.6377 \pm 0.1624$ |

*All zero-shot metrics evaluated on external BUS-BRA ($1,875$ images; $94,080,000$ pixels per seed). Results reported as $\text{Mean} \pm \text{Sample Standard Deviation}$ ($N=10$).*

---

## 3. Independent Statistical Audit of All Six Planned Paired Comparisons

All paired differences are evaluated across the 10 matched seeds ($df = 9$) using paired Student's $t$-tests and Wilcoxon signed-rank tests. Multiplicity is controlled using the predeclared Holm-Bonferroni procedure across the 7 primary evaluation metrics:

### Comparison 1: Vanilla Swin-Tiny vs. ResNet-50 U-Net
*Testing whether pretrained ViT maintains superiority over pretrained CNN under equalized pretraining:*
- **Foreground Recall**: Diff = $+13.68\%$, $95\%\text{ CI}: [+4.26\%, +23.09\%]$, Cohen's $d_z = 1.039$, $t = 3.287$, Raw $p = 0.0094$, **Holm $p = 0.0660$** (Marginally significant trend).
- **Foreground IoU**: Diff = $+7.14\%$, $95\%\text{ CI}: [+0.42\%, +13.86\%]$, Cohen's $d_z = 0.761$, $t = 2.405$, Raw $p = 0.0396$, **Holm $p = 0.2374$** (Non-significant after correction).
- **Foreground Dice**: Diff = $+6.19\%$, $95\%\text{ CI}: [+0.17\%, +12.22\%]$, Cohen's $d_z = 0.735$, $t = 2.324$, Raw $p = 0.0452$, **Holm $p = 0.2374$** (Non-significant after correction).
- **Macro mIoU**: Diff = $+3.75\%$, $t = 2.231$, Raw $p = 0.0526$, Holm $p = 0.2374$.
- **Macro mDice**: Diff = $+3.19\%$, $t = 2.223$, Raw $p = 0.0533$, Holm $p = 0.2374$.
- **Pixel Accuracy**: Diff = $+0.41\%$, $t = 0.933$, Raw $p = 0.3751$, Holm $p = 0.7501$.
- **Foreground Precision**: Diff = $-2.18\%$, $t = -0.696$, Raw $p = 0.5038$, Holm $p = 0.7501$.

### Comparison 2: Standard U-Net vs. ResNet-50 U-Net
*Testing the isolated effect of ImageNet pretraining within convolutional U-Net:*
- **Foreground Precision**: Diff = $+7.57\%$, $95\%\text{ CI}: [+3.16\%, +11.98\%]$, Cohen's $d_z = 1.228$, $t = 3.884$, Raw $p = 0.0037$, **Holm $p = 0.0260$ (Statistically Significant!)**. Standard U-Net has significantly higher precision than ResNet-50 U-Net.
- **Pixel Accuracy**: Diff = $+0.46\%$, $t = 1.956$, Raw $p = 0.0822$, Holm $p = 0.4930$.
- **Foreground Recall**: Diff = $-4.27\%$, $t = -1.047$, Raw $p = 0.3225$, Holm $p = 1.0000$.
- **Foreground IoU**: Diff = $+0.43\%$, $t = 0.169$, Raw $p = 0.8692$, Holm $p = 1.0000$ (No difference).
- **Foreground Dice**: Diff = $+0.58\%$, $t = 0.245$, Raw $p = 0.8121$, Holm $p = 1.0000$ (No difference).

### Comparison 3: Vanilla Swin-Tiny vs. Standard U-Net
*Evaluating Pretrained ViT vs. Scratch Canonical CNN:*
- **Foreground Recall**: Diff = $+17.94\%$, Cohen's $d_z = 1.966$, $t = 6.218$, Raw $p = 0.000155$, **Holm $p = 0.001088$ (Statistically Significant!)**.
- **Foreground Precision**: Diff = $-9.75\%$, Cohen's $d_z = -1.252$, $t = -3.958$, Raw $p = 0.003315$, **Holm $p = 0.019888$ (Statistically Significant!)**.
- **Foreground IoU**: Diff = $+6.71\%$, Cohen's $d_z = 1.097$, $t = 3.467$, Raw $p = 0.007077$, **Holm $p = 0.035385$ (Statistically Significant!)**.
- **Foreground Dice**: Diff = $+5.62\%$, Cohen's $d_z = 1.060$, $t = 3.352$, Raw $p = 0.008496$, **Holm $p = 0.035385$ (Statistically Significant!)**.
- **Macro mDice**: Diff = $+2.77\%$, $t = 3.053$, Raw $p = 0.013734$, **Holm $p = 0.041201$ (Statistically Significant!)**.
- **Macro mIoU**: Diff = $+3.29\%$, $t = 2.982$, Raw $p = 0.015395$, **Holm $p = 0.041201$ (Statistically Significant!)**.

### Comparison 4: Vanilla Swin-Tiny vs. Attention U-Net
- **Foreground Recall**: Diff = $+25.19\%$, $t = 5.390$, Raw $p = 0.000438$, **Holm $p = 0.003069$ (Statistically Significant!)**.
- **Foreground IoU**: Diff = $+12.15\%$, $t = 5.052$, Raw $p = 0.000688$, **Holm $p = 0.004131$ (Statistically Significant!)**.
- **Foreground Dice**: Diff = $+10.98\%$, $t = 4.496$, Raw $p = 0.001497$, **Holm $p = 0.005892$ (Statistically Significant!)**.
- **Foreground Precision**: Diff = $-10.94\%$, $t = -3.457$, Raw $p = 0.007197$, **Holm $p = 0.014395$ (Statistically Significant!)**.

### Comparison 5: Standard U-Net vs. Attention U-Net
- **All 7 metrics are non-significant after Holm-Bonferroni correction**:
  - Foreground IoU: Diff = $+5.44\%$, $t = 1.704$, Raw $p = 0.1225$, **Holm $p = 0.7168$**.
  - Foreground Recall: Diff = $+7.24\%$, $t = 1.611$, Raw $p = 0.1416$, **Holm $p = 0.7168$**.
  - Foreground Precision: Diff = $-1.19\%$, $t = -0.618$, Raw $p = 0.5518$, **Holm $p = 0.7168$**.
- *Audit Verdict*: The empirical data fail to demonstrate any statistically significant benefit from skip-connection attention gates on breast ultrasound.

### Comparison 6: Standard-MLP Swin-MoE vs. Vanilla Swin-Tiny
- **All 7 metrics show no statistically significant shift in central tendency**:
  - Foreground IoU: Diff = $+0.35\%$, $t = 0.232$, Raw $p = 0.8218$, **Holm $p = 1.0000$**.
  - Foreground Recall: Diff = $+0.83\%$, $t = 0.428$, Raw $p = 0.6785$, **Holm $p = 1.0000$**.
- *Audit Verdict*: MoE's primary empirical effect is **cross-seed variance reduction** ($s$ reduced from $4.74\%$ to $2.15\%$, a $54.6\%$ variance reduction), rather than a statistically significant increase in mean IoU.

---

## 4. Volume Ratio Formulation Audit

- **Mean Per-Seed Ratio**: $\frac{1}{N} \sum_{i=1}^N \frac{\text{Pred}_i}{\text{GT}_i}$
- **Pooled Ratio**: $\frac{\sum \text{Pred}_i}{\sum \text{GT}_i}$
- Because every seed was evaluated on the identical unseen test set ($\text{GT} = 8,146,649$ pixels across all runs), the mean per-seed ratio and pooled ratio are mathematically identical:
  - Vanilla Swin-Tiny: **$1.1005 \pm 0.1805$** (Pooled: $1.1005$)
  - Standard-MLP Swin-MoE: **$1.1122 \pm 0.1138$** (Pooled: $1.1122$)
  - ResNet-50 U-Net: **$0.8746 \pm 0.1728$** (Pooled: $0.8746$)
  - Standard U-Net: **$0.7365 \pm 0.1212$** (Pooled: $0.7365$)
  - Attention U-Net: **$0.6377 \pm 0.1624$** (Pooled: $0.6377$)
- **Resolution of $0.97\times$ Citation**: The $0.97\times$ citation in exploratory planning was an unverified inverted ratio ($\text{GT}/\text{Pred} = 81.46\text{M} / 89.65\text{M} = 0.908$) or per-image average. The true dataset-level ratio is strictly **$1.10\times$**.
- **Reporting Directive**: Label these values strictly as **2D pixel-count ratios** ($\frac{\text{Predicted Pixels}}{\text{Ground Truth Pixels}}$), never as clinical 3D volumetric metrics.

---

## 5. Audit of Scientific Claims

| Claim | Audit Classification | Formal Evidence |
|---|---|---|
| **A. Pretraining alone does not explain the ViT vs. CNN gap** | **Supported by verified evidence** | ResNet-50 U-Net ($50.36\% \pm 6.98\%$) remains $7.14\%$ below Swin-Tiny ($57.50\% \pm 4.74\%$) despite identical ImageNet pretraining, and equals scratch U-Net ($50.79\% \pm 3.68\%$). |
| **B. Swin-Tiny significantly outperforms ResNet-50 U-Net on FG IoU** | **Descriptively observed but not statistically established** | Raw $p = 0.0396$, but Holm-adjusted $p = 0.2374 > 0.05$. Do not claim significance. |
| **C. Swin-Tiny significantly outperforms ResNet-50 U-Net on recall** | **Descriptively observed but not statistically established (Trend)** | Raw $p = 0.0094$, Holm-adjusted $p = 0.0660 > 0.05$. Strong trend ($d_z = 1.04$), but non-significant at $\alpha = 0.05$. |
| **D. Pretraining improves in-domain ResNet-50 performance** | **Supported by verified evidence** | In-domain BUSI test FG IoU reached $59.26\% \pm 4.35\%$ ($+3.20\%$ over scratch U-Net). |
| **E. Pretraining causes precision reduction relative to scratch U-Net** | **Supported by verified evidence** | Precision dropped from $80.19\%$ to $72.62\%$ ($t = 3.884$, Holm $p = 0.0260 < 0.05$). |
| **F. CNN convolutions misinterpret acoustic shadowing vs. ViT speckle suppression** | **Plausible hypothesis requiring further experiments** | Interpretative hypothesis consistent with precision/recall shifts, but not directly tested by feature-attribution perturbation experiments. |

---

## 6. Record of Discrepancies and Corrections Made

1. **Swin-MoE Parameter Count**: Corrected from $33.64\text{M}$ ($28.32\text{M}$) to **$81.15\text{M}$ total ($46.64\text{M}$ active)**.
2. **Swin-MoE Recall & Precision SDs**: Corrected from $4.49\%$ and $6.84\%$ to **$3.73\%$** and **$4.33\%$**.
3. **Standard U-Net Recall & Precision SDs**: Corrected from $4.05\%$ and $4.54\%$ to **$6.49\%$** and **$4.73\%$**.
4. **Swin-Tiny Volume Ratio**: Corrected from $0.97\times$ to **$1.1005 \pm 0.1805$** ($1.10\times$).
5. **BUS-BRA Evaluation Cohort**: Clarified from $1,063$ images to **$1,875$ images from the 1,063-patient BUS-BRA clinical repository** ($94,080,000$ pixels evaluated per seed).

---

## 7. Manuscript Readiness Verdict & Remaining Limitations

### Verdict: **READY FOR MANUSCRIPT PREPARATION**
The study is mathematically and statistically sound. All per-seed records, confusion matrices, and code artifacts are reproducible and harmonized.

### Remaining Limitations to Note in Manuscript:
1. **Sample Size of Paired Runs ($N = 10$)**: While 10 matched seeds provide adequate power to detect large effects (such as Swin-Tiny vs. Standard U-Net, $d_z = 1.10, p < 0.01$), marginal effects (such as Swin-Tiny vs. ResNet-50 Recall, $p_{\text{adj}} = 0.066$) require larger seed cohorts ($N \ge 20$) to reach definitive significance under strict Holm-Bonferroni correction.
2. **Capacity Confound in MoE**: Swin-MoE ($81.15\text{M}$) has $2.77\times$ the parameters of Vanilla Swin ($29.29\text{M}$). Its variance reduction cannot be separated from overparameterization without a parameter-matched dense control (Wide-FFN Swin).
3. **2D Slice Formulation**: All models operate on 2D ultrasound frames ($224 \times 224$); 3D volumetric context between adjacent scan planes is not modeled.
