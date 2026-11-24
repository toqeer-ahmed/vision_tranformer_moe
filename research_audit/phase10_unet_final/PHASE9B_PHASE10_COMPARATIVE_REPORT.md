# Cross-Phase Architectural Benchmark & Synthesis: Phase 9B (Swin ViT / Swin-MoE) vs. Phase 10 (Standard U-Net)

**Document Identifier**: `research_audit/phase10_unet_final/PHASE9B_PHASE10_COMPARATIVE_REPORT.md`  
**Auditor**: Senior Computer Vision Research Engineer & Scientific Auditor  
**Repository**: `D:\Study Material\Research\CV\vision_transformer_research`  
**Date**: 2026-11-25  

---

## 1. Benchmark Overview & Historical Experiment Context

This report establishes the unified comparative benchmark bridging **Phase 9B** (Hierarchical Vision Transformers: Vanilla Swin-Tiny and Standard-MLP Swin-MoE) and **Phase 10** (Canonical 2D Convolutional Neural Network: Standard U-Net).

> **Important Scientific Label**:  
> Comparisons between Phase 9B and Phase 10 represent a **historical cross-experiment benchmark**. All models share identical 80/10/10 seed-controlled data splits on BUSI, identical data augmentation pipelines, identical image resolutions ($224 \times 224$), identical loss formulations (Focal Tversky Loss), and identical zero-shot evaluation protocols on external BUS-BRA ($N = 1,063$). However, the experiments were executed in sequential phases with distinct pretraining regimes.

---

## 2. Corrected Master Cross-Architecture Comparative Table

| Model Architecture | Model Family | Parameters | Pretraining Status | Zero-Shot Fg IoU (%) | Zero-Shot Fg Dice (%) | Macro mIoU (%) | Macro mDice (%) | Fg Precision (%) | Fg Recall (%) | Pixel Acc (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vanilla Swin-Tiny** | Hierarchical ViT | **$29,290,365$** ($29.29\text{M}$) | ImageNet-1K (`pretrained=True`) | $57.50 \pm 4.74$ | $72.91 \pm 3.98$ | $76.10 \pm 2.82$ | $85.09 \pm 2.23$ | $70.44 \pm 6.61$ | $76.51 \pm 6.79$ | $95.06 \pm 0.98$ |
| **Standard-MLP MoE** | ViT + MoE (Top-2) | **$81,148,989$** ($81.15\text{M}$) | ImageNet Warm-Init | **$57.85 \pm 2.15$** | **$73.28 \pm 1.73$** | **$76.30 \pm 1.31$** | **$85.29 \pm 0.99$** | $69.92 \pm 4.33$ | **$77.35 \pm 3.73$** | $95.11 \pm 0.50$ |
| **Standard U-Net** | Canonical 2D CNN | **$31,037,763$** ($31.04\text{M}$) | None (`from_scratch=True`) | **$50.79 \pm 3.68$** | **$67.29 \pm 3.33$** | **$72.82 \pm 1.93$** | **$82.32 \pm 1.72$** | **$80.19 \pm 4.73$** | **$58.57 \pm 6.49$** | **$95.11 \pm 0.28$** |

*All statistics denote $\text{Mean} \pm \text{Sample Standard Deviation}$ ($N=10$). Parameter counts programmatically verified.*

---

## 3. Pairwise Statistical Comparisons with Multiple-Testing Corrections

Pairwise tests across the 10 matched seeds with Family-Wise Error Rate (FWER) controlled via **Holm-Bonferroni correction** across all 14 planned pairwise comparisons:

| Comparison Pair | Metric | Mean Paired $\Delta$ | $95\%$ CI for $\Delta$ | Paired $t$ | Raw $p_t$ | **Holm $p_t$** | Wilcoxon $W$ | Raw $p_W$ | **Holm $p_W$** | Cohen's $d_z$ | Empirical Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **U-Net vs. Vanilla Swin** | Foreground IoU | $\mathbf{-6.71\%}$ | $[-11.09\%, -2.33\%]$ | $-3.467$ | $0.0071$ | **$0.0425$** | $4.0$ | $0.0137$ | $0.0820$ | $-1.10$ | **Significantly Inferior ($p < 0.05$)** |
| | Foreground Dice | $\mathbf{-5.62\%}$ | $[-9.41\%, -1.83\%]$ | $-3.352$ | $0.0085$ | **$0.0425$** | $4.0$ | $0.0137$ | $0.0820$ | $-1.06$ | **Significantly Inferior ($p < 0.05$)** |
| | Macro mIoU | $-3.29\%$ | $[-5.78\%, -0.79\%]$ | $-2.982$ | $0.0154$ | $0.0549$ | $4.0$ | $0.0137$ | $0.0820$ | $-0.94$ | *Borderline trend ($p > 0.05$)* |
| | Macro mDice | $-2.77\%$ | $[-4.83\%, -0.72\%]$ | $-3.053$ | $0.0137$ | $0.0549$ | $4.0$ | $0.0137$ | $0.0820$ | $-0.97$ | *Borderline trend ($p > 0.05$)* |
| | Fg Precision | $\mathbf{+9.75\%}$ | $[+4.18\%, +15.32\%]$ | $+3.958$ | $0.0033$ | **$0.0232$** | $1.0$ | $0.0039$ | **$0.0312$** | $+1.25$ | **Significantly Superior ($p < 0.05$)** |
| | Fg Recall | $\mathbf{-17.94\%}$ | $[-24.47\%, -11.42\%]$ | $-6.218$ | $0.0002$ | **$0.0019$** | $0.0$ | $0.0020$ | **$0.0273$** | $-1.97$ | **Significantly Inferior ($p < 0.01$)** |
| **U-Net vs. Swin-MoE** | Foreground IoU | $\mathbf{-7.06\%}$ | $[-9.58\%, -4.54\%]$ | $-6.342$ | $0.0001$ | **$0.0017$** | $0.0$ | $0.0020$ | **$0.0273$** | $-2.01$ | **Significantly Inferior ($p < 0.01$)** |
| | Foreground Dice | $\mathbf{-5.99\%}$ | $[-8.25\%, -3.73\%]$ | $-5.991$ | $0.0002$ | **$0.0023$** | $0.0$ | $0.0020$ | **$0.0273$** | $-1.89$ | **Significantly Inferior ($p < 0.01$)** |
| | Macro mIoU | $\mathbf{-3.49\%}$ | $[-4.89\%, -2.08\%]$ | $-5.598$ | $0.0003$ | **$0.0034$** | $0.0$ | $0.0020$ | **$0.0273$** | $-1.77$ | **Significantly Inferior ($p < 0.01$)** |
| | Macro mDice | $\mathbf{-2.97\%}$ | $[-4.17\%, -1.77\%]$ | $-5.593$ | $0.0003$ | **$0.0034$** | $0.0$ | $0.0020$ | **$0.0273$** | $-1.77$ | **Significantly Inferior ($p < 0.01$)** |
| | Fg Precision | $\mathbf{+10.27\%}$ | $[+5.02\%, +15.52\%]$ | $+4.429$ | $0.0017$ | **$0.0132$** | $2.0$ | $0.0059$ | **$0.0410$** | $+1.40$ | **Significantly Superior ($p < 0.05$)** |
| | Fg Recall | $\mathbf{-18.78\%}$ | $[-23.57\%, -13.99\%]$ | $-8.869$ | $0.0000$ | **$0.0001$** | $0.0$ | $0.0020$ | **$0.0273$** | $-2.80$ | **Significantly Inferior ($p < 0.001$)** |

---

## 4. The Pretraining Confound: Decoupling Inductive Bias from Initialization

A core scientific mandate of this audit is to avoid conflating architecture with pretraining status:

1. **The Confounder**:
   - Vanilla Swin-Tiny ($29.29\text{M}$) and Standard U-Net ($31.04\text{M}$) possess virtually identical capacity.
   - However, Swin initialized its representation space using $1.28\text{M}$ ImageNet-1K images, whereas U-Net learned all $31.04\text{M}$ parameters from scratch using only 624 BUSI images.
   - The $+6.71\%$ IoU advantage of Swin reflects **both** the architectural benefit of hierarchical self-attention **and** the feature robustness endowed by pretraining.
2. **Next Necessary Scientific Step**:
   - To truly isolate convolutional inductive bias, an identically pre-trained convolutional baseline (e.g. ResNet-50 U-Net or ConvNeXt-Tiny U-Net) must be evaluated in future phases.

---

## 5. Variance Profile & The MoE Variance Observation

In Phase 9B, Standard-MLP Swin-MoE exhibited lower sample variance ($s = 2.15\%$) than Vanilla Swin-Tiny ($s = 4.74\%$). In Phase 10, Standard U-Net exhibits intermediate sample variance ($s = 3.68\%$).

### Critical Scientific Qualification:
- In this specific 10-seed experiment, Swin-MoE avoided the catastrophic failure modes observed in both Vanilla Swin (Seed 123: $47.33\%$) and Standard U-Net (Seed 999: $42.12\%$).
- However, with $N=10$, statistical variance equality tests (Brown-Forsythe $p = 0.584$) lack statistical power to prove that MoE is universally a variance buffer.
- Therefore, we report reduced run-to-run volatility as an **empirical observation in this 10-seed sample**, rather than a mathematically proven invariant property.
