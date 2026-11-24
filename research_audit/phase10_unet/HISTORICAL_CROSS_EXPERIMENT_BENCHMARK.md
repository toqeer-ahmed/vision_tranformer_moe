# Historical Cross-Experiment Benchmark & Architectural Synthesis: Phase 9B (Swin ViT / Swin-MoE) vs. Phase 10 (Standard U-Net)

**Document Identifier**: `research_audit/phase10_unet/HISTORICAL_CROSS_EXPERIMENT_BENCHMARK.md`  
**Auditor & Lead Architect**: Senior Computer Vision Research Engineer & Scientific Auditor  
**Repository**: `D:\Study Material\Research\CV\vision_transformer_research`  
**Date**: 2026-11-25  

---

## 1. Research Overview & Benchmark Design

A central open question in medical image segmentation is whether **Vision Transformers** genuinely provide superior cross-dataset generalization compared to canonical **Convolutional Neural Networks (CNNs)**, or whether reported gains are artifacts of hyperparameter tuning, metric definitions, or non-matched seeds.

To resolve this with scientific rigor, Phase 9B and Phase 10 established an exact, seed-controlled empirical comparison across three models:
1. **Standard U-Net (Ronneberger et al., 2015)**: Canonical 2D convolutional encoder-decoder with symmetric skip connections ($31,037,763$ trainable parameters).
2. **Vanilla Swin-Tiny (Liu et al., 2021)**: Hierarchical Vision Transformer with shifted window self-attention ($29,290,365$ trainable parameters).
3. **Standard-MLP Swin-MoE**: Hierarchical Vision Transformer with Sparse Mixture-of-Experts routing ($81,148,989$ total parameters, active conditional routing).

### Strict Experimental Controls:
- **Parameter Parity**: Standard U-Net ($31.04\text{M}$) and Vanilla Swin-Tiny ($29.29\text{M}$) are within $5.9\%$ parameter parity.
- **Identical 10 Matched Seeds**: $\mathcal{S} = \{42, 123, 314, 777, 999, 1001, 2026, 2027, 2718, 9001\}$.
- **Identical Training Data**: Held-in BUSI dataset ($80\%$ train, $10\%$ val, $10\%$ in-domain test).
- **Identical External Benchmark**: Completely unseen BUS-BRA cohort ($N = 1,063$ images) with zero fine-tuning, threshold tuning, or adaptation.
- **Identical Loss & Optimizer**: Focal Tversky Loss, AdamW ($lr=1\times 10^{-4}, wd=0.05$), CosineAnnealingLR ($T_{\max}=50$).

---

## 2. Master Cross-Architecture Benchmark Results

The table below reports the verified empirical metrics on the unseen external BUS-BRA dataset ($N=1,063$) across all 10 matched seeds:

| Metric | Vanilla Swin-Tiny (29.29M) | Standard-MLP MoE (81.15M) | Standard U-Net (31.04M) | $\Delta(\text{U-Net} - \text{Vanilla})$ | $\Delta(\text{U-Net} - \text{MoE})$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Foreground IoU** | $57.50\% \pm 4.74\%$ | **$57.85\% \pm 2.15\%$** | $50.79\% \pm 3.68\%$ | **$-6.71\%$** ($p=0.0071$) | **$-7.06\%$** ($p=0.0001$) |
| **Foreground Dice** | $72.91\% \pm 3.98\%$ | **$73.28\% \pm 1.73\%$** | $67.29\% \pm 3.33\%$ | **$-5.62\%$** ($p=0.0085$) | **$-5.99\%$** ($p=0.0002$) |
| **Macro mIoU** | $76.10\% \pm 2.82\%$ | **$76.30\% \pm 1.31\%$** | $72.82\% \pm 1.93\%$ | **$-3.29\%$** ($p=0.0154$) | **$-3.49\%$** ($p=0.0003$) |
| **Macro mDice** | $85.09\% \pm 2.23\%$ | **$85.29\% \pm 0.99\%$** | $82.32\% \pm 1.72\%$ | **$-2.77\%$** ($p=0.0137$) | **$-2.97\%$** ($p=0.0003$) |
| **Foreground Precision** | $70.44\% \pm 6.61\%$ | $69.92\% \pm 4.33\%$ | **$80.19\% \pm 4.73\%$** | **$+9.75\%$** ($p=0.0033$) | **$+10.27\%$** ($p=0.0017$) |
| **Foreground Recall** | $76.51\% \pm 6.79\%$ | **$77.35\% \pm 3.73\%$** | $58.57\% \pm 6.49\%$ | **$-17.94\%$** ($p=0.0002$) | **$-18.78\%$** ($p < 0.00005$) |
| **Pixel Accuracy** | $95.06\% \pm 0.98\%$ | $95.11\% \pm 0.50\%$ | **$95.11\% \pm 0.28\%$** | $+0.04\%$ ($p=0.8939$) | $+0.00\%$ ($p=0.9958$) |
| **Worst-Seed Fg IoU** | $47.33\%$ (Seed 123) | **$54.18\%$** (Seed 123) | $42.12\%$ (Seed 999) | — | — |
| **Best-Seed Fg IoU** | **$61.33\%$** (Seed 9001) | $60.42\%$ (Seed 2718) | $55.33\%$ (Seed 42) | — | — |
| **Seed Range (Max - Min)**| $14.00\%$ | **$6.24\%$** | $13.21\%$ | — | — |

---

## 3. Deep Architectural & Scientific Insights

### A. The Failure Mode of From-Scratch Convolutions: The Precision-Recall Wedge
Standard U-Net achieves the highest precision in the benchmark ($80.19\%$, $+10.27\%$ higher than MoE), yet loses by over $7\%$ IoU on foreground segmentation.
- **Cause**: In breast ultrasound, malignant and benign tumors present with variable acoustic attenuation, posterior shadowing, and ill-defined boundaries.
- Because U-Net lacks large-scale pretraining and global receptive fields, its local $3 \times 3$ kernels only activate on high-contrast, unambiguous central lesion tissue.
- It systematically fails to segment faint acoustic margins and boundary extensions, resulting in a severe **$-18.78\%$ collapse in recall** ($58.57\%$ vs $77.35\%$).
- Pretrained Vision Transformers, by contrast, possess broad contextual attention that detects subtle contrast variations, achieving superior balance ($70\%$ precision, $77\%$ recall).

### B. Inductive Bias vs. Pretraining Capacity at Equal Parameter Budgets
At ~30M parameters, the inductive bias of canonical 2D convolutions (translational equivariance, strict locality) is insufficient to compensate for the absence of pre-trained visual representations when transferring across clinical domains.
- Standard U-Net ($31.04\text{M}$): $50.79\%$ IoU
- Vanilla Swin-Tiny ($29.29\text{M}$): $57.50\%$ IoU
- **Net Transformer Advantage**: $+6.71\%$ IoU ($p = 0.0071$, Cohen's $d_z = 1.10$).

### C. MoE Routing Provides the Strongest Variance Defense
- **Vanilla Swin-Tiny**: Suffers from run-to-run vulnerability ($s = 4.74\%$, worst seed $= 47.33\%$).
- **Standard U-Net**: Also suffers high seed volatility ($s = 3.68\%$, worst seed $= 42.12\%$).
- **Standard-MLP Swin-MoE**: Dramatically compresses seed volatility ($s = 2.15\%$, worst seed $= 54.18\%$).
- MoE's multi-expert routing ensures that no single seed experiences catastrophic failure, elevating the worst-case cross-dataset performance by **$+12.06\%$ IoU** over Standard U-Net and **$+6.85\%$ IoU** over Vanilla Swin.

---

## 4. Next Scientific Research Phase: Pretrained CNN Baselines

To complete the scientific narrative, the research team should isolate the effect of pretraining on convolutional architectures:
1. **Phase 11 Recommendation**: Implement a **Pretrained ResNet-50 / ConvNeXt-Tiny U-Net**.
2. **Scientific Question**: Does a pretrained CNN close the $+6.71\%$ IoU gap with Swin-Tiny, or does shifted-window self-attention provide an intrinsic advantage for acoustic speckle invariance?
