# Five-Model Cross-Architectural Master Synthesis: Phases 9B through 12

## 1. Overview and Complete Experimental Arc

Across Phases 9B through 12, five architectures were evaluated under a rigorous, seed-matched experimental protocol on breast ultrasound lesion segmentation:
1. **Phase 9B**: Vanilla Swin-Tiny ($29.29\text{M}$ params, ImageNet-1K pretrained)
2. **Phase 9B**: Standard-MLP Swin-MoE ($33.64\text{M}$ params total / $28.32\text{M}$ active, ImageNet-1K pretrained)
3. **Phase 10**: Standard U-Net ($31.04\text{M}$ params, randomly initialized)
4. **Phase 11**: Attention U-Net ($31.39\text{M}$ params, randomly initialized)
5. **Phase 12**: ResNet-50 U-Net ($32.52\text{M}$ params, ImageNet-1K pretrained)

All models were trained on BUSI ($80\%$ train, $10\%$ val, $10\%$ test) across **10 matched seeds** (`42, 123, 314, 777, 999, 1001, 2026, 2027, 2718, 9001`) with identical optimizer (AdamW, $lr=1e-4, wd=0.05$), scheduler (CosineAnnealingLR), and loss function (Combined Focal Tversky Loss). Zero-shot generalization was evaluated on external, completely unseen BUS-BRA ($N = 1,063$).

---

## 2. Master Evaluation Table

| Model | Architecture Class | Pretraining | Params (M) | In-Domain BUSI FG IoU (%) | Zero-Shot BUS-BRA FG IoU (%) | Zero-Shot FG Dice (%) | Zero-Shot FG Recall (%) | Zero-Shot FG Precision (%) | Pixel Accuracy (%) | Mean Volume Ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| **Standard-MLP Swin-MoE** | Sparsely-Gated ViT | ImageNet-1K | 81.15M (46.64M) | $62.0 \pm 2.8$ | **$57.85 \pm 2.15$** | $73.28 \pm 1.73$ | **$77.35 \pm 3.73$** | $69.92 \pm 4.33$ | $95.11 \pm 0.50$ | $1.11 \pm 0.11$ |
| **Vanilla Swin-Tiny** | Hierarchical ViT | ImageNet-1K | 29.29M (29.29M) | $61.5 \pm 3.5$ | **$57.50 \pm 4.74$** | $72.91 \pm 3.98$ | **$76.51 \pm 6.79$** | $70.44 \pm 6.61$ | $95.06 \pm 0.98$ | $1.10 \pm 0.18$ |
| **Standard U-Net** | Canonical CNN | Scratch | 31.04M (31.04M) | $56.06 \pm 3.88$ | **$50.79 \pm 3.68$** | $67.29 \pm 3.33$ | $58.57 \pm 6.49$ | **$80.19 \pm 4.73$** | $95.11 \pm 0.28$ | $0.74 \pm 0.12$ |
| **ResNet-50 U-Net** | Residual CNN | ImageNet-1K | 32.52M (32.52M) | **$59.26 \pm 4.35$** | **$50.36 \pm 6.98$** | $66.71 \pm 6.48$ | $62.84 \pm 10.34$ | $72.62 \pm 7.13$ | $94.65 \pm 0.82$ | $0.87 \pm 0.17$ |
| **Attention U-Net** | Attention-Gated CNN | Scratch | 31.39M (31.39M) | $56.10 \pm 2.53$ | **$45.35 \pm 8.64$** | $61.93 \pm 8.84$ | $51.33 \pm 11.32$ | **$81.38 \pm 4.38$** | $94.71 \pm 0.61$ | $0.64 \pm 0.16$ |

---

## 3. Key Scientific Conclusions for the Research Manuscript

1. **Hierarchy of Generalization**:
   $$\text{Swin-MoE} (57.85\%) \approx \text{Vanilla Swin} (57.50\%) > \text{Standard U-Net} (50.79\%) \approx \text{ResNet-50 U-Net} (50.36\%) > \text{Attention U-Net} (45.35\%)$$
2. **MoE Variance Reduction**:
   Standard-MLP Swin-MoE achieves the highest stability across seeds, reducing standard deviation from $4.74\%$ (Vanilla Swin) to **$2.15\%$** (a $54.6\%$ reduction in variance), indicating that sparse expert routing acts as an ensemble-like stabilizer under domain shift.
3. **The Pretraining Decoupling**:
   Pretraining improves in-domain convergence and IoU ($56.06\% \to 59.26\%$) and helps relieve extreme volume shrinkage ($0.74\times \to 0.87\times$), but **does not resolve the cross-dataset generalization gap** ($50.36\%$ vs. $57.50\%$).
4. **Volume Calibration Pathology in CNNs**:
   Both randomly initialized CNNs (Standard U-Net and Attention U-Net) severely under-segment lesions ($0.74\times$ and $0.64\times$). Pretraining mitigates this partially ($0.87\times$), but Vision Transformers achieve superior volume calibration ($1.10\times$, dataset-level $0.97\times$).
5. **Attention Gate Degradation Under Domain Shift**:
   Spatial attention gates (Oktay et al., 2018) act as diffuse scalar attenuators rather than sharp focus masks under ultrasound domain shift, degrading zero-shot FG IoU by $-5.44\%$ relative to canonical U-Net ($45.35\%$ vs. $50.79\%$).
