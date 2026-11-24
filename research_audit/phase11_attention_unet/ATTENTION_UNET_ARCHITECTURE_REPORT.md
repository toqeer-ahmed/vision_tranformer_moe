# Attention U-Net: Architectural Specification & Design Report

**Experiment Identifier**: `phase11_attention_unet`  
**Lead Architect & Auditor**: Senior Computer Vision Research Engineer & Scientific Auditor  
**Repository**: `D:\Study Material\Research\CV\vision_transformer_research`  
**Date**: 2026-11-25  
**Theoretical Reference**: Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas", MIDL 2018 / arXiv:1804.03999  

---

## 1. Architectural Motivation & Clinical Context

In Phase 10, the canonical Standard U-Net ($31.04\text{M}$ parameters) demonstrated a significant zero-shot generalization deficit compared to pre-trained Vision Transformers:
- **Zero-Shot Foreground IoU**: $50.79\%$ (vs. $57.50\%$ Vanilla Swin, $p = 0.0425$).
- **The Precision-Recall Wedge**: $80.19\%$ Precision vs. $58.57\%$ Recall.
- **Volume Under-Prediction**: U-Net predicted only $6.00$ Megapixels across BUS-BRA ($N = 1,063$) against an actual ground truth of $8.15$ Megapixels ($0.74\times$ volume ratio), missing $41.43\%$ of tumor tissue.

### The Architectural Flaw in Standard U-Net Skip Connections:
In standard U-Net, encoder features $x^l$ are concatenated directly into decoder stages:
$$d^l = \operatorname{DoubleConv}([x^l, \operatorname{Up}(d^{l+1})])$$
Because $x^l$ carries raw, uncurated spatial activations across the entire ultrasound field of view (including speckle noise, acoustic reverberations, and normal fibroglandular parenchyma), the decoder experiences feature clutter. To maintain low loss, the network becomes overly conservative, firing only on dense central tumor cores and failing to segment subtle, tapering boundary margins.

### The Attention U-Net Remedy:
Attention Gates (AGs) dynamically filter encoder features $x^l$ prior to concatenation, using the coarser decoder signal $g$ (which contains rich contextual information) to generate a spatial attention map $\alpha \in [0, 1]$ that suppresses non-lesion acoustic background and accentuates salient lesion boundaries.

---

## 2. Mathematical Formulation of Additive Attention Gates

For a skip connection at encoder level $l$:
- **Skip feature map**: $x^l \in \mathbb{R}^{F_l \times H_l \times W_l}$
- **Gating signal**: $g \in \mathbb{R}^{F_g \times H_g \times W_g}$ (upsampled from the lower decoder stage)
- **Intermediate channel dimension**: $F_{\text{int}} = F_l // 2$

```
Skip Features x (F_l, H, W) ───────► [Conv 1x1: F_l -> F_int] (W_x) ──┐
                                                                       ▼
                                                                     [ + ] ──► [ReLU] ──► [Conv 1x1: F_int -> 1] (psi) ──► [BatchNorm] ──► [Sigmoid] ──► Attention Map alpha (1, H, W)
                                                                       ▲                                                                                        │
Gating Signal g (F_g, H, W) ───────► [Conv 1x1: F_g -> F_int] (W_g) ──┘                                                                                        │
                                                                                                                                                               ▼
Filtered Skip Features x_hat = alpha * x ◄─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────[ * ]
```

### Analytical Steps:
1. **Linear Transformations**:
   $$q = W_g^T g + b_g, \quad k = W_x^T x^l + b_x$$
   where $W_g \in \mathbb{R}^{F_{\text{int}} \times F_g}$, $W_x \in \mathbb{R}^{F_{\text{int}} \times F_l}$.
2. **Spatial Bilinear Alignment**: If spatial dimensions differ ($H_g \ne H_l$), $q$ is bilinearly upsampled to match $x^l$.
3. **Additive Attention & Non-Linear Activation**:
   $$\sigma_1 = \operatorname{ReLU}(q + k)$$
4. **Attention Coefficient Generation**:
   $$\alpha = \operatorname{Sigmoid}(\psi^T \sigma_1 + b_{\psi})$$
   where $\psi \in \mathbb{R}^{1 \times F_{\text{int}}}$, followed by 1-channel Batch Normalization.
5. **Gated Skip Output**:
   $$\hat{x}^l = \alpha \odot x^l$$
   The gated features $\hat{x}^l$ are then concatenated with the upsampled decoder features $g$:
   $$d^l = \operatorname{DoubleConv}([\hat{x}^l, g])$$

---

## 3. Stage-by-Stage Network Architecture

| Stage | Module Name | Input Dimensions | Operation Details | Output Dimensions | Parameter Count |
| :--- | :--- | :---: | :--- | :---: | :---: |
| **Input** | `input` | — | RGB Ultrasound Image | $(B, 3, 224, 224)$ | $0$ |
| **Encoder 0** | `inc` | $(B, 3, 224, 224)$ | DoubleConv: $3 \rightarrow 64 \rightarrow 64$ | $(B, 64, 224, 224)$ | $38,848$ |
| **Encoder 1** | `down1` | $(B, 64, 224, 224)$ | MaxPool $2\times 2$ + DoubleConv: $64 \rightarrow 128 \rightarrow 128$ | $(B, 128, 112, 112)$ | $221,696$ |
| **Encoder 2** | `down2` | $(B, 128, 112, 112)$ | MaxPool $2\times 2$ + DoubleConv: $128 \rightarrow 256 \rightarrow 256$ | $(B, 256, 56, 56)$ | $885,760$ |
| **Encoder 3** | `down3` | $(B, 256, 56, 56)$ | MaxPool $2\times 2$ + DoubleConv: $256 \rightarrow 512 \rightarrow 512$ | $(B, 512, 28, 28)$ | $3,540,992$ |
| **Bottleneck**| `down4` | $(B, 512, 28, 28)$ | MaxPool $2\times 2$ + DoubleConv: $512 \rightarrow 1024 \rightarrow 1024$| $(B, 1024, 14, 14)$| $14,159,872$ |
| **Decoder 1** | `up1.up` | $(B, 1024, 14, 14)$| ConvTranspose2d: $1024 \rightarrow 512$ ($2\times 2$, stride 2) | $(B, 512, 28, 28)$ | $2,097,664$ |
| | `up1.ag` | $g:(B,512,28,28), x:(B,512,28,28)$| AttentionGate: $F_g=512, F_l=512, F_{\text{int}}=256$ | $(B, 512, 28, 28)$ | **$263,425$** |
| | `up1.conv`| $(B, 1024, 28, 28)$| DoubleConv: $1024 \rightarrow 512 \rightarrow 512$ | $(B, 512, 28, 28)$ | $7,079,936$ |
| **Decoder 2** | `up2.up` | $(B, 512, 28, 28)$ | ConvTranspose2d: $512 \rightarrow 256$ ($2\times 2$, stride 2) | $(B, 256, 56, 56)$ | $524,544$ |
| | `up2.ag` | $g:(B,256,56,56), x:(B,256,56,56)$| AttentionGate: $F_g=256, F_l=256, F_{\text{int}}=128$ | $(B, 256, 56, 56)$ | **$66,177$** |
| | `up2.conv`| $(B, 512, 56, 56)$ | DoubleConv: $512 \rightarrow 256 \rightarrow 256$ | $(B, 256, 56, 56)$ | $1,770,496$ |
| **Decoder 3** | `up3.up` | $(B, 256, 56, 56)$ | ConvTranspose2d: $256 \rightarrow 128$ ($2\times 2$, stride 2) | $(B, 128, 112, 112)$| $131,200$ |
| | `up3.ag` | $g:(B,128,112,112), x:(B,128,112,112)$| AttentionGate: $F_g=128, F_l=128, F_{\text{int}}=64$ | $(B, 128, 112, 112)$| **$16,705$** |
| | `up3.conv`| $(B, 256, 112, 112)$| DoubleConv: $256 \rightarrow 128 \rightarrow 128$ | $(B, 128, 112, 112)$| $442,880$ |
| **Decoder 4** | `up4.up` | $(B, 128, 112, 112)$| ConvTranspose2d: $128 \rightarrow 64$ ($2\times 2$, stride 2) | $(B, 64, 224, 224)$ | $32,832$ |
| | `up4.ag` | $g:(B,64,224,224), x:(B,64,224,224)$| AttentionGate: $F_g=64, F_l=64, F_{\text{int}}=32$ | $(B, 64, 224, 224)$ | **$5,225$** |
| | `up4.conv`| $(B, 128, 224, 224)$| DoubleConv: $128 \rightarrow 64 \rightarrow 64$ | $(B, 64, 224, 224)$ | $110,848$ |
| **Head** | `outc` | $(B, 64, 224, 224)$ | Conv2d $1\times 1$: $64 \rightarrow 3$ | $(B, 3, 224, 224)$ | $195$ |
| **Total** | — | — | Full Attention U-Net Network | — | **$31,389,295$** |

---

## 4. Architectural Comparison with Phase 9B and Phase 10

| Property | Phase 9B Vanilla Swin-Tiny | Phase 9B Standard Swin-MoE | Phase 10 Standard U-Net | Phase 11 Attention U-Net |
| :--- | :---: | :---: | :---: | :---: |
| **Model Family** | Hierarchical ViT | Hierarchical ViT + MoE | Canonical 2D CNN | 2D CNN + Spatial Attention Gates |
| **Attention Mechanism** | Shifted-Window Self-Attention | Shifted-Window Self-Attention | None | Additive Spatial Gating on Skips |
| **Trainable Parameters** | $29,290,365$ | $81,148,989$ | $31,037,763$ | **$31,389,295$** |
| **Parameter Delta vs Swin**| Baseline | $+177.1\%$ | $+5.97\%$ | **$+7.17\%$** |
| **Parameter Delta vs UNet**| $-5.63\%$ | $+161.5\%$ | Baseline | **$+1.13\%$** ($+351,532$ params) |
| **Pretraining Status** | ImageNet-1K | ImageNet Warm-Init | None (From scratch) | None (From scratch) |
