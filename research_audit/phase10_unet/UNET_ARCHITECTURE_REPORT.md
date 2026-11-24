# Canonical 2D U-Net Architecture Report

**Experiment Identifier**: `phase10_unet`  
**Architect**: Senior Computer Vision Research Engineer & Scientific Auditor  
**Repository**: `D:\Study Material\Research\CV\vision_transformer_research`  
**Reference Paper**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015.  

---

## 1. Architectural Design & Philosophy

To establish a pure convolutional baseline for breast ultrasound lesion segmentation, the architecture strictly adheres to the canonical 2D U-Net design:
1. **Four Encoder Stages**: Each block consists of two sequential $3 \times 3$ unpadded convolutions with Batch Normalization and ReLU activations, followed by $2 \times 2$ max-pooling with stride 2.
2. **Bottleneck Stage**: Deepest representation layer ($1,024$ channels) connecting the contracting and expanding paths.
3. **Four Symmetric Decoder Stages**: Each stage performs $2 \times 2$ transposed convolutions (up-convolutions) that halve feature channels and double spatial dimensions, concatenates the corresponding skip connections from the contracting path, and processes the concatenated representations with two $3 \times 3$ convolutions.
4. **Output Head**: Final $1 \times 1$ convolution projecting the 64 intermediate channels directly to the class logits.
5. **Ablation Controls**:
   - **No Attention Gates** (avoids confounding with Attention U-Net).
   - **No Transformer Layers** (pure convolutional inductive bias).
   - **No Pretrained Weights** (trained strictly from scratch on BUSI, providing a pure architectural comparison).

---

## 2. Programmatic Parameter Count Audit

The parameter count was calculated programmatically from the instantiated model tensors (`models/unet.py#count_parameters`):

```
Input (3 x 224 x 224)
   │
[Stage 1: Conv 64]  ──Skip (64)───► [UpStage 1: Conv 64] ──► Output (3 x 224 x 224)
   │ (MaxPool 2x2)                         ▲
[Stage 2: Conv 128] ──Skip (128)──► [UpStage 2: Conv 128]
   │ (MaxPool 2x2)                         ▲
[Stage 3: Conv 256] ──Skip (256)──► [UpStage 3: Conv 256]
   │ (MaxPool 2x2)                         ▲
[Stage 4: Conv 512] ──Skip (512)──► [UpStage 4: Conv 512]
   │ (MaxPool 2x2)                         ▲
[Bottleneck: Conv 1024] ───────────────────┘
```

### Exact Parameter Breakdown:

| Sub-Module / Block | Layer Operation | Kernel Size | In Channels | Out Channels | Parameter Count | Trainable Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Stage 1 (Contracting)** | `inc.conv1` + BN<br>`inc.conv2` + BN | $3 \times 3$<br>$3 \times 3$ | $3$<br>$64$ | $64$<br>$64$ | $1,856$<br>$36,992$ | Trainable<br>Trainable |
| **Stage 2 (Contracting)** | `down1.conv1` + BN<br>`down1.conv2` + BN | $3 \times 3$<br>$3 \times 3$ | $64$<br>$128$ | $128$<br>$128$ | $73,984$<br>$147,712$ | Trainable<br>Trainable |
| **Stage 3 (Contracting)** | `down2.conv1` + BN<br>`down2.conv2` + BN | $3 \times 3$<br>$3 \times 3$ | $128$<br>$256$ | $256$<br>$256$ | $295,424$<br>$590,336$ | Trainable<br>Trainable |
| **Stage 4 (Contracting)** | `down3.conv1` + BN<br>`down3.conv2` + BN | $3 \times 3$<br>$3 \times 3$ | $256$<br>$512$ | $512$<br>$512$ | $1,180,672$<br>$2,360,320$ | Trainable<br>Trainable |
| **Total Encoder** | **4 Stages** | — | — | — | **4,687,296** | **100% Trainable** |
| **Bottleneck** | `down4.conv1` + BN<br>`down4.conv2` + BN | $3 \times 3$<br>$3 \times 3$ | $512$<br>$1024$ | $1024$<br>$1024$ | $4,720,640$<br>$9,439,232$ | Trainable<br>Trainable |
| **Total Bottleneck** | **Bottleneck Block** | — | — | — | **14,159,872** | **100% Trainable** |
| **UpStage 4 (Expanding)** | `up1.transposed_conv`<br>`up1.conv1` + BN<br>`up1.conv2` + BN | $2 \times 2$<br>$3 \times 3$<br>$3 \times 3$ | $1024$<br>$1024$<br>$512$ | $512$<br>$512$<br>$512$ | $2,097,664$<br>$4,719,616$<br>$2,360,320$ | Trainable<br>Trainable<br>Trainable |
| **UpStage 3 (Expanding)** | `up2.transposed_conv`<br>`up2.conv1` + BN<br>`up2.conv2` + BN | $2 \times 2$<br>$3 \times 3$<br>$3 \times 3$ | $512$<br>$512$<br>$256$ | $256$<br>$256$<br>$256$ | $524,544$<br>$1,180,160$<br>$590,336$ | Trainable<br>Trainable<br>Trainable |
| **UpStage 2 (Expanding)** | `up3.transposed_conv`<br>`up3.conv1` + BN<br>`up3.conv2` + BN | $2 \times 2$<br>$3 \times 3$<br>$3 \times 3$ | $256$<br>$256$<br>$128$ | $128$<br>$128$<br>$128$ | $131,200$<br>$295,168$<br>$147,712$ | Trainable<br>Trainable<br>Trainable |
| **UpStage 1 (Expanding)** | `up4.transposed_conv`<br>`up4.conv1` + BN<br>`up4.conv2` + BN | $2 \times 2$<br>$3 \times 3$<br>$3 \times 3$ | $128$<br>$128$<br>$64$ | $64$<br>$64$<br>$64$ | $32,832$<br>$73,856$<br>$36,992$ | Trainable<br>Trainable<br>Trainable |
| **Total Decoder** | **4 Stages** | — | — | — | **12,190,400** | **100% Trainable** |
| **Final Head** | `outc.conv` ($1 \times 1$) | $1 \times 1$ | $64$ | $3$ | **195** | **100% Trainable** |
| **Grand Total Model** | **Canonical U-Net** | — | — | — | **31,037,763** | **100% Trainable** |

---

## 3. Parameter Parity with Vanilla Swin-Tiny

A remarkable scientific property of the canonical Ronneberger et al. (2015) U-Net ($b=64$) is its near-exact parameter parity with Vanilla Swin-Tiny:

- **Canonical Standard U-Net**: $31,037,763$ parameters ($31.04\text{M}$)
- **Vanilla Swin-Tiny**: $29,290,365$ parameters ($29.29\text{M}$)
- **Parameter Difference**: $+1,747,398$ parameters ($+5.9\%$ delta)
- **Standard-MLP Swin-MoE**: $81,148,989$ parameters ($81.15\text{M}$, $2.61\times$ U-Net capacity)

This tight capacity alignment between Canonical U-Net (~31M) and Vanilla Swin-Tiny (~29M) enables an almost **capacity-controlled evaluation of 2D convolutional inductive bias versus windowed transformer self-attention**.
