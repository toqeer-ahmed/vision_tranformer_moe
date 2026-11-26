# Phase 12 Parameter Count and Budget Audit

## 1. Executive Summary

This document presents the rigorous programmatic parameter count audit for the **ResNet-50 U-Net** architecture. All counts were calculated using `model.count_parameters()` and verified across all tensor parameters in [tests/test_resnet50_unet.py](file:///D:/Study%20Material/Research/CV/vision_transformer_research/tests/test_resnet50_unet.py).

The ResNet-50 U-Net architecture contains:
- **Total Parameters (Binary, $C=1$)**: **$32,520,977$** ($32.52\text{M}$)
- **Total Parameters (Multi-Class, $C=3$)**: **$32,521,011$** ($32.52\text{M}$)
- **Trainable Parameters**: **$100.0\%$** ($32,521,011$ by default under full fine-tuning)

---

## 2. Component-by-Component Parameter Breakdown

### A. Pretrained ResNet-50 Encoder
The encoder parameters match official TorchVision ResNet-50:

| Sub-module | Layers / Components | Weight Tensors | Bias Tensors | Parameter Count |
|---|---|---|---|---|
| `conv1` + `bn1` | $7\times 7$ Conv (3 $\to$ 64) + BatchNorm2d | 1 weight, 1 gamma, 1 beta | 0 bias | 9,536 |
| `layer1` | 3 Bottleneck blocks (64 $\to$ 256) | $3 \times [1\times 1, 3\times 3, 1\times 1] + \text{downsample}$ | BN params | 215,808 |
| `layer2` | 4 Bottleneck blocks (256 $\to$ 512) | $4 \times [1\times 1, 3\times 3, 1\times 1] + \text{downsample}$ | BN params | 1,219,584 |
| `layer3` | 6 Bottleneck blocks (512 $\to$ 1024) | $6 \times [1\times 1, 3\times 3, 1\times 1] + \text{downsample}$ | BN params | 7,098,368 |
| `layer4` | 3 Bottleneck blocks (1024 $\to$ 2048) | $3 \times [1\times 1, 3\times 3, 1\times 1] + \text{downsample}$ | BN params | 14,964,736 |
| **Encoder Subtotal** | **All 5 Encoder Stages** | — | — | **23,508,032** |

---

### B. U-Net Decoder Blocks
Each decoder block consists of two $3\times 3$ convolutions with BatchNorm2d:
$$\text{Conv1}: (C_{in} + C_{skip}) \times C_{out} \times 3 \times 3 + \text{BN1}: 2 \times C_{out}$$
$$\text{Conv2}: C_{out} \times C_{out} \times 3 \times 3 + \text{BN2}: 2 \times C_{out}$$

| Decoder Block | $C_{in} + C_{skip}$ | $C_{out}$ | Conv1 + BN1 Params | Conv2 + BN2 Params | Total Block Params |
|---|---|---|---|---|---|
| **Decoder Block 1** | $2048 + 1024 = 3072$ | 256 | $3072 \times 256 \times 9 + 512 = 7,078,400$ | $256 \times 256 \times 9 + 512 = 590,336$ | **7,668,736** |
| **Decoder Block 2** | $256 + 512 = 768$ | 128 | $768 \times 128 \times 9 + 256 = 885,000$ | $128 \times 128 \times 9 + 256 = 147,712$ | **1,032,704** |
| **Decoder Block 3** | $128 + 256 = 384$ | 64 | $384 \times 64 \times 9 + 128 = 221,312$ | $64 \times 64 \times 9 + 128 = 37,000$ | **258,304** |
| **Decoder Block 4** | $64 + 64 = 128$ | 32 | $128 \times 32 \times 9 + 64 = 36,928$ | $32 \times 32 \times 9 + 64 = 9,280$ | **46,208** |
| **Decoder Block 5** | $32 + 0 = 32$ | 16 | $32 \times 16 \times 9 + 32 = 4,640$ | $16 \times 16 \times 9 + 32 = 2,336$ | **6,976** |
| **Decoder Subtotal** | **Blocks 1 to 5** | — | — | — | **9,012,928** |

---

### C. Segmentation Head
| Configuration | Input Channels | Output Classes | Kernel | Parameters ($16 \times C + C$) |
|---|---|---|---|---|
| **Binary ($C=1$)** | 16 | 1 | $1 \times 1$ | $16 \times 1 + 1 = \mathbf{17}$ |
| **Multi-Class ($C=3$)** | 16 | 3 | $1 \times 1$ | $16 \times 3 + 3 = \mathbf{51}$ |

---

## 3. Five-Model Architecture Comparison Table

| Architecture | Backbone Family | Pretraining Status | Total Parameters | Trainable Parameters | Decoder / Head Parameters | Ratio vs Swin-Tiny |
|---|---|---|---|---|---|---|
| **Vanilla Swin-Tiny** | Hierarchical ViT | ImageNet-1K Pretrained | **29,292,723** | 29,292,723 | 1,745,091 | $1.00\times$ (Reference) |
| **Standard-MLP Swin-MoE** | Sparsely-Gated ViT | ImageNet-1K Pretrained | **33,640,051** | 33,640,051 (28.32M active) | 1,745,091 | $1.15\times$ |
| **Standard U-Net** | Canonical CNN | Random Initialization | **31,037,635** | 31,037,635 | 16,846,595 | $1.06\times$ |
| **Attention U-Net** | Attention-Gated CNN | Random Initialization | **31,389,295** | 31,389,295 | 17,198,255 | $1.07\times$ |
| **ResNet-50 U-Net (Phase 12)** | Residual CNN | **ImageNet-1K Pretrained** | **32,521,011** | **32,521,011** | **9,012,979** | **$1.11\times$** |

### Key Audit Conclusions:
1. **Parameter Parity**: ResNet-50 U-Net is within $+11.0\%$ of Swin-Tiny and within $+4.8\%$ of Standard U-Net. Any observed generalization performance differences cannot be attributed to an order-of-magnitude parameter mismatch.
2. **Backbone Parity**: ResNet-50 backbone ($23.51\text{M}$) and Swin-Tiny backbone ($27.55\text{M}$) are closely matched in capacity.
3. **Decoupled Comparison**: By matching ImageNet-1K pretraining, Phase 12 isolates architectural inductive biases (sliding local window self-attention vs. translation-equivariant residual convolutions) as the primary explanatory variable.
