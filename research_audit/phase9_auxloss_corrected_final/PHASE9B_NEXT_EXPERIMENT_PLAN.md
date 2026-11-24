# Next Research Experiment Plan: Standard U-Net Baseline

**Target Milestone**: Phase 10 — Classical CNN Baseline Benchmarking  
**Architect & Scientific Auditor**: Senior Computer Vision Research Engineer  
**Repository**: `D:\Study Material\Research\CV\vision_transformer_research`  
**Mandate**: Fully design the next experiment protocol without initiating training in this task.  

---

## 1. Scientific Motivation

A critical requirement for publication in top medical imaging venues (MICCAI, IEEE Transactions on Medical Imaging) is demonstrating that advanced architectural innovations provide genuine advantages over established, parsimonious baselines.

Reviewers will raise the following question:
> *"Does an $81\text{M}$-parameter Vision Transformer Mixture-of-Experts actually outperform a classical, properly-regularized $8\text{M}$-parameter U-Net on a small ultrasound cohort ($N=780$)?"*

Because Vision Transformers lack the hard translation equivariance and local spatial inductive biases inherent to convolutional neural networks, they often require larger pretraining corpora. Establishing a matched **Standard U-Net baseline** trained under identical conditions is essential to contextualize the performance of both Vanilla Swin-Tiny (29.29M) and Standard-MLP Swin-MoE (81.15M).

---

## 2. Model Architecture: Standard U-Net

The baseline architecture will be a canonical 4-stage encoder-decoder U-Net (Ronneberger et al., 2015):

```
Input (3 x 224 x 224)
   │
[Stage 1: Conv 64]  ──Skip───► [UpStage 1: Conv 64] ──► Output (3 x 224 x 224)
   │ (MaxPool)                    ▲
[Stage 2: Conv 128] ──Skip───► [UpStage 2: Conv 128]
   │ (MaxPool)                    ▲
[Stage 3: Conv 256] ──Skip───► [UpStage 3: Conv 256]
   │ (MaxPool)                    ▲
[Stage 4: Conv 512] ──Skip───► [UpStage 4: Conv 512]
   │ (MaxPool)                    ▲
[Bottleneck: Conv 1024] ──────────┘
```

### Architectural Specifications:
- **Input Resolution**: $224 \times 224 \times 3$.
- **Encoder**: 4 downsampling stages. Each stage contains two `Conv2d(3x3, padding=1)` $\to$ `BatchNorm2d` $\to$ `ReLU` blocks, followed by `MaxPool2d(2x2, stride=2)`.
  - Stage 1: $64$ feature maps.
  - Stage 2: $128$ feature maps.
  - Stage 3: $256$ feature maps.
  - Stage 4: $512$ feature maps.
- **Bottleneck**: Two `Conv2d(3x3)` $\to$ `BatchNorm2d` $\to$ `ReLU` blocks with $1,024$ feature channels.
- **Decoder**: 4 upsampling stages using `ConvTranspose2d(2x2, stride=2)` $\to$ channel concatenation with skip connection $\to$ two `Conv2d(3x3)` $\to$ `BatchNorm2d` $\to$ `ReLU` blocks.
- **Head**: Final `Conv2d(1x1)` projecting 64 channels to 3 output logits.
- **Target Parameters**: $\sim 7.76$ million parameters ($3.77\times$ smaller than Vanilla Swin, $10.45\times$ smaller than Swin-MoE).

---

## 3. Experimental Protocol & Matched Controls

To guarantee strict scientific equivalence with Phase 9B:

1. **Dataset Splits**:
   - In-domain training: BUSI ($N=780$), 80% train ($624$), 10% validation ($78$), 10% test ($78$).
   - Random partition generated using the identical seed-controlled `torch.Generator().manual_seed(seed)`.
   - Zero-shot evaluation: Full external BUS-BRA dataset ($N=1,063$).
2. **Data Transforms**:
   - Identical medical augmentation pipeline: `Resize(224, 224)`, `HorizontalFlip`, `VerticalFlip`, `RandomRotate90`, `Affine`, `ElasticTransform`, `GridDistortion`, `ColorJitter`, and ImageNet normalization.
3. **Loss Function**:
   - Identical Combined Segmentation Loss: Focal Tversky Loss ($\alpha=0.7, \beta=0.3, \gamma=0.75$, `dice_weight=1.0`).
4. **Optimization**:
   - Optimizer: AdamW, initial learning rate $1.0 \times 10^{-4}$, weight decay $0.05$.
   - Scheduler: CosineAnnealingLR ($T_{\max}=50, \eta_{\min}=10^{-6}$).
   - Batch size: $16$.
   - Epochs: 50 with Early Stopping (patience = 10 epochs on validation mIoU).
5. **Matched Random Seeds**:
   - Evaluated across the identical 10 random seeds:
     `[42, 123, 314, 777, 999, 1001, 2026, 2027, 2718, 9001]`.

---

## 4. Scientific Hypotheses & Decision Matrix

| Hypothesis | Expected Empirical Outcome | Methodological Interpretation |
| :--- | :--- | :--- |
| **Hypothesis A (CNN Inductive Bias Dominance)** | U-Net achieves Foreground IoU $\ge 58\%$ on BUS-BRA with equal or lower variance. | Hard convolutional spatial bias is superior to large-capacity transformers on small medical cohorts ($N < 1,000$). |
| **Hypothesis B (Transformer Generalization Advantage)** | U-Net achieves high in-domain BUSI IoU ($>70\%$) but collapses to $<50\%$ on external BUS-BRA. | Global self-attention enables better domain shift resistance than pure local convolutions. |
| **Hypothesis C (Seed Volatility Equivalence)** | U-Net exhibits seed instability similar to Vanilla Swin ($s \approx 4\%-5\%$). | Small sample training volatility is an inherent property of ultrasound data, not an artifact of transformer attention. |

---

## 5. Execution Roadmap (Next Task)

When approved to proceed, Phase 10 execution will follow these steps:
1. Implement canonical `models/unet.py` (~7.76M params) with unit test `tests/test_unet.py`.
2. Generate 10 configuration files `configs/phase10_unet/unet_seed*.yaml`.
3. Create training script `training/train_unet.py` following the compact checkpointing protocol.
4. Prepare Kaggle notebook `Phase10_UNet_Baseline_Reproducibility.ipynb`.
5. Execute the 10-seed matched runs and compute paired statistical tests against Vanilla Swin and Swin-MoE.
