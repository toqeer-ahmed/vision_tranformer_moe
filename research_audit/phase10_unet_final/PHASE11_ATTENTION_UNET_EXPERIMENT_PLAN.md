# Phase 11: Attention U-Net Architectural Specification & Experiment Execution Plan

**Document Identifier**: `research_audit/phase10_unet_final/PHASE11_ATTENTION_UNET_EXPERIMENT_PLAN.md`  
**Lead Architect & Auditor**: Senior Computer Vision Research Engineer & Scientific Auditor  
**Repository**: `D:\Study Material\Research\CV\vision_transformer_research`  
**Target Milestone**: Phase 11 Attention U-Net Baseline Evaluation  
**Status**: Architecture Verified & Experiment Specified (DO NOT BEGIN TRAINING DURING AUDIT)  

---

## 1. Scientific Objective & Rationale

Phase 10 proved that **Standard U-Net** suffers from a severe **precision-recall wedge** on out-of-distribution breast ultrasound images:
- High Precision: $80.19\%$
- Crippled Recall: $58.57\%$ (Volume under-prediction ratio: $0.74 \times \text{GT}$)
- Foreground IoU: $50.79\%$ (Trailing Swin by $-6.71\%$)

Standard U-Net transfers all low-level encoder features across skip connections indiscriminately. This introduces background clutter, forcing the decoder to over-constrain its threshold and predict only high-confidence lesion centers.

### The Attention U-Net Hypothesis (Oktay et al., 2018):
Can integrating **Attention Gates (AGs)** into the skip connections selectively filter out background acoustic noise, amplify salient tumor boundary features, and **boost recall from $58.57\%$ toward $\ge 70\%$ without degrading precision**?

---

## 2. Model Architecture & Exact Parameter Verification

Attention U-Net integrates Attention Gates at every decoder stage where skip connections merge.

```
Input (3, 224, 224)
   │
   ▼
[DoubleConv: 3 -> 64] ───────────► [Attention Gate 4] ──────────┐
   │ (MaxPool 2x2)                         ▲                    │
   ▼                                       │                    ▼
[DoubleConv: 64 -> 128] ─────────► [Attention Gate 3] ──┐   [DoubleConv: 128 -> 64]
   │ (MaxPool 2x2)                         ▲            │       │
   ▼                                       │            ▼       ▼
[DoubleConv: 128 -> 256] ────────► [Attention Gate 2] ──┴─► [DoubleConv: 256 -> 128]
   │ (MaxPool 2x2)                         ▲                    │
   ▼                                       │                    ▼
[DoubleConv: 256 -> 512] ────────► [Attention Gate 1] ────────► [DoubleConv: 512 -> 256]
   │ (MaxPool 2x2)                         ▲                    │
   ▼                                       │                    ▼
[DoubleConv: 512 -> 1024] ──► [UpConv 1024 -> 512] ───────────► [DoubleConv: 1024 -> 512]
(Bottleneck Stage)
```

### Mathematical Formulation of Attention Gate (AG):
For gating signal $g \in \mathbb{R}^{F_g \times H \times W}$ (from lower decoder stage) and skip features $x^l \in \mathbb{R}^{F_l \times H \times W}$ (from encoder stage $l$):
1. **Linear Projections**:
   $$q_{\text{att}} = W_g^T g + b_g, \quad k_{\text{att}} = W_x^T x^l + b_x$$
   where $W_g \in \mathbb{R}^{F_{\text{int}} \times F_g}$, $W_x \in \mathbb{R}^{F_{\text{int}} \times F_l}$, and $F_{\text{int}} = F_l / 2$.
2. **Spatial Activation**:
   $$\sigma_1 = \operatorname{ReLU}(q_{\text{att}} + k_{\text{att}})$$
3. **Attention Coefficient Generation**:
   $$\alpha = \operatorname{Sigmoid}(\psi^T \sigma_1 + b_{\psi})$$
   where $\psi \in \mathbb{R}^{1 \times F_{\text{int}}}$.
4. **Gated Output**:
   $$\hat{x}^l = \alpha \odot x^l$$

### Programmatically Verified Parameter Breakdown:
Using canonical base channels $C_{\text{base}} = 64$:
- **Encoder (inc + down1..4)**: $18,847,168$ parameters
- **Decoder Convolutions & UpConvs**: $12,190,400$ parameters
- **Output Head (1x1 Conv)**: $195$ parameters
- **Attention Gate 1 ($F_g=512, F_l=512, F_{\text{int}}=256$)**: $263,425$ parameters
- **Attention Gate 2 ($F_g=256, F_l=256, F_{\text{int}}=128$)**: $66,177$ parameters
- **Attention Gate 3 ($F_g=128, F_l=128, F_{\text{int}}=64$)**: $16,705$ parameters
- **Attention Gate 4 ($F_g=64, F_l=64, F_{\text{int}}=32$)**: $5,225$ parameters
- **Total Attention Gate Parameters**: **$351,532$ parameters**
- **Total Trainable Parameters**: **$31,389,295$ parameters** ($31.39\text{M}$)

*Comparison*: Attention U-Net adds only **$+1.13\%$ parameters** over Standard U-Net ($31.04\text{M}$), maintaining exact architectural comparability.

---

## 3. Strict Experimental Controls & Hyperparameters

To ensure seamless integration into the research benchmark, all training hyperparameters are held identical to Phase 9B and Phase 10:

| Hyperparameter | Phase 10 Standard U-Net | Phase 11 Attention U-Net | Control Status |
| :--- | :---: | :---: | :---: |
| **Model Family** | Canonical 2D CNN | 2D CNN + Spatial Attention Gates | Comparative Target |
| **Trainable Parameters** | $31,037,763$ ($31.04\text{M}$) | **$31,389,295$** ($31.39\text{M}$) | **$+1.13\%$ delta** (Near-exact match) |
| **Pretrained Weights** | None (`from_scratch=True`) | None (`from_scratch=True`) | **Identical** (From scratch) |
| **Training Dataset** | BUSI ($N=780$) | BUSI ($N=780$) | **Identical** |
| **Split Protocol** | 80% train, 10% val, 10% test | 80% train, 10% val, 10% test | **Identical** (624 / 78 / 78) |
| **Seeds ($N=10$)** | $\{42, 123, 314, 777, 999, 1001, 2026, 2027, 2718, 9001\}$ | Same 10 seeds | **Identical** |
| **Resolution** | $224 \times 224$ | $224 \times 224$ | **Identical** |
| **Batch Size** | $16$ | $16$ | **Identical** |
| **Optimizer** | AdamW | AdamW | **Identical** |
| **Initial Learning Rate** | $1.0 \times 10^{-4}$ | $1.0 \times 10^{-4}$ | **Identical** |
| **Weight Decay** | $0.05$ | $0.05$ | **Identical** |
| **LR Scheduler** | CosineAnnealingLR ($T_{\max}=50$) | CosineAnnealingLR ($T_{\max}=50$) | **Identical** |
| **Loss Function** | Focal Tversky Loss | Focal Tversky Loss | **Identical** |
| **Patience** | 10 epochs | 10 epochs | **Identical** |
| **External Zero-Shot** | BUS-BRA ($N=1,063$) | BUS-BRA ($N=1,063$) | **Identical** |

---

## 4. Competitive Scientific Hypotheses & Target Thresholds

| Metric | Phase 10 Standard U-Net | Phase 9B Vanilla Swin-Tiny | Phase 11 Hypothesis A (Attention Recovers Recall) | Phase 11 Hypothesis B (Attention Ineffective) |
| :--- | :---: | :---: | :---: | :---: |
| **Foreground Recall** | $58.57\% \pm 6.49\%$ | $76.51\% \pm 6.79\%$ | **$\ge 70.0\%$** (Gating recovers faint acoustic borders) | $< 62.0\%$ (Skip gating cannot fix lack of pretraining) |
| **Foreground Precision** | $80.19\% \pm 4.73\%$ | $70.44\% \pm 6.61\%$ | **$\ge 75.0\%$** (Noise filtering preserves high precision) | $< 72.0\%$ (Boundary bleed degrades precision) |
| **Foreground IoU** | $50.79\% \pm 3.68\%$ | $57.50\% \pm 4.74\%$ | **$\ge 55.0\%$** (Closes the ViT generalization gap) | $\le 51.5\%$ (Fails to outperform Standard U-Net) |
| **Volume Ratio** | $0.74 \times \text{GT}$ | $1.10 \times \text{GT}$ | **$0.95 - 1.05 \times \text{GT}$** (Resolves under-prediction) | $\le 0.80 \times \text{GT}$ |

---

## 5. Execution Rules & Immediate Next Steps

> **Mandate**:  
> In accordance with supervisor directives, **Attention U-Net training will NOT begin during this audit turn**.

### Next Actions for Phase 11 Launch:
1. Implement `models/attention_unet.py` and unit test in `tests/test_attention_unet.py`.
2. Generate 10 configuration files in `configs/phase11_attention_unet/att_unet_seed[42...9001].yaml`.
3. Create training pipeline in `training/train_attention_unet.py`.
4. Package the execution notebook `Phase11_Attention_UNet_Reproducibility.ipynb` for Kaggle GPU execution.
