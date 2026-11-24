# Attention U-Net: Full Training & Evaluation Protocol

**Experiment Identifier**: `phase11_attention_unet`  
**Auditor**: Senior Computer Vision Research Engineer & Scientific Auditor  
**Repository**: `D:\Study Material\Research\CV\vision_transformer_research`  
**Date**: 2026-11-25  

---

## 1. Experimental Design & Control Guarantees

Phase 11 evaluates **Attention U-Net** under conditions identical to Phase 9B (**Vanilla Swin-Tiny**, **Standard-MLP Swin-MoE**) and Phase 10 (**Standard U-Net**).

### Experimental Controls:
- **Exact Matched Random Seeds ($N=10$)**:
  $$\mathcal{S} = \{42, 123, 314, 777, 999, 1001, 2026, 2027, 2718, 9001\}$$
- **Exact Split Parity**: 
  - BUSI dataset ($N=780$ pairs): 624 train ($80\%$), 78 validation ($10\%$), 78 in-domain test ($10\%$).
  - Generated deterministically via `random_split(..., generator=torch.Generator().manual_seed(seed))`.
- **Zero-Shot External Benchmark**:
  - BUS-BRA dataset ($N=1,063$ pairs) remains strictly unseen during training, validation, and hyperparameter tuning.

---

## 2. Optimization & Hyperparameter Specification

| Hyperparameter | Value / Specification | Parity to Phase 10 |
| :--- | :---: | :---: |
| **Model Architecture** | Attention U-Net (Oktay et al., 2018) | Phase 11 Target |
| **Base Channels ($C_{\text{base}}$)** | $64$ | Identical |
| **Total Trainable Parameters** | **$31,389,295$** ($31.39\text{M}$) | $+1.13\%$ vs. Standard U-Net ($31.04\text{M}$) |
| **Pretrained Weights** | None (`from_scratch=True`) | Identical |
| **Input Image Resolution** | $224 \times 224$ pixels | Identical |
| **Batch Size** | $16$ | Identical |
| **Optimizer** | AdamW | Identical |
| **Initial Learning Rate** | $1.0 \times 10^{-4}$ | Identical |
| **Weight Decay** | $0.05$ | Identical |
| **LR Scheduler** | CosineAnnealingLR ($T_{\max}=50, \eta_{\min}=10^{-6}$) | Identical |
| **Loss Function** | Combined Focal Tversky Loss ($\alpha=0.7, \beta=0.3, \gamma=0.75$) | Identical |
| **Maximum Epochs** | $50$ | Identical |
| **Early Stopping** | Validation mIoU, Patience $= 10$ epochs | Identical |
| **Automatic Mixed Precision** | PyTorch AMP (`torch.amp.autocast('cuda')`) | Identical |
| **Checkpoint Strategy** | Compact `state_dict` ($125.65$ MB per seed) | Identical |

---

## 3. Evaluation & Metric Logging Requirements

For every seed, the evaluation pipeline will record:
1. **In-Domain BUSI Test ($N=78$)**:
   - `test_metrics.json` recorded from the best validation checkpoint.
2. **External BUS-BRA Zero-Shot ($N=1,063$)**:
   - **Foreground IoU ($\text{class\_iou}[1]$)**: $\frac{\text{TP}_1}{\text{TP}_1 + \text{FP}_1 + \text{FN}_1}$
   - **Foreground Dice ($\text{class\_dice}[1]$)**: $\frac{2 \cdot \text{TP}_1}{2 \cdot \text{TP}_1 + \text{FP}_1 + \text{FN}_1}$
   - **Macro mIoU**: $\frac{1}{2} (\text{IoU}_0 + \text{IoU}_1)$
   - **Macro mDice**: $\frac{1}{2} (\text{Dice}_0 + \text{Dice}_1)$
   - **Foreground Precision**: $\frac{\text{TP}_1}{\text{TP}_1 + \text{FP}_1}$
   - **Foreground Recall**: $\frac{\text{TP}_1}{\text{TP}_1 + \text{FN}_1}$
   - **Pixel Accuracy**: $\frac{\text{TP}_0 + \text{TP}_1}{\text{Total Pixels}}$
   - **Predicted Lesion Volume**: $\text{TP}_1 + \text{FP}_1$ (pixels)
   - **Ground Truth Lesion Volume**: $\text{TP}_1 + \text{FN}_1$ ($8,146,649$ pixels)
   - **Volume Ratio**: $\frac{\text{Predicted Volume}}{\text{Ground Truth Volume}}$
   - **Full Confusion Matrices**: `class_tp`, `class_fp`, `class_fn` saved in `attention_unet_reproducibility_metrics.json`.

---

## 4. Execution Pipeline Commands

The training pipeline can be executed as a standalone script or module:

```bash
# Example Single Seed Execution (Seed 42)
python training/train_attention_unet.py --config configs/phase11_attention_unet/attention_unet_seed42.yaml
```

Outputs are automatically organized into:
- Checkpoints: `outputs/attention_unet_seed{seed}_{timestamp}/checkpoints/best_model.pth`
- Logs & In-Domain Metrics: `outputs/attention_unet_seed{seed}_{timestamp}/logs/test_metrics.json`
