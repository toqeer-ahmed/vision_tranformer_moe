# Standard U-Net Baseline: Architecture Verification & Historical Cross-Experiment Report

**Experiment Identifier**: `phase10_unet`  
**Auditor & Lead Architect**: Senior Computer Vision Research Engineer & Scientific Auditor  
**Repository**: `D:\Study Material\Research\CV\vision_transformer_research`  
**Date**: 2026-11-24  
**Status**: Architecture & Pipeline Verified (Ready for Cloud GPU Execution)  

---

## 1. Executive Implementation Summary

The canonical 2D U-Net baseline (Ronneberger et al., 2015) has been implemented, programmatically verified, and integrated into the research repository:

1. **Model Implementation**:
   - `models/unet.py`: Canonical 4-stage encoder-decoder U-Net with $64$ base channels, double $3 \times 3$ convolutions, Batch Normalization, ReLU, $2 \times 2$ max-pooling, and $2 \times 2$ transposed convolutions.
   - **Exact Trainable Parameters**: **$31,037,763$** ($31.04\text{M}$ parameters, $100\%$ trainable, no frozen layers).
   - Near-exact parameter parity with Vanilla Swin-Tiny ($29.29\text{M}$ parameters, $+5.9\%$ delta), providing a controlled baseline to isolate 2D convolutional inductive bias versus windowed transformer attention.
2. **Automated Unit Testing**:
   - `tests/test_unet.py`: 5 automated unit tests passed in $24.0$ seconds on CPU, validating tensor shapes $(B, 3, 224, 224)$, Focal Tversky Loss, backward gradient flow across all 64 parameter tensors, checkpoint save/load, and metric accumulation.
3. **Reproducibility Configurations & Engine**:
   - 10 configuration files created in `configs/phase10_unet/unet_seed[42...9001].yaml`.
   - Training script implemented in `training/train_unet.py` with compact state-dict checkpointing ($118.49$ MB per checkpoint) and automated disk protection.
4. **Cloud Deployment Notebook**:
   - `Phase10_UNet_Baseline_Reproducibility.ipynb` is ready for one-click execution on Kaggle GPU.

---

## 2. Historical Cross-Experiment Comparison

The table below presents the verified Phase 9B empirical benchmark alongside the execution schema for Phase 10 Standard U-Net:

> **Important Scientific Label**:  
> The comparison between Phase 9B (Vision Transformers) and Phase 10 (Standard U-Net) represents a **historical cross-experiment comparison**. Both experiments utilize identical 80/10/10 seed-controlled BUSI splits, identical data augmentations, identical Focal Tversky loss, and identical zero-shot BUS-BRA ($N=1,063$) evaluation protocols. However, U-Net is trained from scratch, whereas Swin Transformers utilize pre-trained visual representations.

| Metric | Phase 9B Vanilla Swin-Tiny (Verified Empirical) | Phase 9B Standard-MLP MoE (Verified Empirical) | Phase 10 Standard U-Net (Scheduled Kaggle Run) | Scientific Comparison Angle |
| :--- | :---: | :---: | :---: | :--- |
| **Model Family** | Hierarchical ViT | ViT + Mixture-of-Experts | Canonical 2D CNN | CNN vs. Transformer Inductive Bias |
| **Pretrained Weights** | ImageNet-1K (`pretrained=True`) | ImageNet-1K (`warm_init=True`) | None (`from_scratch=True`) | Pretrained ViT vs. From-Scratch CNN |
| **Total Parameters** | **$29,290,365$** ($29.29\text{M}$) | **$81,148,989$** ($81.15\text{M}$) | **$31,037,763$** ($31.04\text{M}$) | **Near-exact parameter match with Vanilla Swin** |
| **Foreground IoU** | $57.50\% \pm 4.74\%$ | $57.85\% \pm 2.15\%$ | *Pending Kaggle Execution* | Does 2D conv bias exceed $57.5\%$ on small data? |
| **Foreground Dice** | $72.91\% \pm 3.98\%$ | $73.28\% \pm 1.73\%$ | *Pending Kaggle Execution* | Tumor boundary precision |
| **Macro mIoU** | $76.10\% \pm 2.82\%$ | $76.30\% \pm 1.31\%$ | *Pending Kaggle Execution* | Multi-class segmentation fidelity |
| **Macro mDice** | $85.09\% \pm 2.23\%$ | $85.29\% \pm 0.99\%$ | *Pending Kaggle Execution* | Global overlap consistency |
| **Foreground Precision** | $70.44\% \pm 6.61\%$ | $69.92\% \pm 4.33\%$ | *Pending Kaggle Execution* | False-positive resistance |
| **Foreground Recall** | $76.51\% \pm 6.79\%$ | $77.35\% \pm 3.73\%$ | *Pending Kaggle Execution* | Lesion sensitivity |
| **Observed Min IoU** | $47.33\%$ (Seed 123) | $54.18\%$ (Seed 123) | *Pending Kaggle Execution* | Does CNN suffer seed collapse on Seed 123? |
| **Observed Max IoU** | $61.33\%$ (Seed 9001) | $60.42\%$ (Seed 2718) | *Pending Kaggle Execution* | Peak performance comparison |

---

## 3. Immediate Next Step for Execution

To generate the empirical 10-seed results for Standard U-Net:
1. Upload [Phase10_UNet_Baseline_Reproducibility.ipynb](file:///D:/Study%20Material/Research/CV/vision_transformer_research/Phase10_UNet_Baseline_Reproducibility.ipynb) to Kaggle.
2. Select **GPU T4 x2** or **P100** accelerator and enable Internet.
3. Attach `busi-dataset` and `busbra-dataset`.
4. Click **Run All**.
5. Download the auto-packaged `results_phase10_unet.zip` and place it into `outputs/results_phase10_unet/`.
