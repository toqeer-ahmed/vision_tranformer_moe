# Phase 9 Reproducibility Audit: Controlled Multi-Seed Experimentation

**Audit Date**: November 12, 2026  
**Auditor**: Senior Computer Vision Research Engineer & Scientific Auditor  
**Repository**: `D:\Study Material\Research\CV\vision_transformer_research`  
**Target Milestone**: Phase 9 Controlled Multi-Seed Reproducibility Experiment  

---

## 1. Executive Summary of Reproducibility Audit

The Phase 9 experiment was designed to test whether replacing native Feed-Forward Networks (FFNs) in a Vision Transformer (`microsoft/swin-tiny-patch4-window7-224`) with a Standard-MLP Mixture-of-Experts (`StandardSwinMoELayer`) improves zero-shot cross-dataset generalization from the Breast Ultrasound Images (**BUSI**) dataset to the Brazilian Breast Ultrasound (**BUS-BRA**) dataset.

Six training runs were executed in a controlled cloud GPU environment (Kaggle T4 x2):
- **Vanilla Swin-Tiny**: Seeds `42`, `123`, `2026`
- **Standard-MLP Swin-MoE**: Seeds `42`, `123`, `2026`

This audit verifies all 15 operational criteria across the six runs, examines implementation code in `training/train_swin_vanilla.py`, `training/train_swin_standard_moe.py`, `Phase9_Controlled_Reproducibility.ipynb`, and identifies four critical engineering and methodological findings.

---

## 2. Comprehensive 15-Point Audit Matrix

The following matrix compares the execution configurations across all six runs:

| # | Audit Criterion | Vanilla Swin (Seed 42, 123, 2026) | Standard-MLP Swin-MoE (Seed 42, 123, 2026) | Controlled? | Evidence / Code Location |
|---|:---|:---|:---|:---:|:---|
| 1 | **Random Seed** | `42`, `123`, `2026` | `42`, `123`, `2026` | **YES** | `configs/reproducibility/swin_*.yaml` |
| 2 | **Training Dataset & Split** | BUSI (80% Train, 10% Val, 10% Test) | BUSI (80% Train, 10% Val, 10% Test) | **YES** | `datasets/medical_dataset.py#L210-L217` (`manual_seed(seed)`) |
| 3 | **Evaluation Dataset & Split** | Full BUS-BRA dataset (Zero-Shot) | Full BUS-BRA dataset (Zero-Shot) | **YES** | `Phase9_Controlled_Reproducibility.ipynb` (Cell 4) |
| 4 | **Batch Size** | `16` | `16` | **YES** | `configs/reproducibility/*.yaml` |
| 5 | **Learning Rate** | `0.0001` ($1 \times 10^{-4}$) | `0.0001` ($1 \times 10^{-4}$) | **YES** | `configs/reproducibility/*.yaml` |
| 6 | **Weight Decay** | `0.05` | `0.05` | **YES** | `configs/reproducibility/*.yaml` |
| 7 | **Optimizer** | `torch.optim.AdamW` | `torch.optim.AdamW` | **YES** | `training/train_swin_vanilla.py#L97`, `train_swin_standard_moe.py#L107` |
| 8 | **LR Scheduler** | `CosineAnnealingLR` ($T_{\max}=50, \eta_{\min}=10^{-6}$) | `CosineAnnealingLR` ($T_{\max}=50, \eta_{\min}=10^{-6}$) | **YES** | `train_swin_vanilla.py#L103`, `train_swin_standard_moe.py#L113` |
| 9 | **Loss Function** | Focal Tversky Loss (`dice_weight=1.0`) | Focal Tversky Loss (`dice_weight=1.0`) | **DISCREPANCY** | `train_swin_standard_moe.py#L142` (MoE Aux loss omitted) |
| 10 | **Input Resolution** | $224 \times 224$ pixels | $224 \times 224$ pixels | **YES** | `datasets/medical_dataset.py#L176` |
| 11 | **Data Augmentation** | Albumentations pipeline (Flip, Rotate, Affine, Elastic, Distort, Jitter) | Albumentations pipeline (Flip, Rotate, Affine, Elastic, Distort, Jitter) | **YES** | `datasets/medical_dataset.py#L175-L186` |
| 12 | **Number of Epochs** | 50 (Maximum) | 50 (Maximum) | **YES** | `configs/reproducibility/*.yaml` |
| 13 | **Early Stopping** | Patience = 10 epochs on validation mIoU | Patience = 10 epochs on validation mIoU | **YES** | `train_swin_vanilla.py#L117`, `train_swin_standard_moe.py#L127` |
| 14 | **Checkpoint Selection** | Best validation mIoU on BUSI | Best validation mIoU on BUSI | **YES** | `save_checkpoint(..., filename="best_model.pth")` |
| 15 | **Evaluation Procedure** | Streaming accumulator (`num_classes=2`, argmax > 0) | Streaming accumulator (`num_classes=2`, argmax > 0) | **YES** | `evaluation/metrics.py#L138`, `Phase9_Controlled_Reproducibility.ipynb` |

---

## 3. Detailed Audit of Individual Runs

### Run 1: Vanilla Swin — Seed 42
- **Config**: `configs/reproducibility/swin_vanilla_seed42.yaml`
- **Output Directory**: `outputs/swin_vanilla_20261004_141337`
- **In-Domain BUSI Test Results**: `mIoU: 0.8641 | mDice: 0.9189 | Pixel Acc: 0.9642`
- **Zero-Shot BUS-BRA Metrics**:
  - `mIoU`: 0.76882 (76.88%)
  - `Foreground IoU`: 0.58916 (58.92%)
  - `mDice`: 0.85752 (85.75%)
  - `Foreground Dice`: 0.74148 (74.15%)
  - `Foreground Precision`: 0.69515 (69.52%)
  - `Foreground Recall`: 0.79442 (79.44%)
- **Verification Status**: **VERIFIED** (Exact per-seed dictionary captured in log).

### Run 2: Vanilla Swin — Seed 123
- **Config**: `configs/reproducibility/swin_vanilla_seed123.yaml`
- **Output Directory**: `outputs/swin_vanilla_20261004_*`
- **Zero-Shot BUS-BRA Metrics**:
  - `mIoU`: 0.77403 (77.40%)
  - `Foreground IoU`: 0.59496 (59.50%)
  - `mDice`: 0.86102 (86.10%)
  - `Foreground Dice`: 0.74605 (74.61%)
  - `Foreground Precision`: 0.74768 (74.77%)
  - `Foreground Recall`: 0.74443 (74.44%)
- **Verification Status**: **VERIFIED** (Exact per-seed dictionary captured in log).

### Run 3: Vanilla Swin — Seed 2026
- **Config**: `configs/reproducibility/swin_vanilla_seed2026.yaml`
- **Output Directory**: `outputs/swin_vanilla_20261004_*`
- **Zero-Shot BUS-BRA Metrics**:
  - `mIoU`: 0.72140 (72.14%)
  - `Foreground IoU`: 0.49839 (49.84%)
  - `mDice`: 0.81832 (81.83%)
  - `Foreground Dice`: 0.66524 (66.52%)
  - `Foreground Precision`: 0.73940 (73.94%)
  - `Foreground Recall`: 0.60460 (60.46%)
- **Verification Status**: **VERIFIED** (Exact per-seed dictionary captured in log).

### Run 4: Standard-MLP Swin-MoE — Seed 42
- **Config**: `configs/reproducibility/swin_standard_moe_seed42.yaml`
- **Output Directory**: `outputs/swin_standard_moe_20261008_220211` / `outputs/swin_standard_moe_20261008_224105`
- **In-Domain BUSI Test Results**: `mIoU: 0.8624 | mDice: 0.9167 | Pixel Acc: 0.9635 | Precision: 0.5819 | Recall: 0.5849`
- **Zero-Shot BUS-BRA Evaluation**: Evaluated in Kaggle session at step 3915.
- **Verification Status**: **PARTIALLY VERIFIED (Cloud-Executed)**. The aggregate distribution is verified from terminal logs; individual JSON was saved to `/kaggle/working/results/phase9/paper_assets/`.

### Run 5: Standard-MLP Swin-MoE — Seed 123
- **Config**: `configs/reproducibility/swin_standard_moe_seed123.yaml`
- **Output Directory**: `outputs/swin_standard_moe_20261008_221522`
- **Zero-Shot BUS-BRA Evaluation**: Evaluated in Kaggle session at step 3915.
- **Verification Status**: **PARTIALLY VERIFIED (Cloud-Executed)**.

### Run 6: Standard-MLP Swin-MoE — Seed 2026
- **Config**: `configs/reproducibility/swin_standard_moe_seed2026.yaml`
- **Output Directory**: `outputs/swin_standard_moe_20261008_222535`
- **Zero-Shot BUS-BRA Evaluation**: Evaluated in Kaggle session at step 3915.
- **Verification Status**: **PARTIALLY VERIFIED (Cloud-Executed)**.

---

## 4. Discrepancies and Methodological Findings

### Finding 1: Auxiliary Load-Balancing Loss Disconnection (CRITICAL)
- **Description**: In `configs/reproducibility/swin_standard_moe_seed*.yaml`, `balance_loss_coef: 0.01` is explicitly configured. Inside `models/moe/standard_swin_moe_layer.py#L42-L51`, `TopKRouter` calculates `balance_loss` and stores it into `self.aux_loss`.
- **The Issue**: In `training/train_swin_standard_moe.py#L141-L146`:
  ```python
  with torch.amp.autocast('cuda'):
      logits = model(images)
      loss = criterion(logits, targets)
  scaler.scale(loss).backward()
  ```
  The training loop never queries `model` for auxiliary losses. `model` does not implement an `aux_loss` accumulator, and `loss` consists solely of the task loss (`CombinedSegmentationLoss`).
- **Scientific Impact**: The load-balancing loss was **never backpropagated** during training. The router gradients came solely from the task gradients propagated through the top-2 selected experts. The model succeeded without expert collapse due to **warm initialization** and **noisy gating**, but the configuration claimed load-balancing that was not active in gradient optimization.

### Finding 2: Timestamp-Based Folder Naming vs. Seed Discovery
- **Description**: Both training scripts hardcoded their output directories using current system timestamps:
  ```python
  experiment_dir = os.path.join("outputs", f"swin_{type}_{timestamp}")
  ```
  They ignored `model_cfg["name"]` (`swin_vanilla_seed42`).
- **The Issue**: When `Phase9_Controlled_Reproducibility.ipynb` (Cell 4) attempted to discover checkpoints by matching `f"outputs/swin_{model_type}_seed{seed}_*"`, zero directories matched. This caused an initial silent failure where the evaluation cell ran without evaluating checkpoints.
- **Resolution Taken in Transcript**: The evaluation logic was updated in step 3914 to read the embedded config/log files or sort timestamped directories chronologically to map folders to seeds (`[42, 123, 2026]`).

### Finding 3: Kaggle Pre-Installed Library Namespace Collision
- **Description**: The Kaggle runtime environment pre-loads HuggingFace's `datasets` library into memory before custom notebook execution.
- **The Issue**: Because `datasets` was already in `sys.modules`, `from datasets.medical_dataset import ...` failed with `ModuleNotFoundError: No module named 'datasets.medical_dataset'`.
- **Resolution Taken in Transcript**: Fixed by prepending the repo path to `sys.path` and deleting `sys.modules['datasets']` before importing local modules.

### Finding 4: Phase 8 to Phase 9 Hyperparameter Shift
- **Description**: During initial Phase 8 and early Phase 9 testing, training ran with `batch_size=32` and `weight_decay=0.01`.
- **The Issue**: A forensic review in October 2026 revealed that Vanilla Swin baseline had previously trained with `batch_size=16` and `weight_decay=0.05`.
- **Resolution**: Both architectures were re-standardized to `batch_size=16` and `weight_decay=0.05` across all 3 seeds in `configs/reproducibility/`, guaranteeing a controlled comparison.

---

## 5. Audit Verification Conclusion

The training protocol across the six runs was **strictly controlled** in terms of dataset splits, optimization parameters, resolution, and evaluation procedures. The empirical results reported in `PHASE9_FINAL_REPORT.md` reflect real executions on Kaggle, but the documentation must clearly note that the auxiliary load balancing loss was inactive during backpropagation.

