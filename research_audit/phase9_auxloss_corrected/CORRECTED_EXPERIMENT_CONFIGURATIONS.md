# Corrected Phase 9 Experiment Configurations

**Experiment Identifier**: `phase9_auxloss_corrected`  
**Configuration Directory**: `configs/phase9_auxloss_corrected/`  
**Target Architecture**: Vanilla Swin-Tiny vs. Standard-MLP Swin-MoE  
**Auditor**: Senior Computer Vision Research Engineer & Scientific Auditor  

---

## 1. Experimental Design & Seed Protocol

To eliminate random initialization bias and achieve statistical validity, we expanded the multi-seed protocol from 3 to **10 matched random seeds**:

$$\mathcal{S} = \{42, 123, 2026, 777, 999, 314, 2718, 1001, 2027, 9001\}$$

- **Primary Tier (5 Seeds)**: `[42, 123, 2026, 777, 999]` (recommended for standard GPU budget, $df=4$).
- **Extended Tier (10 Seeds)**: `[42, 123, 2026, 777, 999, 314, 2718, 1001, 2027, 9001]` (for high statistical power, $df=9$).

Each seed has two exactly matched configuration files:
1. `swin_vanilla_seed{SEED}.yaml`
2. `swin_standard_moe_seed{SEED}.yaml`

---

## 2. Configuration Comparison Matrix

| Parameter Category | Hyperparameter | Vanilla Swin-Tiny | Standard-MLP Swin-MoE | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Model** | Backbone | `microsoft/swin-tiny-patch4-window7-224` | `microsoft/swin-tiny-patch4-window7-224` | **IDENTICAL** |
| | Number of Classes | 3 (Background, Benign, Malignant) | 3 (Background, Benign, Malignant) | **IDENTICAL** |
| | Number of Experts | 1 (Native FFN) | 4 | Intended MoE |
| | Top-k Selection | N/A | 2 | Intended MoE |
| | Expert Architecture | Native Swin FFN | `SimpleMLPBlock` (Pure 2-layer MLP) | **MATCHED** |
| | Noisy Gating | N/A | True (Gaussian exploration) | Intended MoE |
| | Balance Loss Coef | N/A | 0.01 | **REPAIRED** |
| | Capacity Factor | N/A | 1.2 | Intended MoE |
| | Warm Initialization | N/A | True (From ImageNet FFN) | Intended MoE |
| **Dataset** | Dataset & Split | BUSI (80% Train, 10% Val, 10% Test) | BUSI (80% Train, 10% Val, 10% Test) | **IDENTICAL** |
| | Split Generator | `torch.Generator().manual_seed(seed)` | `torch.Generator().manual_seed(seed)` | **IDENTICAL** |
| | Input Resolution | $224 \times 224$ pixels | $224 \times 224$ pixels | **IDENTICAL** |
| | Augmentation | Flips, Rotate, Affine, Elastic, Jitter | Flips, Rotate, Affine, Elastic, Jitter | **IDENTICAL** |
| **Optimization**| Batch Size | 16 | 16 | **IDENTICAL** |
| | Optimizer | AdamW | AdamW | **IDENTICAL** |
| | Learning Rate | $1 \times 10^{-4}$ ($0.0001$) | $1 \times 10^{-4}$ ($0.0001$) | **IDENTICAL** |
| | Weight Decay | $0.05$ | $0.05$ | **IDENTICAL** |
| | Scheduler | CosineAnnealingLR ($T_{\max}=50, \eta_{\min}=10^{-6}$) | CosineAnnealingLR ($T_{\max}=50, \eta_{\min}=10^{-6}$) | **IDENTICAL** |
| | Mixed Precision | PyTorch AMP (`torch.amp.autocast`) | PyTorch AMP (`torch.amp.autocast`) | **IDENTICAL** |
| | Primary Loss | Focal Tversky Loss (`dice_weight=1.0`) | Focal Tversky Loss (`dice_weight=1.0`) | **IDENTICAL** |
| | Auxiliary Loss | None ($0.0$) | Layer-aggregated $\mathcal{L}_{\text{aux}}$ | **REPAIRED** |
| | Max Epochs | 50 | 50 | **IDENTICAL** |
| | Early Stopping | 10 epochs patience on Val mIoU | 10 epochs patience on Val mIoU | **IDENTICAL** |

---

## 3. Sample Configuration Files

### Vanilla Configuration (`configs/phase9_auxloss_corrected/swin_vanilla_seed42.yaml`):
```yaml
model:
  backbone: microsoft/swin-tiny-patch4-window7-224
  name: swin_vanilla_seed42
  num_classes: 3
training:
  batch_size: 16
  early_stopping_patience: 10
  img_size: 224
  learning_rate: 0.0001
  num_epochs: 50
  seed: 42
  weight_decay: 0.05
```

### Repaired MoE Configuration (`configs/phase9_auxloss_corrected/swin_standard_moe_seed42.yaml`):
```yaml
model:
  backbone: microsoft/swin-tiny-patch4-window7-224
  name: swin_standard_moe_seed42
  num_classes: 3
moe:
  balance_loss_coef: 0.01
  capacity_factor: 1.2
  noisy_gating: true
  num_experts: 4
  top_k: 2
  warm_init: true
training:
  batch_size: 16
  early_stopping_patience: 10
  img_size: 224
  learning_rate: 0.0001
  num_epochs: 50
  seed: 42
  weight_decay: 0.05
```

---

## 4. Execution Workflow

The complete multi-seed training and evaluation pipeline is contained in:  
`Phase9_AuxLoss_Corrected_Reproducibility.ipynb`

### Kaggle Execution Steps:
1. Open Kaggle and create a new notebook with GPU accelerator (Dual T4 or P100).
2. Attach datasets: `busi-dataset` and `busbra-dataset`.
3. Import `Phase9_AuxLoss_Corrected_Reproducibility.ipynb`.
4. Run all cells:
   - Cell 1 clones repository and verifies diagnostic tests.
   - Cell 2 mounts datasets.
   - Cell 3 trains the matched models.
   - Cell 4 runs zero-shot inference, computes foreground/macro metrics, generates CSVs, and zips the results into `results_phase9_auxloss_corrected.zip`.

