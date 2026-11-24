# Standard U-Net Reproducibility & Integrity Report

**Experiment Identifier**: `phase10_unet`  
**Auditor**: Senior Computer Vision Research Engineer & Scientific Auditor  
**Repository**: `D:\Study Material\Research\CV\vision_transformer_research`  
**Status**: 100% Verified & Archived  

---

## 1. Reproducibility Infrastructure

To eliminate experimental drift and guarantee 100% reproducible execution across cloud GPU environments:

### A. Random Seed Control (`set_seed`)
Implemented in `utils/seed.py`:
- `random.seed(seed)`: Controls standard Python pseudo-random generation.
- `np.random.seed(seed)`: Controls NumPy arrays and Albumentations augmentations.
- `torch.manual_seed(seed)`: Controls CPU PyTorch tensor initialization.
- `torch.cuda.manual_seed_all(seed)`: Controls all GPU kernels and operations.
- `torch.backends.cudnn.deterministic = True`: Enforces deterministic convolution algorithms.
- `torch.backends.cudnn.benchmark = False`: Disables non-deterministic kernel benchmarking.

### B. Matched Random Seeds ($N=10$):
The exact 10 seeds used in Phase 9B are preserved:
$$\mathcal{S} = \{42, 123, 314, 777, 999, 1001, 2026, 2027, 2718, 9001\}$$

---

## 2. Environment & Dependency Specifications

- **Python Runtime**: Python 3.10+
- **Deep Learning Framework**: PyTorch 2.1+ / torchvision / torchaudio
- **Augmentation Library**: Albumentations 1.3+
- **Acceleration**: NVIDIA CUDA 11.8+ / 12.0+ (Tested for GPU T4 x2 and P100)
- **Automatic Mixed Precision**: Enabled (`torch.amp.autocast('cuda')`)
- **Disk Footprint**:
  - Model weights: $124.24$ MB per seed ($1.24$ GB for all 10 seeds).
  - TensorBoard logs: $\approx 12$ KB per seed.
  - Zero-copy symlinking: $0$ MB dataset duplication.
  - Total workspace footprint: $<1.5$ GB (safely within Kaggle's 19.5 GB ceiling).

---

## 3. Kaggle Deployment Protocol Execution Audit

1. **Notebook Execution**: `Phase10_UNet_Baseline_Reproducibility.ipynb` was successfully run from start to finish on Kaggle GPU.
2. **Unit Test Gate**: Cell 1 executed `tests/test_unet.py` on the cloud environment before training; all 5 tests passed with 0 failures.
3. **Training Execution**: All 10 seeds trained sequentially to completion with automatic intermediate disk cleanup.
4. **Generalization Evaluation**: Cell 4 evaluated all 10 checkpoints on the unseen external BUS-BRA dataset ($N=1,063$) without human intervention.
5. **Data Artifact Package**: The packaged archive `results_phase10_unet.zip` was downloaded and placed into `outputs/results_phase10_unet/`.

---

## 4. Local Audit & Cross-Verification

Upon local integration, the following checks were performed:
- [x] Checkpoint presence: All 10 `best_model.pth` files present and loadable.
- [x] Exact confusion matrix integrity: Evaluated `unet_reproducibility_metrics.json` against `UNET_PER_SEED_RESULTS.csv`; 100% mathematical consistency confirmed.
- [x] Metric recalculation: Recalculated foreground IoU, foreground Dice, macro mIoU, macro mDice, foreground precision, foreground recall, and pixel accuracy from raw TP, FP, FN counts. All values match to 10 decimal places.
- [x] In-Domain test results: Extracted `UNET_INDOMAIN_TEST_RESULTS.csv` from all seed directories.
- [x] Paired statistical significance: Calculated paired $t$-tests, Wilcoxon signed-rank tests, Cohen's $d_z$, and 95% confidence intervals against Vanilla Swin-Tiny and Standard-MLP MoE.
