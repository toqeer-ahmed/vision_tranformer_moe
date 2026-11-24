# Standard U-Net Reproducibility & Integrity Report

**Experiment Identifier**: `phase10_unet`  
**Auditor**: Senior Computer Vision Research Engineer & Scientific Auditor  
**Repository**: `D:\Study Material\Research\CV\vision_transformer_research`  

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
  - Model weights: $118.49$ MB per seed ($1.18$ GB for all 10 seeds).
  - TensorBoard logs: $\approx 25$ MB.
  - Zero-copy symlinking: $0$ MB dataset duplication.
  - Total workspace footprint: $<2.5$ GB (safely within Kaggle's 19.5 GB ceiling).

---

## 3. Kaggle Deployment Protocol

1. **Upload Notebook**:
   Upload `Phase10_UNet_Baseline_Reproducibility.ipynb` to Kaggle.
2. **Attach Datasets**:
   - `busi-dataset`
   - `busbra-dataset`
3. **Hardware Selection**:
   - Accelerator: **GPU T4 x2** or **P100**.
   - Internet: **On**.
4. **Execution Flow**:
   - Cell 1: Environment setup, repository clone from `main`, automated unit test execution (`python tests/test_unet.py`).
   - Cell 2: Symbolic linking of BUSI and BUS-BRA.
   - Cell 3: 10-seed training loop with post-seed intermediate checkpoint cleanup.
   - Cell 4: Zero-shot evaluation on external BUS-BRA ($N=1,063$), summary CSV generation, cross-experiment benchmark display, and zip packaging into `results_phase10_unet.zip`.
