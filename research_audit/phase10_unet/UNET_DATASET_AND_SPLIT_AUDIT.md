# Dataset & Split Protocol Audit: Standard U-Net Baseline

**Experiment Identifier**: `phase10_unet`  
**Auditor**: Senior Computer Vision Research Engineer & Scientific Auditor  
**Repository**: `D:\Study Material\Research\CV\vision_transformer_research`  

---

## 1. Dataset Manifests & Verification

### A. Training & In-Domain Evaluation: BUSI Dataset
- **Clinical Origin**: Breast Ultrasound Images Dataset (BUSI), collected from 600 female patients at Baheya Hospital, Cairo, Egypt.
- **Total Images**: $780$ ultrasound images categorized into:
  - Normal cases: $133$ images
  - Benign cases: $437$ images
  - Malignant cases: $210$ images
- **Resolution**: Variable acquisition sizes ($\sim 500 \times 500$). Resized uniformly to $224 \times 224$.
- **Image-Mask Pairing Protocol**:
  - Implemented in `datasets/medical_dataset.py#MedicalImageMaskDataset`.
  - Matches input ultrasound images (`data/medical_dataset/images/*.png`) with ground truth segmentation masks (`data/medical_dataset/masks/*_mask*.png`).
  - Stems verified: every image maps to exactly one verified binary/multi-class mask target.

### B. Out-of-Distribution Zero-Shot Benchmark: BUS-BRA Dataset
- **Clinical Origin**: Breast Ultrasound Bi-Rads Assessment dataset (BUS-BRA), acquired in Brazil across different sonographic hardware.
- **Total Images**: $1,063$ clinical ultrasound images.
- **Label Mapping**: Binary lesion masks (0: background parenchymal tissue, 1: tumor).
- **Strict Blind Isolation Rule**:
  - BUS-BRA is evaluated **strictly zero-shot**.
  - No BUS-BRA image or label is ever exposed to the model during training, validation, early stopping, or learning rate scheduling.

---

## 2. Seed-Controlled Partitioning Protocol

To guarantee matched pairs with Phase 9B experiments, the dataset splitting utilizes an exact mathematical replica of the audited protocol:

```python
train_len = int(0.8 * len(full_dataset))  # 624 images (80%)
val_len = int(0.1 * len(full_dataset))    # 78 images (10%)
test_len = len(full_dataset) - train_len - val_len  # 78 images (10%)

train_subset, val_subset, test_subset = random_split(
    full_dataset, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(seed)
)
```

### Partition Properties:
1. **Seed Reproducibility**:
   Because `random_split` uses `torch.Generator().manual_seed(seed)`, every run of `unet_seed{s}` receives the **identical 624/78/78 subset of images** as `swin_vanilla_seed{s}` and `swin_standard_moe_seed{s}`.
2. **Patient Isolation**:
   Evaluations are performed exclusively on the held-out in-domain test set ($N=78$) and the external BUS-BRA benchmark ($N=1,063$).

---

## 3. Data Augmentation & Preprocessing Pipeline

The augmentation pipeline is shared with the audited Phase 9B experiments to prevent data preprocessing from confounding model comparison:

### A. Training Transforms (`train_transform`):
- `A.Resize(224, 224)`: Standard input resolution.
- `A.HorizontalFlip(p=0.5)`: Lateral tissue invariance.
- `A.VerticalFlip(p=0.5)`: Orientation invariance.
- `A.RandomRotate90(p=0.5)`: Orthogonal transducer rotation.
- `A.Affine(scale=(0.85, 1.15), rotate=(-30, 30), translate_percent=(-0.1, 0.1), p=0.5)`: Probe tilt and scaling.
- `A.ElasticTransform(alpha=1, sigma=50, p=0.3)`: Non-rigid breast tissue deformation.
- `A.GridDistortion(p=0.3)`: Acoustic lens distortion simulation.
- `A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.3)`: Ultrasound gain and dynamic range variation.
- `A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))`: Standard ImageNet normalization.
- `ToTensorV2()`: Conversion to PyTorch float32 tensor.

### B. Validation & Zero-Shot Testing Transforms (`val_transform`):
- `A.Resize(224, 224)`
- `A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))`
- `ToTensorV2()`: Deterministic evaluation without stochastic distortion.
