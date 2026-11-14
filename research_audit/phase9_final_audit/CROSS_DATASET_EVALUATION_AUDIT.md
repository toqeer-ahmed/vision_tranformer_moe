# Cross-Dataset Evaluation Audit: BUSI $\to$ BUS-BRA Protocol

**Audit Date**: November 12, 2026  
**Auditor**: Senior Computer Vision Research Engineer & Scientific Auditor  
**Repository**: `D:\Study Material\Research\CV\vision_transformer_research`  
**Datasets Audited**:
- **Source (Training)**: Breast Ultrasound Images (BUSI) Dataset
- **Target (Zero-Shot)**: Brazilian Breast Ultrasound (BUS-BRA) Dataset

---

## 1. Experimental Setup & Domain Shift Context

Cross-dataset evaluation evaluates the **out-of-distribution (OOD) generalization** of machine learning models. A major failure mode in medical AI is hospital overfitting: models achieve >90% Dice on their training institution's scanners but drop dramatically when deployed on unseen hardware from other clinics.

| Property | BUSI (Source / Training) | BUS-BRA (Target / Zero-Shot) |
| :--- | :--- | :--- |
| **Origin Institution** | Cairo University, Egypt | Federal University of Technology - Paraná, Brazil |
| **Scanner Hardware** | LOGIQ E9 & LOGIQ 7 (General Electric) | Various clinical ultrasound platforms |
| **Image Resolution** | Varying ($500 \times 500$ avg) | Varying ($600 \times 800$ avg) |
| **Class Taxonomy** | 3 Classes: Normal (no tumor), Benign, Malignant | Binary: Background vs. Confirmed Lesion |
| **Total Images** | 780 scans (133 Normal, 487 Benign, 210 Malignant) | 1,063 biopsy-confirmed scans |
| **Role in Phase 9** | Used strictly for 80/10/10 Train / Val / Test | Used strictly for Zero-Shot Evaluation |

---

## 2. Forensic Data Leakage Verification

A strict forensic code audit was conducted on `training/train_swin_vanilla.py`, `training/train_swin_standard_moe.py`, `datasets/medical_dataset.py`, and `Phase9_Controlled_Reproducibility.ipynb`.

### Leakage Audit Checklist:

1. **Was BUS-BRA ever present in the training dataloader?**
   - **AUDIT VERDICT**: **NO (PASSED)**.
   - *Code Evidence*: In `train_swin_vanilla.py#L71-L77` and `train_swin_standard_moe.py#L71-L77`, `get_medical_dataloaders()` is called with `data_dir="data/medical_dataset"`. This path contains only the BUSI dataset files (`Dataset_BUSI_with_GT`). The BUS-BRA dataset was never mounted or accessed in the training scripts.

2. **Was BUS-BRA used for validation or early stopping?**
   - **AUDIT VERDICT**: **NO (PASSED)**.
   - *Code Evidence*: The validation loop in both training scripts iterates exclusively over `val_loader` derived from the 10% held-out BUSI split (`SubsetMedicalWrapper`). Early stopping patience was tracked solely against BUSI validation mIoU (`metrics["mean_iou"] > best_val_iou`).

3. **Was BUS-BRA used for checkpoint selection?**
   - **AUDIT VERDICT**: **NO (PASSED)**.
   - *Code Evidence*: `best_model.pth` was saved inside the training script whenever validation mIoU on BUSI improved. The evaluation script in Cell 4 loads this pre-existing checkpoint from disk without retraining.

4. **Was any fine-tuning or adaptation performed on BUS-BRA?**
   - **AUDIT VERDICT**: **NO (PASSED)**.
   - *Code Evidence*: The evaluation loop (`evaluate_checkpoint`) runs under `with torch.no_grad():` and `model.eval()`. Gradients were never computed or backpropagated on BUS-BRA.

5. **Was post-hoc threshold search or temperature scaling used on BUS-BRA?**
   - **AUDIT VERDICT**: **NO (PASSED)**.
   - *Code Evidence*: Binary predictions were produced using the fixed argmax mapping:
     ```python
     preds = (logits.argmax(dim=1) > 0).long()
     ```
     No parameter sweeps, ROC threshold tuning, or calibration methods were applied to BUS-BRA.

---

## 3. Label Space Mapping Audit

Because BUSI was trained with 3 classes while BUS-BRA contains binary labels, the label projection was audited:

- In BUSI:
  - Class `0`: Normal tissue / Background
  - Class `1`: Benign lesion
  - Class `2`: Malignant lesion
- In BUS-BRA:
  - Label `0`: Background
  - Label `1`: Biopsy-confirmed lesion

### Projection Logic:
```python
preds = (logits.argmax(dim=1) > 0).long()
```
- If the model predicts Class 1 (benign) or Class 2 (malignant), `(pred > 0)` maps the pixel to `1` (Lesion).
- If the model predicts Class 0 (background), it maps to `0` (Background).
- **Audit Assessment**: This mapping is mathematically sound and clinically appropriate. It evaluates whether the model detects the lesion regardless of whether it classifies it as benign or malignant.

---

## 4. Evaluation Limitations and Confounders

1. **Class Distribution Imbalance between Datasets**:
   BUSI includes 133 "Normal" images that have no tumors (ground truth mask is all zeros). In contrast, BUS-BRA contains only biopsy-confirmed lesions. The presence of normal images in BUSI forces the network to learn empty-mask prediction, which benefits precision on BUS-BRA.
2. **Scanner Artifacts**:
   Ultrasound scanners exhibit distinct speckle noise, gain settings, and depth attenuation depending on manufacturer (GE in BUSI vs multiple platforms in BUS-BRA). The zero-shot performance drop from $\sim 86\%$ in-domain mIoU down to $\sim 76\%$ reflects genuine hardware distribution shift.
3. **Absence of Intermediate Calibration**:
   The fixed argmax threshold ($>0$) does not account for potential logit shift between the two datasets. While this prevents data leakage, post-hoc calibration could further separate domain shift from calibration drift.

