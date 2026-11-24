# Attention U-Net: Kaggle Deployment & Reproducibility Plan

**Experiment Identifier**: `phase11_attention_unet`  
**Auditor**: Senior Computer Vision Research Engineer & Scientific Auditor  
**Repository**: `D:\Study Material\Research\CV\vision_transformer_research`  
**Date**: 2026-11-25  

---

## 1. Cloud Execution Architecture

To ensure 100% reproducible execution and eliminate disk space constraints, the entire Phase 11 experiment is packaged into an automated cloud notebook:
- **Notebook Path in Repo**: [Phase11_Attention_UNet_Reproducibility.ipynb](file:///D:/Study%20Material/Research/CV/vision_transformer_research/Phase11_Attention_UNet_Reproducibility.ipynb)
- **Alternate Path**: [notebooks/Phase11_Attention_UNet_Reproducibility.ipynb](file:///D:/Study%20Material/Research/CV/vision_transformer_research/notebooks/Phase11_Attention_UNet_Reproducibility.ipynb)

---

## 2. Step-by-Step Kaggle Deployment Instructions

### Step 1: Upload Notebook to Kaggle
1. Navigate to [Kaggle Notebooks](https://www.kaggle.com/code).
2. Click **New Notebook** -> **File** -> **Upload Notebook**.
3. Select `Phase11_Attention_UNet_Reproducibility.ipynb` from your local workspace.

### Step 2: Configure Environment & Hardware
1. In the right-hand panel under **Settings**:
   - **Accelerator**: Select **GPU T4 x2** (or **GPU P100**).
   - **Internet**: Set to **On** (required to clone latest repository and install packages).
   - **Persistence**: No special configuration needed.

### Step 3: Attach Attached Datasets
Click **+ Add Input** in the top right and attach:
1. `busi-dataset` (or search for Breast Ultrasound Images Dataset).
2. `busbra-dataset` (or search for BUS-BRA dataset).

### Step 4: Run the Experiment
Click **Run All** (or `Shift + Enter` through cells sequentially):
- **Cell 1**: Clones the latest `main` branch from GitHub and executes the 12-test assertion suite (`tests/test_attention_unet.py`).
- **Cell 2**: Performs zero-copy symbolic linking of BUSI and BUS-BRA images and masks into `data/`.
- **Cell 3**: Sequentially trains Attention U-Net across all 10 matched seeds ($\mathcal{S} = \{42, 123, 314, 777, 999, 1001, 2026, 2027, 2718, 9001\}$). Automatically removes intermediate `last_checkpoint.pth` files to maintain $>15\text{ GB}$ disk headroom.
- **Cell 4**: Evaluates all 10 trained `best_model.pth` checkpoints on unseen external BUS-BRA ($N = 1,063$), calculates volume ratios, outputs per-seed and aggregate summary CSVs, and packages all artifacts into `results_phase11_attention_unet.zip`.

---

## 3. Post-Run Local Integration Protocol

Once Kaggle execution finishes (~1.5 to 2 hours total runtime):
1. Download `results_phase11_attention_unet.zip` from the Kaggle Output file explorer.
2. Extract the archive into:
   `D:\Study Material\Research\CV\vision_transformer_research\outputs\results_phase11_attention_unet\`
3. Run the Phase 11 statistical audit and cross-comparison against Phase 9B (ViTs) and Phase 10 (Standard U-Net).
