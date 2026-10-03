# Phase 9: Standard-MLP Swin-MoE Kaggle Experiment

## Objective
To isolate the effect of conditional routing in Swin-MoE by removing the spatial depthwise-convolution from the experts. This experiment uses a `StandardMLPExpert` (Linear -> GELU -> Linear) which exactly matches Swin's native FFN architecture.

## Files Modified/Created
1. `models/moe/standard_swin_moe_layer.py`: Added `StandardSwinMoELayer` which uses `SimpleMLPBlock` instead of `SpatiallyAwareExpert`.
2. `models/swin.py`: Added `replace_swin_ffn_with_standard_moe()` with correct **Warm Initialization** logic that copies weights directly from the native Swin FFN to all experts.
3. `training/train_swin_standard_moe.py`: A dedicated training script for this experiment.

---

## 1. Parameter Accounting
*Note: Run `python count_params_phase9.py` locally to populate exact counts before running.*

| Model | Expert type | Experts | Parameters | Relative increase |
| ----- | ----------- | ------- | ---------- | ----------------- |
| Vanilla Swin | N/A | 1 (Native FFN) | 29,290,365 | Baseline |
| Spatial Swin-MoE | `SpatiallyAwareExpert` (3x3 DWConv) | 4 | 81,855,549 | ~2.8x |
| Standard-MLP Swin-MoE | `SimpleMLPBlock` (No DWConv) | 4 | 81,148,989 | ~2.77x |

---

## 2. Kaggle Setup Instructions
1. Open Kaggle and create a new Notebook.
2. Select **GPU T4 x2** (or P100) as the accelerator.
3. Add the following datasets to your Kaggle environment:
   - `busi-dataset`
   - `busbra-dataset`
4. Make sure Internet is turned **ON** in the Kaggle session so it can download the Swin-Tiny backbone.
5. Create a new cell and copy-paste the entire code block below into it.

---

## 3. The Kaggle Training Cell

```python
# ==========================================
# STEP 1: ENVIRONMENT SETUP
# ==========================================
import os
import sys

# Clone the repository
!git clone https://github.com/toqeer-ahmed/vision_tranformer_moe.git vision_transformer_research
os.chdir('vision_transformer_research')
sys.path.append(os.path.abspath('.'))

# Install requirements
!pip install -r requirements.txt -q

# ==========================================
# STEP 2: RESTRUCTURE DATASETS (KAGGLE SPECIFIC)
# ==========================================
import shutil
print("Setting up dataset paths...")
os.makedirs("data/medical_dataset/images", exist_ok=True)
os.makedirs("data/medical_dataset/masks", exist_ok=True)

# Copy BUSI
for file in os.listdir("/kaggle/input/busi-dataset/images"):
    shutil.copy(f"/kaggle/input/busi-dataset/images/{file}", f"data/medical_dataset/images/{file}")
for file in os.listdir("/kaggle/input/busi-dataset/masks"):
    shutil.copy(f"/kaggle/input/busi-dataset/masks/{file}", f"data/medical_dataset/masks/{file}")

# ==========================================
# STEP 3: CONFIGURE EXPERIMENT
# ==========================================
config_yaml = """
model:
  name: "swin_standard_moe"
  backbone: "microsoft/swin-tiny-patch4-window7-224"
  num_classes: 3

moe:
  num_experts: 4
  top_k: 2
  noisy_gating: true
  balance_loss_coef: 0.01
  warm_init: true

training:
  batch_size: 32
  learning_rate: 0.0001
  weight_decay: 0.01
  num_epochs: 50
  early_stopping_patience: 10
  img_size: 224
  seed: 42
"""
with open("configs/swin_standard_moe_kaggle.yaml", "w") as f:
    f.write(config_yaml)

# ==========================================
# STEP 4: SMOKE TEST (1 BATCH)
# ==========================================
print("Running Smoke Test...")
import torch
import yaml
from models.swin import SwinSegmentation, replace_swin_ffn_with_standard_moe
from datasets.medical_dataset import get_medical_dataloaders
from evaluation.losses import CombinedSegmentationLoss

with open("configs/swin_standard_moe_kaggle.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Initialize Model and Inject MoE
model = SwinSegmentation(num_classes=3, pretrained=True)
replace_swin_ffn_with_standard_moe(model, config["moe"])
model.cuda()

# Get 1 Batch
train_loader, _, _ = get_medical_dataloaders("data/medical_dataset", batch_size=2, img_size=224, seed=42)
images, targets = next(iter(train_loader))
images, targets = images.cuda(), targets.cuda()

# Forward, Loss, Backward
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = CombinedSegmentationLoss(loss_type="focal_tversky").cuda()

with torch.amp.autocast('cuda'):
    logits = model(images)
    loss = criterion(logits, targets)

loss.backward()
optimizer.step()

print(f"Smoke Test Passed! Initial Loss: {loss.item():.4f}")
print("Gradients successfully flowed through Standard-MLP MoE layers.")
del model, optimizer, criterion, images, targets
torch.cuda.empty_cache()

# ==========================================
# STEP 5: FULL TRAINING
# ==========================================
print("Starting Full Training...")
!python training/train_swin_standard_moe.py --config configs/swin_standard_moe_kaggle.yaml --warm-init

# ==========================================
# STEP 6: ZERO-SHOT EVALUATION ON BUS-BRA
# ==========================================
print("Setting up BUS-BRA for Zero-Shot Evaluation...")
os.makedirs("data/busbra_dataset/images", exist_ok=True)
os.makedirs("data/busbra_dataset/masks", exist_ok=True)

# Note: The evaluation script must handle the pairing correctly.
!python scripts/evaluate_zero_shot.py --model swin_standard_moe --checkpoint outputs/swin_standard_moe_*/checkpoints/best_model.pth --dataset busbra --data_dir /kaggle/input/busbra-dataset
```

---

## 4. Expected Output & Download
After the cell finishes executing, Kaggle will have created:
1. `/kaggle/working/vision_transformer_research/outputs/swin_standard_moe_<timestamp>/`
2. `checkpoints/best_model.pth`
3. `logs/test_metrics.json`
4. `/kaggle/working/vision_transformer_research/outputs/zero_shot_results/zero_shot_metrics.json`

**To download:**
In the right-hand panel of Kaggle, navigate to `/kaggle/working/vision_transformer_research/outputs/` and click the three dots next to the folders to download them back to your local machine.

---

## 5. Required Final Metrics
The evaluation scripts already produce exactly what you requested:
- foreground IoU (mIoU)
- Dice (mDice)
- precision
- recall
- pixel accuracy

The final comparison table to be populated post-experiment is:

| Model | Params | mIoU | Dice | Precision | Recall | Pixel Acc |
| ----- | ------ | ---- | ---- | --------- | ------ | --------- |
| Vanilla Swin | 29.3M | ... | ... | ... | ... | ... |
| Spatial Swin-MoE | 81.9M | ... | ... | ... | ... | ... |
| Standard-MLP Swin-MoE | 81.1M | ... | ... | ... | ... | ... |

*(Note: The exact params will be printed by the parameter counter script).*

## 6. Interpretation Rule
As requested, DO NOT decide the conclusion yet. We will review the outputs of this Kaggle experiment to determine if the strange behavior of Swin MoE was caused by the conditional router, or by accidentally injecting a $3 \times 3$ depthwise convolution into a pure windowed-attention network.
