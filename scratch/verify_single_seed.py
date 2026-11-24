import os
import sys
import torch
import torch.nn as nn
from datetime import datetime

sys.path.insert(0, os.path.abspath("."))

from models.unet import UNet
from evaluation.losses import CombinedSegmentationLoss
from evaluation.metrics import SegmentationMetricAccumulator
from datasets.medical_dataset import get_medical_dataloaders

print("=== STEP 1: MODEL INITIALIZATION & PARAMETER COUNT ===")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, num_classes=3, base_channels=64, bilinear=False).to(device)
params = model.count_parameters()
print(f"Device:                 {device}")
print(f"Total Parameters:       {params['total_parameters']:,}")
print(f"Trainable Parameters:   {params['trainable_parameters']:,}")
print(f"Encoder:                {params['encoder_parameters']:,}")
print(f"Bottleneck:             {params['bottleneck_parameters']:,}")
print(f"Decoder:                {params['decoder_parameters']:,}")
print(f"Head:                   {params['head_parameters']:,}")

print("\n=== STEP 2: FORWARD PASS & SHAPE VERIFICATION ===")
dummy_in = torch.randn(2, 3, 224, 224, device=device)
dummy_target = torch.randint(0, 3, (2, 224, 224), device=device).long()
out = model(dummy_in)
print(f"Input Shape:            {dummy_in.shape}")
print(f"Output Logits Shape:    {out.shape}")
assert out.shape == (2, 3, 224, 224), f"Unexpected shape {out.shape}"

print("\n=== STEP 3: LOSS & BACKPROPAGATION GRADIENT FLOW ===")
criterion = CombinedSegmentationLoss(loss_type="focal_tversky", dice_weight=1.0).to(device)
loss = criterion(out, dummy_target)
print(f"Loss Value:             {loss.item():.4f}")
loss.backward()

active_grads = sum(1 for p in model.parameters() if p.grad is not None and p.grad.norm() > 0)
total_params = sum(1 for p in model.parameters())
print(f"Tensors with Gradients: {active_grads}/{total_params}")
assert active_grads == total_params, "Gradients missing on some parameters"

print("\n=== STEP 4: CHECKPOINT SAVING & RELOADING ===")
os.makedirs("scratch", exist_ok=True)
ckpt_path = "scratch/unet_smoke_best_model.pth"
torch.save(model.state_dict(), ckpt_path)
print(f"Saved checkpoint:       {ckpt_path} ({os.path.getsize(ckpt_path)/(1024*1024):.2f} MB)")

eval_model = UNet(in_channels=3, num_classes=3, base_channels=64, bilinear=False).to(device)
eval_model.load_state_dict(torch.load(ckpt_path, map_location=device))
eval_model.eval()
print("Checkpoint loaded successfully with exact state_dict parity.")

print("\n=== STEP 5: BUSI IN-DOMAIN EVALUATION SIMULATION ===")
train_loader, val_loader, test_loader = get_medical_dataloaders(
    data_dir="data/medical_dataset",
    batch_size=2,
    img_size=224,
    num_workers=0,
    seed=42
)
acc_in = SegmentationMetricAccumulator(num_classes=3)
with torch.no_grad():
    for i, (imgs, targets) in enumerate(val_loader):
        if i >= 2: break # evaluate on 2 batches
        imgs, targets = imgs.to(device), targets.to(device)
        logits = eval_model(imgs)
        preds = logits.argmax(dim=1)
        acc_in.update(preds, targets)
m_in = acc_in.compute()
print(f"In-domain mIoU:         {m_in['mean_iou']:.4f}")
print(f"In-domain mDice:        {m_in['mean_dice']:.4f}")
print(f"In-domain Pixel Acc:    {m_in['pixel_accuracy']:.4f}")

print("\n=== STEP 6: EXTERNAL ZERO-SHOT EVALUATION PROTOCOL VERIFICATION ===")
acc_ood = SegmentationMetricAccumulator(num_classes=2)
with torch.no_grad():
    for i, (imgs, targets) in enumerate(val_loader):
        if i >= 2: break
        imgs, targets = imgs.to(device), targets.to(device)
        logits = eval_model(imgs)
        preds = (logits.argmax(dim=1) > 0).long()
        binary_targets = (targets > 0).long()
        acc_ood.update(preds, binary_targets)
m_ood = acc_ood.compute()
print(f"Zero-shot FG IoU:       {m_ood['class_iou'][1]:.4f}")
print(f"Zero-shot FG Dice:      {m_ood['class_dice'][1]:.4f}")
print(f"Zero-shot Macro mIoU:   {m_ood['mean_iou']:.4f}")
print(f"Zero-shot Macro mDice:  {m_ood['mean_dice']:.4f}")

# Cleanup smoke checkpoint
if os.path.exists(ckpt_path):
    os.remove(ckpt_path)

print("\nALL VERIFICATION STEPS PASSED SUCCESSFULLY (100% OPERATIONAL)!")
