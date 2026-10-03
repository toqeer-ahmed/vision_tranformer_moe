import torch
import sys
import os

# Ensure local path is accessible
sys.path.append(os.path.abspath('.'))

from models.swin import SwinSegmentation, replace_swin_ffn_with_moe, replace_swin_ffn_with_standard_moe
from copy import deepcopy

print("Counting parameters for Phase 9 Models...")

# 1. Vanilla Swin
vanilla_model = SwinSegmentation(num_classes=3, pretrained=False)
vanilla_params = sum(p.numel() for p in vanilla_model.parameters())
print(f"Vanilla Swin: {vanilla_params:,}")

# 2. Spatial Swin-MoE
moe_cfg = {"num_experts": 4, "top_k": 2, "noisy_gating": True}
spatial_model = deepcopy(vanilla_model)
replace_swin_ffn_with_moe(spatial_model, moe_cfg)
spatial_params = sum(p.numel() for p in spatial_model.parameters())
print(f"Spatial Swin-MoE: {spatial_params:,}")

# 3. Standard-MLP Swin-MoE
standard_model = deepcopy(vanilla_model)
replace_swin_ffn_with_standard_moe(standard_model, moe_cfg)
standard_params = sum(p.numel() for p in standard_model.parameters())
print(f"Standard-MLP Swin-MoE: {standard_params:,}")
