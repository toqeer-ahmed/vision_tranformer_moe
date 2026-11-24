import os
import yaml

out_dir = "configs/phase10_unet"
os.makedirs(out_dir, exist_ok=True)

seeds = [42, 123, 314, 777, 999, 1001, 2026, 2027, 2718, 9001]

for s in seeds:
    cfg = {
        "model": {
            "name": f"unet_seed{s}",
            "architecture": "unet_canonical_2d",
            "in_channels": 3,
            "num_classes": 3,
            "base_channels": 64,
            "bilinear": False
        },
        "training": {
            "batch_size": 16,
            "early_stopping_patience": 10,
            "img_size": 224,
            "learning_rate": 0.0001,
            "num_epochs": 50,
            "seed": s,
            "weight_decay": 0.05
        }
    }
    path = os.path.join(out_dir, f"unet_seed{s}.yaml")
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

print(f"Generated {len(seeds)} U-Net configuration files in {out_dir}")
