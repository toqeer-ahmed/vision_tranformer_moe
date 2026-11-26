import os
import sys
import glob
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add repo to sys.path
repo_dir = r'D:\Study Material\Research\CV\vision_transformer_research'
sys.path.insert(0, repo_dir)
from models.attention_unet import AttentionUNet

# Paths
busi_raw_dir = os.path.join(repo_dir, 'datasets', 'extracted_busi', 'Dataset_BUSI_with_GT')
busbra_images_dir = os.path.join(repo_dir, 'datasets', 'extracted_busbra', 'BUSBRA', 'BUSBRA', 'Images')
busbra_masks_dir = os.path.join(repo_dir, 'datasets', 'extracted_busbra', 'BUSBRA', 'BUSBRA', 'Masks')
outputs_base = os.path.join(repo_dir, 'results', 'results_phase11_attention_unet', 'outputs')

# Transforms
val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# 1. Helper to load BUSI dataset pairs
def get_busi_pairs():
    pairs = []
    for cat in ['benign', 'malignant', 'normal']:
        cat_dir = os.path.join(busi_raw_dir, cat)
        if not os.path.exists(cat_dir):
            continue
        all_files = os.listdir(cat_dir)
        img_files = [f for f in all_files if not ('mask' in f.lower() or '_gt' in f.lower())]
        for img_f in img_files:
            stem = os.path.splitext(img_f)[0]
            # find mask
            mask_matches = [f for f in all_files if f.startswith(stem) and ('mask' in f.lower() or '_gt' in f.lower())]
            if mask_matches:
                pairs.append((os.path.join(cat_dir, img_f), os.path.join(cat_dir, mask_matches[0])))
    pairs.sort(key=lambda x: x[0])
    return pairs

# 2. Helper to load BUS-BRA pairs
def get_busbra_pairs(max_count=200):
    pairs = []
    img_files = sorted(os.listdir(busbra_images_dir))
    for img_f in img_files:
        stem = os.path.splitext(img_f)[0]
        # bus_XXXX -> mask_XXXX
        mask_stem = stem.replace('bus_', 'mask_', 1)
        mask_f = f'{mask_stem}.png'
        mask_path = os.path.join(busbra_masks_dir, mask_f)
        if os.path.exists(mask_path):
            pairs.append((os.path.join(busbra_images_dir, img_f), mask_path))
        if len(pairs) >= max_count:
            break
    return pairs

busi_pairs = get_busi_pairs()
print(f'Total BUSI pairs loaded: {len(busi_pairs)}')
busbra_pairs = get_busbra_pairs(max_count=200)
print(f'Total BUS-BRA pairs loaded: {len(busbra_pairs)}')

# 3. Model loading helper
def load_model(seed):
    model = AttentionUNet(in_channels=3, num_classes=3, base_channels=64, bilinear=False)
    # find run directory
    pattern = os.path.join(outputs_base, f'attention_unet_seed{seed}_*')
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        raise FileNotFoundError(f'No directory for seed {seed}')
    ckpt_path = os.path.join(dirs[-1], 'checkpoints', 'best_model.pth')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model

# Seeds to investigate
target_seeds = [42, 123, 2026, 2718]

# 4. Activation Analysis Function
def analyze_activations(model, pairs, seed, dataset_name):
    # Store gate stats across all images
    gate_data = {1: [], 2: [], 3: [], 4: []}
    skip_norms_pre = {1: [], 2: [], 3: [], 4: []}
    skip_norms_post = {1: [], 2: [], 3: [], 4: []}

    with torch.no_grad():
        for img_path, mask_path in pairs:
            img = np.array(Image.open(img_path).convert('RGB'))
            mask = np.array(Image.open(mask_path).convert('L'))
            mask_bin = (mask > 127).astype(np.float32)

            augmented = val_transform(image=img, mask=mask_bin)
            x_tensor = augmented['image'].unsqueeze(0) # (1, 3, 224, 224)

            # Forward pass intermediate tracking
            # Replicate forward to capture skip features x1, x2, x3, x4
            x1 = model.inc(x_tensor)
            x2 = model.down1(x1)
            x3 = model.down2(x2)
            x4 = model.down3(x3)
            x5 = model.down4(x4)

            # Up blocks with attention gates
            # Gate 1: Up1 (x5, x4)
            d1, a1 = model.up1(x5, x4)
            # Gate 2: Up2 (d1, x3)
            d2, a2 = model.up2(d1, x3)
            # Gate 3: Up3 (d2, x2)
            d3, a3 = model.up3(d2, x2)
            # Gate 4: Up4 (d3, x1)
            d4, a4 = model.up4(d3, x1)

            # Store attention maps (flattened numpy)
            for g_idx, alpha, x_s in [(1, a1, x4), (2, a2, x3), (3, a3, x2), (4, a4, x1)]:
                a_np = alpha.squeeze().cpu().numpy().flatten()
                gate_data[g_idx].append(a_np)

                # Skip norm
                pre_norm = torch.norm(x_s, p=1).item() / x_s.numel()
                post_norm = torch.norm(x_s * alpha, p=1).item() / x_s.numel()
                skip_norms_pre[g_idx].append(pre_norm)
                skip_norms_post[g_idx].append(post_norm)

    # Compute summary statistics for each gate
    summary_rows = []
    for g_idx in [1, 2, 3, 4]:
        all_alphas = np.concatenate(gate_data[g_idx])
        mean_val = float(np.mean(all_alphas))
        std_val = float(np.std(all_alphas))
        median_val = float(np.median(all_alphas))
        q25 = float(np.percentile(all_alphas, 25))
        q75 = float(np.percentile(all_alphas, 75))
        frac_below_05 = float(np.mean(all_alphas < 0.05))
        frac_below_10 = float(np.mean(all_alphas < 0.10))
        frac_above_90 = float(np.mean(all_alphas > 0.90))
        frac_above_95 = float(np.mean(all_alphas > 0.95))

        pre_n = float(np.mean(skip_norms_pre[g_idx]))
        post_n = float(np.mean(skip_norms_post[g_idx]))
        supp_ratio = post_n / pre_n if pre_n > 0 else 0.0

        summary_rows.append({
            'Seed': seed,
            'Dataset': dataset_name,
            'Gate': f'Gate_{g_idx}',
            'Mean': mean_val,
            'Std': std_val,
            'Median': median_val,
            'Q25': q25,
            'Q75': q75,
            'Frac_below_0.05': frac_below_05,
            'Frac_below_0.10': frac_below_10,
            'Frac_above_0.90': frac_above_90,
            'Frac_above_0.95': frac_above_95,
            'Pre_Skip_Norm': pre_n,
            'Post_Skip_Norm': post_n,
            'Feature_Transmission_Ratio': supp_ratio
        })
    return summary_rows, gate_data

all_results = []
collected_gate_data = {}

for s in target_seeds:
    print(f'\n--- Loading Model Seed {s} ---')
    model = load_model(s)

    # 1. BUSI test split (seed-controlled split: 80% train, 10% val, 10% test = 78 images)
    train_len = int(0.8 * len(busi_pairs))
    val_len = int(0.1 * len(busi_pairs))
    test_len = len(busi_pairs) - train_len - val_len
    # Reproduce generator split
    g = torch.Generator().manual_seed(s)
    indices = torch.randperm(len(busi_pairs), generator=g).tolist()
    test_indices = indices[train_len + val_len:]
    busi_test_pairs = [busi_pairs[i] for i in test_indices]

    print(f'Running Seed {s} on BUSI Test ({len(busi_test_pairs)} images)...')
    rows_busi, gd_busi = analyze_activations(model, busi_test_pairs, s, 'BUSI_Test')
    all_results.extend(rows_busi)
    collected_gate_data[(s, 'BUSI_Test')] = gd_busi

    print(f'Running Seed {s} on BUS-BRA ({len(busbra_pairs)} images)...')
    rows_busbra, gd_busbra = analyze_activations(model, busbra_pairs, s, 'BUS-BRA_ZeroShot')
    all_results.extend(rows_busbra)
    collected_gate_data[(s, 'BUS-BRA_ZeroShot')] = gd_busbra

df_act = pd.DataFrame(all_results)
out_csv = os.path.join(repo_dir, 'research_audit', 'phase11_attention_unet_final', 'ATTENTION_GATE_ACTIVATION_PROFILES.csv')
df_act.to_csv(out_csv, index=False)
print(f'\nSaved activation profile CSV to {out_csv}')

# Print clean comparison table
print('\n=== ATTENTION GATE ACTIVATION SUMMARY TABLE ===')
disp_cols = ['Seed', 'Dataset', 'Gate', 'Mean', 'Median', 'Frac_below_0.10', 'Frac_above_0.90', 'Feature_Transmission_Ratio']
print(df_act[disp_cols].to_string(index=False))

# Generate Publication Plot of Gate Activation Histograms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plot_dir = os.path.join(repo_dir, 'research_audit', 'phase11_attention_unet_final', 'plots')
os.makedirs(plot_dir, exist_ok=True)

fig, axes = plt.subplots(4, 2, figsize=(12, 14), dpi=300)
colors_busi = '#1f77b4'
colors_busbra = '#d62728'

for g_idx in [1, 2, 3, 4]:
    # Row g_idx - 1: Left = Seed 42 (Collapsed), Right = Seed 2026 (Healthy)
    ax_left = axes[g_idx - 1, 0]
    ax_right = axes[g_idx - 1, 1]

    # Seed 42
    s42_busi = np.concatenate(collected_gate_data[(42, 'BUSI_Test')][g_idx])
    s42_busbra = np.concatenate(collected_gate_data[(42, 'BUS-BRA_ZeroShot')][g_idx])
    ax_left.hist(s42_busi, bins=40, density=True, alpha=0.55, color=colors_busi, label='BUSI Test (In-Domain)')
    ax_left.hist(s42_busbra, bins=40, density=True, alpha=0.55, color=colors_busbra, label='BUS-BRA (Zero-Shot)')
    ax_left.set_title(f'Seed 42 (Collapsed) — Gate {g_idx}', fontsize=11, fontweight='bold')
    ax_left.set_xlabel('Gate Activation (Alpha)', fontsize=9)
    ax_left.set_ylabel('Density', fontsize=9)
    ax_left.legend(fontsize=8, loc='upper right')

    # Seed 2026
    s2026_busi = np.concatenate(collected_gate_data[(2026, 'BUSI_Test')][g_idx])
    s2026_busbra = np.concatenate(collected_gate_data[(2026, 'BUS-BRA_ZeroShot')][g_idx])
    ax_right.hist(s2026_busi, bins=40, density=True, alpha=0.55, color=colors_busi, label='BUSI Test (In-Domain)')
    ax_right.hist(s2026_busbra, bins=40, density=True, alpha=0.55, color=colors_busbra, label='BUS-BRA (Zero-Shot)')
    ax_right.set_title(f'Seed 2026 (Healthy) — Gate {g_idx}', fontsize=11, fontweight='bold')
    ax_right.set_xlabel('Gate Activation (Alpha)', fontsize=9)
    ax_right.set_ylabel('Density', fontsize=9)
    ax_right.legend(fontsize=8, loc='upper right')

plt.suptitle('Attention Gate Activation Distributions: In-Domain BUSI vs Zero-Shot BUS-BRA\nContrasting Seed 42 (Severe Volume Collapse) vs Seed 2026 (Healthy Generalization)', fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout()
fig_path = os.path.join(plot_dir, 'fig6_attention_gate_activation_histograms.png')
plt.savefig(fig_path)
plt.close()
print(f'Saved histogram plot to {fig_path}')
