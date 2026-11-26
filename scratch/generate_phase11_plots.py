import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

base_dir = r'D:\Study Material\Research\CV\vision_transformer_research'
audit_dir = os.path.join(base_dir, 'research_audit', 'phase11_attention_unet_final')
plot_dir = os.path.join(audit_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)

master_csv = os.path.join(audit_dir, 'FOUR_MODEL_BENCHMARK_MASTER.csv')
df = pd.read_csv(master_csv)

# 1. Plot 1: 4-Model Foreground IoU Comparison Boxplot
plt.figure(figsize=(9, 6), dpi=300)
models = ['Vanilla Swin-Tiny', 'Standard-MLP Swin-MoE', 'Standard U-Net', 'Attention U-Net']
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
col_map = {
    'Vanilla Swin-Tiny': df['Vanilla_Fg_IoU'] * 100,
    'Standard-MLP Swin-MoE': df['MoE_Fg_IoU'] * 100,
    'Standard U-Net': df['UNet_Fg_IoU'] * 100,
    'Attention U-Net': df['AttnUNet_Fg_IoU'] * 100
}

data_to_plot = [col_map[m] for m in models]
bp = plt.boxplot(data_to_plot, patch_artist=True, labels=models, widths=0.5,
                 medianprops=dict(color='black', linewidth=1.5))

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.65)

# Add jittered scatter points
for idx, m in enumerate(models):
    y = col_map[m]
    x = np.random.normal(idx + 1, 0.04, size=len(y))
    plt.scatter(x, y, color=colors[idx], edgecolor='black', s=50, alpha=0.9, zorder=5)

plt.title('Cross-Dataset Zero-Shot Foreground IoU (BUS-BRA, N=1,063)\nAcross 10 Matched Seeds', fontsize=13, fontweight='bold', pad=12)
plt.ylabel('Foreground IoU (%)', fontsize=11, fontweight='bold')
plt.ylim(20, 68)
plt.axhline(50, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

# Add mean text annotations
for idx, m in enumerate(models):
    mean_val = np.mean(col_map[m])
    std_val = np.std(col_map[m], ddof=1)
    plt.text(idx + 1, 23, f'{mean_val:.2f}%\n± {std_val:.2f}%', ha='center', fontsize=9.5, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=colors[idx], alpha=0.9))

plt.tight_layout()
p1_path = os.path.join(plot_dir, 'fig1_four_model_fg_iou_boxplot.png')
plt.savefig(p1_path)
plt.close()
print(f'Saved {p1_path}')

# 2. Plot 2: Precision-Recall Trade-off Scatter Plot
plt.figure(figsize=(9, 6), dpi=300)
markers = ['o', 's', '^', 'D']
for idx, m in enumerate(models):
    if m == 'Vanilla Swin-Tiny':
        prec = df['Vanilla_Precision'] * 100
        rec = df['Vanilla_Recall'] * 100
    elif m == 'Standard-MLP Swin-MoE':
        prec = df['MoE_Precision'] * 100
        rec = df['MoE_Recall'] * 100
    elif m == 'Standard U-Net':
        prec = df['UNet_Precision'] * 100
        rec = df['UNet_Recall'] * 100
    else:
        prec = df['AttnUNet_Precision'] * 100
        rec = df['AttnUNet_Recall'] * 100

    plt.scatter(rec, prec, color=colors[idx], marker=markers[idx], s=70, alpha=0.85, edgecolor='black', label=f'{m} (Seeds 1-10)')
    # Draw centroid
    plt.scatter(np.mean(rec), np.mean(prec), color=colors[idx], marker=markers[idx], s=220, edgecolor='black', linewidth=2.0)

# Annotate Attention U-Net outlier seeds 42 and 123
s42_rec = df.loc[df['Seed'] == 42, 'AttnUNet_Recall'].values[0] * 100
s42_prec = df.loc[df['Seed'] == 42, 'AttnUNet_Precision'].values[0] * 100
plt.annotate('Attn U-Net (Seed 42)\nSevere Collapse', xy=(s42_rec, s42_prec), xytext=(s42_rec + 4, s42_prec + 2),
             arrowprops=dict(facecolor='red', shrink=0.08, width=1, headwidth=6),
             fontsize=8.5, fontweight='bold', color='#990000',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffcccc', edgecolor='red', alpha=0.8))

s123_rec = df.loc[df['Seed'] == 123, 'AttnUNet_Recall'].values[0] * 100
s123_prec = df.loc[df['Seed'] == 123, 'AttnUNet_Precision'].values[0] * 100
plt.annotate('Attn U-Net (Seed 123)\nSevere Collapse', xy=(s123_rec, s123_prec), xytext=(s123_rec + 4, s123_prec - 6),
             arrowprops=dict(facecolor='red', shrink=0.08, width=1, headwidth=6),
             fontsize=8.5, fontweight='bold', color='#990000',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffcccc', edgecolor='red', alpha=0.8))

plt.title('Precision vs Recall Trade-Off Across 40 Experimental Runs\n(Large Symbols Represent Model Centroids)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Foreground Recall (%) — Lesion Sensitivity', fontsize=11, fontweight='bold')
plt.ylabel('Foreground Precision (%) — Specificity / Non-Lesion Suppression', fontsize=11, fontweight='bold')
plt.xlim(20, 90)
plt.ylim(50, 95)
plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower left')
plt.tight_layout()
p2_path = os.path.join(plot_dir, 'fig2_precision_recall_pareto_scatter.png')
plt.savefig(p2_path)
plt.close()
print(f'Saved {p2_path}')

# 3. Plot 3: Predicted Volume Ratio Comparison
plt.figure(figsize=(9, 5.5), dpi=300)
vr_data = [
    df['Vanilla_VolRatio'],
    df['MoE_VolRatio'],
    df['UNet_VolRatio'],
    df['AttnUNet_VolRatio']
]
vr_means = [np.mean(v) for v in vr_data]
vr_stds = [np.std(v, ddof=1) for v in vr_data]

x_pos = np.arange(len(models))
bars = plt.bar(x_pos, vr_means, yerr=vr_stds, capsize=6, color=colors, alpha=0.75, edgecolor='black', width=0.55)

plt.axhline(1.0, color='black', linestyle='--', linewidth=1.5, label='Ideal Calibration Ratio (1.0x Ground Truth)')
plt.xticks(x_pos, models, fontsize=10, fontweight='bold')
plt.ylabel('Predicted Volume Ratio (Pred / GT)', fontsize=11, fontweight='bold')
plt.title('Tumor Volume Calibration Ratio on BUS-BRA (N=1,063)\nSevere Volume Under-Prediction in Convolutional Baselines', fontsize=13, fontweight='bold', pad=12)
plt.ylim(0, 1.45)

for bar, mean_val, std_val in zip(bars, vr_means, vr_stds):
    y_text = bar.get_height() + std_val + 0.04
    plt.text(bar.get_x() + bar.get_width()/2, y_text, f'{mean_val:.3f}x\n± {std_val:.3f}x', ha='center', fontsize=9.5, fontweight='bold')

plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
plt.tight_layout()
p3_path = os.path.join(plot_dir, 'fig3_tumor_volume_ratio_comparison.png')
plt.savefig(p3_path)
plt.close()
print(f'Saved {p3_path}')

# 4. Plot 4: Paired Delta (Attention U-Net minus Standard U-Net) across 10 seeds
plt.figure(figsize=(9, 5), dpi=300)
seeds = df['Seed'].astype(str)
delta_iou = (df['AttnUNet_Fg_IoU'] - df['UNet_Fg_IoU']) * 100
bar_cols = ['#d62728' if d < 0 else '#2ca02c' for d in delta_iou]

x_s = np.arange(len(seeds))
plt.bar(x_s, delta_iou, color=bar_cols, alpha=0.8, edgecolor='black', width=0.6)
plt.axhline(0, color='black', linewidth=1.0)
plt.axhline(np.mean(delta_iou), color='blue', linestyle='--', linewidth=1.2, label=f'Mean Delta = {np.mean(delta_iou):.2f}% (p=0.1225)')

plt.xticks(x_s, [f'Seed {s}' for s in seeds], fontsize=9.5, rotation=30)
plt.ylabel('Delta Foreground IoU (% points)\n[Attention U-Net - Standard U-Net]', fontsize=11, fontweight='bold')
plt.title('Per-Seed Paired Differences: Effect of Adding Attention Gates to U-Net\n(Negative indicates Attention Gates degraded performance)', fontsize=12, fontweight='bold', pad=12)
plt.ylim(-30, 10)
plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower left')

for i, d in enumerate(delta_iou):
    y_pos = d - 2.0 if d < 0 else d + 1.0
    plt.text(i, y_pos, f'{d:+.1f}%', ha='center', fontsize=8.5, fontweight='bold')

plt.tight_layout()
p4_path = os.path.join(plot_dir, 'fig4_paired_delta_att_vs_standard_unet.png')
plt.savefig(p4_path)
plt.close()
print(f'Saved {p4_path}')

# 5. Plot 5: In-Domain BUSI vs Out-of-Domain BUS-BRA Domain Shift Penalty
p10_indomain = os.path.join(base_dir, 'results', 'phase10_unet', 'UNET_INDOMAIN_TEST_RESULTS.csv')
df_p10_in = pd.read_csv(p10_indomain).sort_values('Seed').reset_index(drop=True)

# Extract Attention U-Net in-domain directly
indomain_records = []
outputs_p11 = os.path.join(base_dir, 'results', 'results_phase11_attention_unet', 'outputs')
for rd in sorted(os.listdir(outputs_p11)):
    if not rd.startswith('attention_unet_seed'):
        continue
    seed = int(rd.split('_')[2].replace('seed', ''))
    tmet = os.path.join(outputs_p11, rd, 'logs', 'test_metrics.json')
    if os.path.exists(tmet):
        import json
        with open(tmet, 'r') as f:
            tm = json.load(f)
        indomain_records.append({
            'Model': 'Attention U-Net',
            'Seed': seed,
            'Test_Pixel_Accuracy': tm.get('pixel_accuracy'),
            'Test_mIoU': tm.get('mean_iou'),
            'Test_Foreground_IoU': tm['class_iou'][1] if len(tm.get('class_iou', [])) > 1 else None,
            'Test_Background_IoU': tm['class_iou'][0] if len(tm.get('class_iou', [])) > 0 else None,
            'Test_mDice': tm.get('mean_dice'),
            'Test_Foreground_Dice': tm['class_dice'][1] if len(tm.get('class_dice', [])) > 1 else None,
            'Test_Background_Dice': tm['class_dice'][0] if len(tm.get('class_dice', [])) > 0 else None,
            'Test_Foreground_Precision': tm['class_precision'][1] if len(tm.get('class_precision', [])) > 1 else None,
            'Test_Foreground_Recall': tm['class_recall'][1] if len(tm.get('class_recall', [])) > 1 else None,
        })
df_indomain = pd.DataFrame(indomain_records).sort_values('Seed').reset_index(drop=True)
p11_indomain_path = os.path.join(audit_dir, 'ATTENTION_UNET_INDOMAIN_TEST_RESULTS.csv')
df_indomain.to_csv(p11_indomain_path, index=False)
# Also copy to results/phase11_attention_unet if directory exists
res_p11_dir = os.path.join(base_dir, 'results', 'phase11_attention_unet')
os.makedirs(res_p11_dir, exist_ok=True)
df_indomain.to_csv(os.path.join(res_p11_dir, 'ATTENTION_UNET_INDOMAIN_TEST_RESULTS.csv'), index=False)

plt.figure(figsize=(8, 5.5), dpi=300)
comp_names = ['Standard U-Net', 'Attention U-Net']
in_means = [df_p10_in['Foreground_IoU'].mean() * 100, df_indomain['Test_Foreground_IoU'].mean() * 100]
in_stds = [df_p10_in['Foreground_IoU'].std() * 100, df_indomain['Test_Foreground_IoU'].std() * 100]

out_means = [df['UNet_Fg_IoU'].mean() * 100, df['AttnUNet_Fg_IoU'].mean() * 100]
out_stds = [df['UNet_Fg_IoU'].std() * 100, df['AttnUNet_Fg_IoU'].std() * 100]

x = np.arange(len(comp_names))
w = 0.35

rects1 = plt.bar(x - w/2, in_means, w, yerr=in_stds, capsize=5, label='In-Domain BUSI Test (N=78)', color='#3b528b', alpha=0.8, edgecolor='black')
rects2 = plt.bar(x + w/2, out_means, w, yerr=out_stds, capsize=5, label='Zero-Shot BUS-BRA (N=1,063)', color='#5ec962', alpha=0.8, edgecolor='black')

plt.xticks(x, comp_names, fontsize=11, fontweight='bold')
plt.ylabel('Foreground IoU (%)', fontsize=11, fontweight='bold')
plt.title('In-Domain vs Out-of-Domain Generalization Gap\nAttention Gates Suffer >2x Domain Shift Penalty', fontsize=12, fontweight='bold', pad=12)
plt.ylim(0, 75)
plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')

# Annotate domain shift drop
drop_unet = in_means[0] - out_means[0]
drop_att = in_means[1] - out_means[1]
plt.text(0, out_means[0] + out_stds[0] + 5, f'Drop: -{drop_unet:.2f}%\n(x1.0)', ha='center', fontsize=9.5, fontweight='bold', color='#b30000')
plt.text(1, out_means[1] + out_stds[1] + 5, f'Drop: -{drop_att:.2f}%\n(x2.4)', ha='center', fontsize=9.5, fontweight='bold', color='#b30000')

plt.tight_layout()
p5_path = os.path.join(plot_dir, 'fig5_domain_shift_penalty_comparison.png')
plt.savefig(p5_path)
plt.close()
print(f'Saved {p5_path}')
