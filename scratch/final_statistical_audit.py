import json
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. Load raw OOD metrics from reproducibility_metrics.json
with open('outputs/results_phase9_auxloss_corrected/results/phase9_auxloss_corrected/reproducibility_metrics.json', 'r') as f:
    ood_raw = json.load(f)

seeds = [42, 123, 314, 777, 999, 1001, 2026, 2027, 2718, 9001]

# Extract per-seed metrics directly from raw json
data = []
for s in seeds:
    v_key = f'swin_vanilla_seed{s}'
    m_key = f'swin_standard_moe_seed{s}'
    
    assert v_key in ood_raw, f'Missing {v_key}'
    assert m_key in ood_raw, f'Missing {m_key}'
    
    v = ood_raw[v_key]
    m = ood_raw[m_key]
    
    data.append({
        'Seed': s,
        # Vanilla
        'V_FG_IoU': v['class_iou'][1],
        'V_FG_Dice': v['class_dice'][1],
        'V_mIoU': v['mean_iou'],
        'V_mDice': v['mean_dice'],
        'V_FG_Precision': v['class_precision'][1],
        'V_FG_Recall': v['class_recall'][1],
        'V_Pixel_Acc': v['pixel_accuracy'],
        # MoE
        'M_FG_IoU': m['class_iou'][1],
        'M_FG_Dice': m['class_dice'][1],
        'M_mIoU': m['mean_iou'],
        'M_mDice': m['mean_dice'],
        'M_FG_Precision': m['class_precision'][1],
        'M_FG_Recall': m['class_recall'][1],
        'M_Pixel_Acc': m['pixel_accuracy'],
    })

df = pd.DataFrame(data).sort_values('Seed').reset_index(drop=True)

metrics = [
    ('Foreground IoU', 'V_FG_IoU', 'M_FG_IoU'),
    ('Foreground Dice', 'V_FG_Dice', 'M_FG_Dice'),
    ('Macro mIoU', 'V_mIoU', 'M_mIoU'),
    ('Macro mDice', 'V_mDice', 'M_mDice'),
    ('Foreground Precision', 'V_FG_Precision', 'M_FG_Precision'),
    ('Foreground Recall', 'V_FG_Recall', 'M_FG_Recall'),
    ('Pixel Accuracy', 'V_Pixel_Acc', 'M_Pixel_Acc'),
]

# Statistical Table
stats_rows = []
for name, v_col, m_col in metrics:
    v_arr = df[v_col].values
    m_arr = df[m_col].values
    diff = m_arr - v_arr
    
    mean_v, std_v = np.mean(v_arr), np.std(v_arr, ddof=1)
    mean_m, std_m = np.mean(m_arr), np.std(m_arr, ddof=1)
    mean_d, std_d = np.mean(diff), np.std(diff, ddof=1)
    
    # Paired t-test
    t_stat, p_t = stats.ttest_rel(m_arr, v_arr)
    # Wilcoxon signed-rank
    w_stat, p_w = stats.wilcoxon(diff)
    # Cohen's dz
    dz = mean_d / std_d if std_d > 0 else 0.0
    # 95% CI
    ci_low, ci_high = stats.t.interval(0.95, df=len(seeds)-1, loc=mean_d, scale=stats.sem(diff))
    
    # F-test on variance (ratio = s_v^2 / s_m^2)
    var_v = std_v ** 2
    var_m = std_m ** 2
    f_stat = var_v / var_m
    # Two-sided p-value for F-test
    p_f = 2.0 * min(stats.f.cdf(f_stat, len(seeds)-1, len(seeds)-1), 1.0 - stats.f.cdf(f_stat, len(seeds)-1, len(seeds)-1))
    
    # Levene test (mean-centered)
    lev_stat, p_lev = stats.levene(v_arr, m_arr, center='mean')
    # Brown-Forsythe test (median-centered)
    bf_stat, p_bf = stats.levene(v_arr, m_arr, center='median')
    
    # Shapiro-Wilk test for normality
    shapiro_v_stat, p_shapiro_v = stats.shapiro(v_arr)
    shapiro_m_stat, p_shapiro_m = stats.shapiro(m_arr)
    shapiro_diff_stat, p_shapiro_diff = stats.shapiro(diff)
    
    stats_rows.append({
        'Metric': name,
        'Vanilla Mean±SD': f'{mean_v*100:.2f}% ± {std_v*100:.2f}%',
        'MoE Mean±SD': f'{mean_m*100:.2f}% ± {std_m*100:.2f}%',
        'Mean Delta': f'{mean_d*100:+.2f}%',
        '95% CI': f'[{ci_low*100:.2f}%, {ci_high*100:.2f}%]',
        't_stat': t_stat,
        'p_t': p_t,
        'W_stat': w_stat,
        'p_w': p_w,
        'Cohen_dz': dz,
        'F_stat': f_stat,
        'p_f_uncorrected': p_f,
        'Levene_p': p_lev,
        'BrownForsythe_p': p_bf,
        'Shapiro_V_p': p_shapiro_v,
        'Shapiro_M_p': p_shapiro_m,
        'Shapiro_Diff_p': p_shapiro_diff
    })

res_df = pd.DataFrame(stats_rows)

# Multiple comparisons correction for the 7 variance tests (p_f_uncorrected)
m_count = len(res_df)
# Bonferroni
res_df['F_p_Bonferroni'] = np.minimum(1.0, res_df['p_f_uncorrected'] * m_count)

# Holm-Bonferroni for F-test
sorted_indices = np.argsort(res_df['p_f_uncorrected'])
holm_p = np.zeros(m_count)
for rank, idx in enumerate(sorted_indices):
    factor = m_count - rank
    holm_p[idx] = min(1.0, res_df['p_f_uncorrected'][idx] * factor)
# Ensure monotonicity
sorted_holm = holm_p[sorted_indices]
for i in range(1, len(sorted_holm)):
    if sorted_holm[i] < sorted_holm[i-1]:
        sorted_holm[i] = sorted_holm[i-1]
res_df['F_p_Holm'] = holm_p

# Benjamini-Hochberg FDR
fdr_p = np.zeros(m_count)
for rank, idx in enumerate(sorted_indices):
    fdr_p[idx] = min(1.0, res_df['p_f_uncorrected'][idx] * m_count / (rank + 1))
sorted_fdr = fdr_p[sorted_indices]
for i in range(len(sorted_fdr)-2, -1, -1):
    if sorted_fdr[i] > sorted_fdr[i+1]:
        sorted_fdr[i] = sorted_fdr[i+1]
res_df['F_p_FDR'] = fdr_p

print('=== INDEPENDENTLY RECALCULATED STATISTICAL VALIDATION TABLE ===')
display_cols = ['Metric', 'Vanilla Mean±SD', 'MoE Mean±SD', 'Mean Delta', '95% CI', 'p_t', 'p_w', 'Cohen_dz', 'F_stat', 'p_f_uncorrected', 'F_p_Bonferroni', 'F_p_Holm', 'BrownForsythe_p', 'Shapiro_V_p']
print(res_df[display_cols].to_string())

# Save analysis csv
out_dir = 'research_audit/phase9_auxloss_corrected_final'
plot_dir = os.path.join(out_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)

res_df.to_csv(os.path.join(out_dir, 'RECALCULATED_STATISTICAL_METRICS.csv'), index=False)
df.to_csv(os.path.join(out_dir, 'RECALCULATED_PER_SEED_PAIRED.csv'), index=False)

# PLOTS:
# 1. Per-seed paired difference bar chart for Foreground IoU
plt.figure(figsize=(10, 5))
deltas = (df['M_FG_IoU'] - df['V_FG_IoU']) * 100
seeds_str = [str(s) for s in df['Seed']]
colors = ['#2ca02c' if d >= 0 else '#d62728' for d in deltas]

plt.bar(seeds_str, deltas, color=colors, edgecolor='black', alpha=0.85)
plt.axhline(0, color='black', linewidth=1.2, linestyle='--')
plt.axhline(deltas.mean(), color='blue', linewidth=1.5, linestyle='-', label=f'Mean Delta = {deltas.mean():+.2f}%')
plt.title('Phase 9B: Per-Seed Paired Difference in Foreground IoU (MoE - Vanilla)', fontsize=13, fontweight='bold')
plt.xlabel('Random Seed', fontsize=11)
plt.ylabel('Delta IoU (%)', fontsize=11)
plt.legend(fontsize=11)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
paired_plot_path = os.path.join(plot_dir, 'paired_difference_foreground_iou.png')
plt.savefig(paired_plot_path, dpi=300)
plt.close()

# 2. Distribution comparison: Boxplot + Jittered Scatter for Vanilla vs MoE Foreground IoU
plt.figure(figsize=(8, 6))
v_iou = df['V_FG_IoU'] * 100
m_iou = df['M_FG_IoU'] * 100

box = plt.boxplot([v_iou, m_iou], labels=['Vanilla Swin-Tiny\n(29.29M)', 'Standard-MLP Swin-MoE\n(81.15M)'],
                  patch_artist=True, widths=0.45,
                  boxprops=dict(facecolor='#e6f2ff', color='blue', alpha=0.6),
                  medianprops=dict(color='darkblue', linewidth=2.0))

# Overlay individual seed points with matched connection lines
x_v = np.ones(len(v_iou)) + np.random.normal(0, 0.02, len(v_iou))
x_m = np.ones(len(m_iou))*2 + np.random.normal(0, 0.02, len(m_iou))

for i in range(len(v_iou)):
    plt.plot([1, 2], [v_iou[i], m_iou[i]], color='gray', linestyle=':', alpha=0.6, linewidth=1.0)
    plt.scatter(1, v_iou[i], color='royalblue', s=45, zorder=3)
    plt.scatter(2, m_iou[i], color='forestgreen', s=45, zorder=3)

plt.title('Distribution of Zero-Shot BUS-BRA Foreground IoU (N=10 Matched Seeds)', fontsize=12, fontweight='bold')
plt.ylabel('Foreground IoU (%)', fontsize=11)
plt.ylim(44, 65)

# Annotate minimums
plt.annotate(f'Vanilla Min: {v_iou.min():.2f}%\n(Seed 123)',
             xy=(1, v_iou.min()), xytext=(1.15, v_iou.min()-1.0),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.2), color='red', fontweight='bold')

plt.annotate(f'MoE Min: {m_iou.min():.2f}%\n(Seed 123)',
             xy=(2, m_iou.min()), xytext=(1.7, m_iou.min()-2.2),
             arrowprops=dict(arrowstyle='->', color='green', lw=1.2), color='green', fontweight='bold')

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
dist_plot_path = os.path.join(plot_dir, 'distribution_comparison_foreground_iou.png')
plt.savefig(dist_plot_path, dpi=300)
plt.close()

print(f'Saved plots to {plot_dir}')
