import os
import json
import pandas as pd
import numpy as np
from scipy import stats

# Paths
base_dir = r'D:\Study Material\Research\CV\vision_transformer_research'
p9_csv = os.path.join(base_dir, 'research_audit', 'phase9_auxloss_corrected_final', 'RECALCULATED_PER_SEED_PAIRED.csv')
p10_csv = os.path.join(base_dir, 'research_audit', 'phase10_unet_final', 'PHASE10_PER_SEED_RECALCULATED.csv')
p11_raw_json = os.path.join(base_dir, 'results', 'results_phase11_attention_unet', 'results', 'phase11_attention_unet', 'attention_unet_reproducibility_metrics.json')

# 1. Load P11 from raw JSON and recalculate
with open(p11_raw_json, 'r') as f:
    p11_raw = json.load(f)

p11_list = []
for k, v in p11_raw.items():
    seed = int(k.replace('attention_unet_seed', ''))
    tp_bg, tp_fg = v['class_tp']
    fp_bg, fp_fg = v['class_fp']
    fn_bg, fn_fg = v['class_fn']

    iou_fg = tp_fg / (tp_fg + fp_fg + fn_fg)
    iou_bg = tp_bg / (tp_bg + fp_bg + fn_bg)
    miou = (iou_fg + iou_bg) / 2.0

    dice_fg = (2.0 * tp_fg) / (2.0 * tp_fg + fp_fg + fn_fg)
    dice_bg = (2.0 * tp_bg) / (2.0 * tp_bg + fp_bg + fn_bg)
    mdice = (dice_fg + dice_bg) / 2.0

    prec_fg = tp_fg / (tp_fg + fp_fg) if (tp_fg + fp_fg) > 0 else 0.0
    rec_fg = tp_fg / (tp_fg + fn_fg) if (tp_fg + fn_fg) > 0 else 0.0
    pix_acc = (tp_bg + tp_fg) / (tp_bg + tp_fg + fp_bg + fp_fg)

    pred_vol = tp_fg + fp_fg
    gt_vol = tp_fg + fn_fg
    vol_ratio = pred_vol / gt_vol

    p11_list.append({
        'Seed': seed,
        'AttnUNet_Fg_IoU': iou_fg,
        'AttnUNet_Fg_Dice': dice_fg,
        'AttnUNet_mIoU': miou,
        'AttnUNet_mDice': mdice,
        'AttnUNet_Precision': prec_fg,
        'AttnUNet_Recall': rec_fg,
        'AttnUNet_PixelAcc': pix_acc,
        'AttnUNet_PredVol': pred_vol,
        'AttnUNet_GTVol': gt_vol,
        'AttnUNet_VolRatio': vol_ratio,
        'TP_FG': tp_fg, 'FP_FG': fp_fg, 'FN_FG': fn_fg,
        'TP_BG': tp_bg, 'FP_BG': fp_bg, 'FN_BG': fn_bg
    })
df_p11 = pd.DataFrame(p11_list).sort_values('Seed').reset_index(drop=True)

# 2. Load P10 and P9
df_p10 = pd.read_csv(p10_csv).sort_values('Seed').reset_index(drop=True)
df_p9 = pd.read_csv(p9_csv).sort_values('Seed').reset_index(drop=True)

# Merge Master Benchmark
df_master = pd.DataFrame()
df_master['Seed'] = df_p11['Seed']

# Attention U-Net
df_master['AttnUNet_Fg_IoU'] = df_p11['AttnUNet_Fg_IoU']
df_master['AttnUNet_Fg_Dice'] = df_p11['AttnUNet_Fg_Dice']
df_master['AttnUNet_mIoU'] = df_p11['AttnUNet_mIoU']
df_master['AttnUNet_mDice'] = df_p11['AttnUNet_mDice']
df_master['AttnUNet_Precision'] = df_p11['AttnUNet_Precision']
df_master['AttnUNet_Recall'] = df_p11['AttnUNet_Recall']
df_master['AttnUNet_PixelAcc'] = df_p11['AttnUNet_PixelAcc']
df_master['AttnUNet_PredVol'] = df_p11['AttnUNet_PredVol']
df_master['AttnUNet_VolRatio'] = df_p11['AttnUNet_VolRatio']

# Standard U-Net
df_master['UNet_Fg_IoU'] = df_p10['UNet_Fg_IoU']
df_master['UNet_Fg_Dice'] = df_p10['UNet_Fg_Dice']
df_master['UNet_mIoU'] = df_p10['UNet_mIoU']
df_master['UNet_mDice'] = df_p10['UNet_mDice']
df_master['UNet_Precision'] = df_p10['UNet_Precision']
df_master['UNet_Recall'] = df_p10['UNet_Recall']
df_master['UNet_PixelAcc'] = df_p10['UNet_PixelAcc']
# Compute U-Net volume ratio
# From P10, Recall = TP / GT, Precision = TP / Pred => Pred / GT = Recall / Precision
df_master['UNet_VolRatio'] = df_master['UNet_Recall'] / df_master['UNet_Precision']
df_master['UNet_PredVol'] = df_master['UNet_VolRatio'] * 8146649

# Vanilla Swin
df_master['Vanilla_Fg_IoU'] = df_p9['V_FG_IoU']
df_master['Vanilla_Fg_Dice'] = df_p9['V_FG_Dice']
df_master['Vanilla_mIoU'] = df_p9['V_mIoU']
df_master['Vanilla_mDice'] = df_p9['V_mDice']
df_master['Vanilla_Precision'] = df_p9['V_FG_Precision']
df_master['Vanilla_Recall'] = df_p9['V_FG_Recall']
df_master['Vanilla_PixelAcc'] = df_p9['V_Pixel_Acc']
df_master['Vanilla_VolRatio'] = df_master['Vanilla_Recall'] / df_master['Vanilla_Precision']
df_master['Vanilla_PredVol'] = df_master['Vanilla_VolRatio'] * 8146649

# Swin MoE
df_master['MoE_Fg_IoU'] = df_p9['M_FG_IoU']
df_master['MoE_Fg_Dice'] = df_p9['M_FG_Dice']
df_master['MoE_mIoU'] = df_p9['M_mIoU']
df_master['MoE_mDice'] = df_p9['M_mDice']
df_master['MoE_Precision'] = df_p9['M_FG_Precision']
df_master['MoE_Recall'] = df_p9['M_FG_Recall']
df_master['MoE_PixelAcc'] = df_p9['M_Pixel_Acc']
df_master['MoE_VolRatio'] = df_master['MoE_Recall'] / df_master['MoE_Precision']
df_master['MoE_PredVol'] = df_master['MoE_VolRatio'] * 8146649

# Print summary means and standard deviations
models = [
    ('Attention U-Net', 'AttnUNet'),
    ('Standard U-Net', 'UNet'),
    ('Vanilla Swin-Tiny', 'Vanilla'),
    ('Standard-MLP Swin-MoE', 'MoE')
]
metrics = [
    ('Foreground IoU', 'Fg_IoU'),
    ('Foreground Dice', 'Fg_Dice'),
    ('Macro mIoU', 'mIoU'),
    ('Macro mDice', 'mDice'),
    ('Foreground Precision', 'Precision'),
    ('Foreground Recall', 'Recall'),
    ('Pixel Accuracy', 'PixelAcc'),
    ('Volume Ratio', 'VolRatio')
]

print('=== 4-MODEL BENCHMARK SUMMARY (MEAN +- SD) ===')
summary_rows = []
for m_name, m_pfx in models:
    row = {'Model': m_name}
    for met_name, met_sfx in metrics:
        col = f'{m_pfx}_{met_sfx}'
        vals = df_master[col].values
        row[f'{met_name}_Mean'] = np.mean(vals)
        row[f'{met_name}_Std'] = np.std(vals, ddof=1)
        row[f'{met_name}_Display'] = f'{np.mean(vals)*100:.2f}% +- {np.std(vals, ddof=1)*100:.2f}%' if 'VolRatio' not in met_sfx else f'{np.mean(vals):.3f}x +- {np.std(vals, ddof=1):.3f}x'
    summary_rows.append(row)
df_summary = pd.DataFrame(summary_rows)
for r in summary_rows:
    print(f"\nModel: {r['Model']}")
    for met_name, _ in metrics:
        print(f"  {met_name}: {r[met_name + '_Display']}")

# Statistical Comparisons for Attention U-Net vs each baseline
comparisons = [
    ('Attention U-Net vs Standard U-Net', 'AttnUNet', 'UNet'),
    ('Attention U-Net vs Vanilla Swin-Tiny', 'AttnUNet', 'Vanilla'),
    ('Attention U-Net vs Standard-MLP Swin-MoE', 'AttnUNet', 'MoE')
]

stat_results = []
t_crit = stats.t.ppf(0.975, df=9) # 2.262157

for comp_name, pfx1, pfx2 in comparisons:
    for met_name, met_sfx in metrics:
        v1 = df_master[f'{pfx1}_{met_sfx}'].values
        v2 = df_master[f'{pfx2}_{met_sfx}'].values
        delta = v1 - v2
        mean_delta = np.mean(delta)
        std_delta = np.std(delta, ddof=1)
        se_delta = std_delta / np.sqrt(len(delta))
        ci_lower = mean_delta - t_crit * se_delta
        ci_upper = mean_delta + t_crit * se_delta

        # Normality test
        shapiro_w, shapiro_p = stats.shapiro(delta)

        # Paired t-test
        t_stat, t_p = stats.ttest_rel(v1, v2)

        # Wilcoxon signed-rank test
        # Check for zero differences
        if np.all(delta == 0):
            w_stat, w_p = 0.0, 1.0
        else:
            try:
                w_res = stats.wilcoxon(v1, v2)
                w_stat, w_p = w_res.statistic, w_res.pvalue
            except Exception as e:
                w_stat, w_p = np.nan, np.nan

        # Cohen's dz
        cohen_dz = mean_delta / std_delta if std_delta > 0 else 0.0

        # Variance ratio F-test: var(v1) / var(v2)
        var1 = np.var(v1, ddof=1)
        var2 = np.var(v2, ddof=1)
        f_ratio = var1 / var2 if var2 > 0 else np.nan
        # F-test two-sided p-value
        if not np.isnan(f_ratio):
            p_f = 2 * min(stats.f.cdf(f_ratio, 9, 9), 1 - stats.f.cdf(f_ratio, 9, 9))
        else:
            p_f = np.nan

        # Levene test
        lev_stat, lev_p = stats.levene(v1, v2, center='median') # Brown-Forsythe

        stat_results.append({
            'Comparison': comp_name,
            'Metric': met_name,
            'Mean_1': np.mean(v1),
            'Std_1': np.std(v1, ddof=1),
            'Mean_2': np.mean(v2),
            'Std_2': np.std(v2, ddof=1),
            'Mean_Delta': mean_delta,
            'Std_Delta': std_delta,
            'SE_Delta': se_delta,
            'CI_95_Lower': ci_lower,
            'CI_95_Upper': ci_upper,
            'Shapiro_W': shapiro_w,
            'Shapiro_p': shapiro_p,
            't_stat': t_stat,
            't_p': t_p,
            'Wilcoxon_W': w_stat,
            'Wilcoxon_p': w_p,
            'Cohen_dz': cohen_dz,
            'Var_Ratio_F': f_ratio,
            'Var_F_p': p_f,
            'Levene_p': lev_p
        })

df_stats = pd.DataFrame(stat_results)

# Apply Holm-Bonferroni correction within each comparison across the 7 segmentation metrics
# (excluding volume ratio from primary family, or including all 8)
# Let's do Holm correction within each comparison across 7 metrics, and also across all 21 tests
for comp in df_stats['Comparison'].unique():
    mask = (df_stats['Comparison'] == comp) & (df_stats['Metric'] != 'Volume Ratio')
    sub_df = df_stats.loc[mask]
    # t-test p-values
    p_vals = sub_df['t_p'].values
    order = np.argsort(p_vals)
    n = len(p_vals)
    p_holm_t = np.zeros(n)
    for i, idx in enumerate(order):
        p_holm_t[idx] = min(1.0, p_vals[idx] * (n - i))
    # enforce monotonicity
    for i in range(1, n):
        idx = order[i]
        prev_idx = order[i-1]
        p_holm_t[idx] = max(p_holm_t[idx], p_holm_t[prev_idx])
    df_stats.loc[mask, 'Holm_p_t'] = p_holm_t

    # Wilcoxon p-values
    p_vals_w = sub_df['Wilcoxon_p'].values
    order_w = np.argsort(p_vals_w)
    p_holm_w = np.zeros(n)
    for i, idx in enumerate(order_w):
        p_holm_w[idx] = min(1.0, p_vals_w[idx] * (n - i))
    for i in range(1, n):
        idx = order_w[i]
        prev_idx = order_w[i-1]
        p_holm_w[idx] = max(p_holm_w[idx], p_holm_w[prev_idx])
    df_stats.loc[mask, 'Holm_p_wilcoxon'] = p_holm_w

# Also global Holm correction across all 21 primary tests (3 comparisons x 7 metrics)
mask_prim = df_stats['Metric'] != 'Volume Ratio'
p_all = df_stats.loc[mask_prim, 't_p'].values
order_all = np.argsort(p_all)
n_all = len(p_all)
p_holm_global = np.zeros(n_all)
for i, idx in enumerate(order_all):
    p_holm_global[idx] = min(1.0, p_all[idx] * (n_all - i))
for i in range(1, n_all):
    idx = order_all[i]
    prev_idx = order_all[i-1]
    p_holm_global[idx] = max(p_holm_global[idx], p_holm_global[prev_idx])
df_stats.loc[mask_prim, 'Holm_p_t_global'] = p_holm_global

print('\n=== DETAILED STATISTICAL COMPARISONS ===')
for comp in df_stats['Comparison'].unique():
    print(f'\n--- {comp} ---')
    sub = df_stats[df_stats['Comparison'] == comp]
    for _, r in sub.iterrows():
        met = r['Metric']
        d_mean = r['Mean_Delta']
        ci_l = r['CI_95_Lower']
        ci_u = r['CI_95_Upper']
        t_p = r['t_p']
        w_p = r['Wilcoxon_p']
        h_p = r.get('Holm_p_t', np.nan)
        dz = r['Cohen_dz']
        sh_p = r['Shapiro_p']
        vr = r['Var_Ratio_F']
        print(f"  {met}: Delta={d_mean*100:+.2f}% (95% CI: [{ci_l*100:+.2f}%, {ci_u*100:+.2f}%]), t_p={t_p:.4e}, Holm_p={h_p:.4e}, Wilc_p={w_p:.4e}, dz={dz:+.2f}, Shap_p={sh_p:.3f}, VarRatio={vr:.2f}")

# Save master CSVs
out_dir_final = os.path.join(base_dir, 'research_audit', 'phase11_attention_unet_final')
os.makedirs(out_dir_final, exist_ok=True)
master_csv_path = os.path.join(out_dir_final, 'FOUR_MODEL_BENCHMARK_MASTER.csv')
df_master.to_csv(master_csv_path, index=False)
stats_csv_path = os.path.join(out_dir_final, 'PHASE11_STATISTICAL_METRICS_WITH_HOLM.csv')
df_stats.to_csv(stats_csv_path, index=False)
p11_per_seed_csv = os.path.join(out_dir_final, 'PHASE11_PER_SEED_RECALCULATED.csv')
df_p11.to_csv(p11_per_seed_csv, index=False)
print(f'\nSaved master files to {out_dir_final}')
