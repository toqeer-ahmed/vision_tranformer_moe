import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('outputs/results_phase9_auxloss_corrected/results/phase9_auxloss_corrected/CORRECTED_PER_SEED_RESULTS.csv')
vanilla = df[df['Model'] == 'Vanilla Swin'].sort_values('Seed').reset_index(drop=True)
moe = df[df['Model'] == 'Standard-MLP MoE'].sort_values('Seed').reset_index(drop=True)

metrics = ['Foreground_IoU', 'Foreground_Dice', 'mIoU', 'mDice', 'Pixel_Accuracy', 'Foreground_Precision', 'Foreground_Recall']

print('Seeds:', vanilla['Seed'].tolist())

results = []
for m in metrics:
    v = vanilla[m].values
    mo = moe[m].values
    diff = mo - v
    
    t_stat, p_val = stats.ttest_rel(mo, v)
    w_stat, w_pval = stats.wilcoxon(diff)
    
    mean_v, std_v = np.mean(v), np.std(v, ddof=1)
    mean_mo, std_mo = np.mean(mo), np.std(mo, ddof=1)
    mean_d, std_d = np.mean(diff), np.std(diff, ddof=1)
    
    # Cohen's dz
    dz = mean_d / std_d if std_d > 0 else 0
    
    # 95% CI
    ci_low, ci_high = stats.t.interval(0.95, df=len(v)-1, loc=mean_d, scale=stats.sem(diff))
    
    # Variance ratio F-test: s_v^2 / s_mo^2
    f_stat = (std_v**2) / (std_mo**2)
    # two-sided p-value for F-test with df1=9, df2=9
    p_f = 2 * min(stats.f.cdf(f_stat, len(v)-1, len(mo)-1), 1 - stats.f.cdf(f_stat, len(v)-1, len(mo)-1))
    
    results.append({
        'Metric': m,
        'Vanilla': f'{mean_v*100:.2f}% ± {std_v*100:.2f}%',
        'MoE': f'{mean_mo*100:.2f}% ± {std_mo*100:.2f}%',
        'Delta': f'{(mean_mo - mean_v)*100:+.2f}%',
        '95% CI': f'[{ci_low*100:.2f}%, {ci_high*100:.2f}%]',
        't-stat': f'{t_stat:.4f}',
        'p-val (t)': f'{p_val:.4f}',
        'W-stat': f'{w_stat:.1f}',
        'p-val (W)': f'{w_pval:.4f}',
        'Cohen dz': f'{dz:.4f}',
        'F-stat (var)': f'{f_stat:.4f}',
        'p-val (F)': f'{p_f:.4f}'
    })

res_df = pd.DataFrame(results)
print('\n=== STATISTICAL HYPOTHESIS TESTS (10 SEEDS) ===')
print(res_df.to_string())

print('\n=== PER SEED MATCHED COMPARISON ===')
per_seed = pd.DataFrame({
    'Seed': vanilla['Seed'],
    'Vanilla IoU': (vanilla['Foreground_IoU'] * 100).round(2),
    'MoE IoU': (moe['Foreground_IoU'] * 100).round(2),
    'Delta IoU': ((moe['Foreground_IoU'] - vanilla['Foreground_IoU']) * 100).round(2),
    'Vanilla Dice': (vanilla['Foreground_Dice'] * 100).round(2),
    'MoE Dice': (moe['Foreground_Dice'] * 100).round(2),
    'Delta Dice': ((moe['Foreground_Dice'] - vanilla['Foreground_Dice']) * 100).round(2),
    'Winner': ['MoE' if mo_i > va_i else 'Vanilla' for mo_i, va_i in zip(moe['Foreground_IoU'], vanilla['Foreground_IoU'])]
})
print(per_seed.to_string(index=False))
