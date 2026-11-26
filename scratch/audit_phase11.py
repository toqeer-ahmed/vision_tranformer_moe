import os
import glob
import json
import yaml
import pandas as pd
import numpy as np
from scipy import stats

base_p11 = r'D:\Study Material\Research\CV\vision_transformer_research\results\results_phase11_attention_unet'
outputs_dir = os.path.join(base_p11, 'outputs')
configs_dir = os.path.join(base_p11, 'configs', 'phase11_attention_unet')
results_dir = os.path.join(base_p11, 'results', 'phase11_attention_unet')

print('=== 1. CONFIG AUDIT ===')
config_files = glob.glob(os.path.join(configs_dir, 'attention_unet_seed*.yaml'))
for cf in sorted(config_files):
    if 'smoke' in cf:
        continue
    with open(cf, 'r') as f:
        c = yaml.safe_load(f)
    print(f"{os.path.basename(cf)}: seed={c.get('seed')}, lr={c.get('training', {}).get('lr')}, epochs={c.get('training', {}).get('epochs')}, loss={c.get('training', {}).get('loss_function')}, batch_size={c.get('training', {}).get('batch_size')}")

print('\n=== 2. OUTPUT RUNS AUDIT ===')
run_dirs = glob.glob(os.path.join(outputs_dir, 'attention_unet_seed*'))
for rd in sorted(run_dirs):
    bname = os.path.basename(rd)
    ckpt = os.path.join(rd, 'checkpoints', 'best_model.pth')
    tmet = os.path.join(rd, 'logs', 'test_metrics.json')
    ckpt_sz = os.path.getsize(ckpt) if os.path.exists(ckpt) else 0
    tmet_ex = os.path.exists(tmet)
    print(f'{bname}: ckpt_sz={ckpt_sz:,} bytes, test_metrics_exists={tmet_ex}')

# 3. In-domain test metrics
indomain_records = []
for rd in sorted(run_dirs):
    bname = os.path.basename(rd)
    seed = int(bname.split('_')[2].replace('seed', ''))
    tmet = os.path.join(rd, 'logs', 'test_metrics.json')
    if os.path.exists(tmet):
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
print('\n=== 3. IN-DOMAIN TEST RESULTS (BUSI Test Split) ===')
print(df_indomain[['Seed', 'Test_Foreground_IoU', 'Test_Foreground_Dice', 'Test_Foreground_Precision', 'Test_Foreground_Recall', 'Test_Pixel_Accuracy']])
print(f"Summary In-Domain Test Foreground IoU: Mean = {df_indomain['Test_Foreground_IoU'].mean():.4f}, Std = {df_indomain['Test_Foreground_IoU'].std():.4f}")

# 4. Out-of-Domain Zero-Shot Evaluation (BUS-BRA) Recalculation from Raw Confusion Matrices
json_metrics_file = os.path.join(results_dir, 'attention_unet_reproducibility_metrics.json')
with open(json_metrics_file, 'r') as f:
    raw_data = json.load(f)

recalc_records = []
for k, v in raw_data.items():
    seed = int(k.replace('attention_unet_seed', ''))
    tp_bg, tp_fg = v['class_tp']
    fp_bg, fp_fg = v['class_fp']
    fn_bg, fn_fg = v['class_fn']

    # Recalculate
    iou_fg = tp_fg / (tp_fg + fp_fg + fn_fg)
    iou_bg = tp_bg / (tp_bg + fp_bg + fn_bg)
    miou = (iou_fg + iou_bg) / 2.0

    dice_fg = (2.0 * tp_fg) / (2.0 * tp_fg + fp_fg + fn_fg)
    dice_bg = (2.0 * tp_bg) / (2.0 * tp_bg + fp_bg + fn_bg)
    mdice = (dice_fg + dice_bg) / 2.0

    prec_fg = tp_fg / (tp_fg + fp_fg) if (tp_fg + fp_fg) > 0 else 0.0
    rec_fg = tp_fg / (tp_fg + fn_fg) if (tp_fg + fn_fg) > 0 else 0.0

    prec_bg = tp_bg / (tp_bg + fp_bg) if (tp_bg + fp_bg) > 0 else 0.0
    rec_bg = tp_bg / (tp_bg + fn_bg) if (tp_bg + fn_bg) > 0 else 0.0

    pix_acc = (tp_bg + tp_fg) / (tp_bg + tp_fg + fp_bg + fp_fg)

    pred_vol = tp_fg + fp_fg
    gt_vol = tp_fg + fn_fg
    vol_ratio = pred_vol / gt_vol

    recalc_records.append({
        'Model': 'Attention U-Net',
        'Seed': seed,
        'mIoU': miou,
        'Foreground_IoU': iou_fg,
        'Background_IoU': iou_bg,
        'mDice': mdice,
        'Foreground_Dice': dice_fg,
        'Background_Dice': dice_bg,
        'Foreground_Precision': prec_fg,
        'Foreground_Recall': rec_fg,
        'Pixel_Accuracy': pix_acc,
        'Predicted_Volume_Pixels': pred_vol,
        'GroundTruth_Volume_Pixels': gt_vol,
        'Volume_Ratio': vol_ratio,
        'TP_FG': tp_fg,
        'FP_FG': fp_fg,
        'FN_FG': fn_fg,
        'TP_BG': tp_bg,
        'FP_BG': fp_bg,
        'FN_BG': fn_bg
    })

df_p11_recalc = pd.DataFrame(recalc_records).sort_values('Seed').reset_index(drop=True)
print('\n=== 4. RECALCULATED BUS-BRA ZERO-SHOT RESULTS (ATTENTION U-NET) ===')
print(df_p11_recalc[['Seed', 'Foreground_IoU', 'Foreground_Dice', 'Foreground_Precision', 'Foreground_Recall', 'Volume_Ratio']])

# Verify against provided CSV
csv_provided = os.path.join(results_dir, 'ATTENTION_UNET_PER_SEED_RESULTS.csv')
df_provided = pd.read_csv(csv_provided).sort_values('Seed').reset_index(drop=True)
max_diff_iou = np.max(np.abs(df_p11_recalc['Foreground_IoU'].values - df_provided['Foreground_IoU'].values))
max_diff_dice = np.max(np.abs(df_p11_recalc['Foreground_Dice'].values - df_provided['Foreground_Dice'].values))
print(f'\nVerification vs Provided CSV: max |diff_iou| = {max_diff_iou:.2e}, max |diff_dice| = {max_diff_dice:.2e}')
assert max_diff_iou < 1e-12, 'Recalculated IoU mismatch!'
assert max_diff_dice < 1e-12, 'Recalculated Dice mismatch!'
print('Verification PASSED: Raw confusion matrix recalculations match provided CSV to machine precision!')
