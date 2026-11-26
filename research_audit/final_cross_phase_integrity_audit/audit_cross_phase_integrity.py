import json
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

# Ensure workspace root is in path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPECTED_SEEDS = [42, 123, 314, 777, 999, 1001, 2026, 2027, 2718, 9001]

def run_audit():
    print("================================================================================")
    print("FINAL CROSS-PHASE INTEGRITY AUDIT — PHASES 9B, 10, 11, AND 12")
    print("================================================================================")

    # 1. Load raw verified per-seed files
    p9_csv_path = os.path.join(REPO_ROOT, "results", "phase9_auxloss_corrected", "CORRECTED_PER_SEED_RESULTS.csv")
    p9_json_path = os.path.join(REPO_ROOT, "results", "phase9_auxloss_corrected", "reproducibility_metrics.json")
    
    p10_csv_path = os.path.join(REPO_ROOT, "results", "phase10_unet", "UNET_PER_SEED_RESULTS.csv")
    p10_in_path = os.path.join(REPO_ROOT, "results", "phase10_unet", "UNET_INDOMAIN_TEST_RESULTS.csv")
    p10_json_path = os.path.join(REPO_ROOT, "results", "phase10_unet", "unet_reproducibility_metrics.json")
    
    p11_csv_path = os.path.join(REPO_ROOT, "results", "phase11_attention_unet", "ATTENTION_UNET_PER_SEED_RESULTS.csv")
    p11_in_path = os.path.join(REPO_ROOT, "results", "phase11_attention_unet", "ATTENTION_UNET_INDOMAIN_TEST_RESULTS.csv")
    p11_json_path = os.path.join(REPO_ROOT, "results", "phase11_attention_unet", "attention_unet_reproducibility_metrics.json")
    
    p12_csv_path = os.path.join(REPO_ROOT, "results", "phase12_resnet50_unet", "RESNET50_UNET_PER_SEED_RESULTS.csv")
    p12_in_path = os.path.join(REPO_ROOT, "results", "phase12_resnet50_unet", "RESNET50_UNET_INDOMAIN_TEST_RESULTS.csv")
    p12_json_path = os.path.join(REPO_ROOT, "results", "phase12_resnet50_unet", "resnet50_unet_reproducibility_metrics.json")

    # Load CSVs
    p9_df = pd.read_csv(p9_csv_path)
    swin_df = p9_df[p9_df["Model"].str.contains("Vanilla")].sort_values("Seed").reset_index(drop=True)
    moe_df = p9_df[p9_df["Model"].str.contains("MoE")].sort_values("Seed").reset_index(drop=True)
    unet_df = pd.read_csv(p10_csv_path).sort_values("Seed").reset_index(drop=True)
    att_df = pd.read_csv(p11_csv_path).sort_values("Seed").reset_index(drop=True)
    res_df = pd.read_csv(p12_csv_path).sort_values("Seed").reset_index(drop=True)

    # In-domain CSVs
    unet_in_df = pd.read_csv(p10_in_path).sort_values("Seed").reset_index(drop=True)
    att_in_df = pd.read_csv(p11_in_path).sort_values("Seed").reset_index(drop=True)
    res_in_df = pd.read_csv(p12_in_path).sort_values("Seed").reset_index(drop=True)

    models_dict = {
        "Vanilla Swin-Tiny": swin_df,
        "Standard-MLP Swin-MoE": moe_df,
        "Standard U-Net": unet_df,
        "Attention U-Net": att_df,
        "ResNet-50 U-Net": res_df
    }

    # Verify seed alignment
    print("\n--- 1. SEED ALIGNMENT AUDIT ---")
    for name, df in models_dict.items():
        actual_seeds = df["Seed"].tolist()
        assert actual_seeds == EXPECTED_SEEDS, f"Seed mismatch in {name}: {actual_seeds} != {EXPECTED_SEEDS}"
        print(f"  [OK] {name:25s}: Exact 10 seeds matched ({actual_seeds})")

    # Ensure Volume Ratio columns
    for name, df in models_dict.items():
        if "Volume_Ratio" not in df.columns:
            df["Volume_Ratio"] = df["Foreground_Recall"] / df["Foreground_Precision"]

    # Compute Canonical Master Table
    metrics = [
        "Foreground_IoU", "Foreground_Dice", "Foreground_Recall", "Foreground_Precision",
        "mIoU", "mDice", "Pixel_Accuracy", "Volume_Ratio"
    ]

    canonical_rows = []
    for name, df in models_dict.items():
        row = {"Model": name}
        for m in metrics:
            vals = df[m].values
            row[f"{m}_Mean"] = np.mean(vals)
            row[f"{m}_Std"] = np.std(vals, ddof=1)
            row[f"{m}_Median"] = np.median(vals)
            row[f"{m}_IQR"] = stats.iqr(vals)
            row[f"{m}_Min"] = np.min(vals)
            row[f"{m}_Max"] = np.max(vals)
        canonical_rows.append(row)

    canonical_summary_df = pd.DataFrame(canonical_rows)
    canonical_summary_csv = os.path.join(OUTPUT_DIR, "CANONICAL_FIVE_MODEL_RESULTS.csv")
    canonical_summary_df.to_csv(canonical_summary_csv, index=False)
    print(f"\n[Saved] Canonical five-model summary -> {canonical_summary_csv}")

    # Display Canonical Results
    print("\n--- 2. CANONICAL FIVE-MODEL BENCHMARK TABLE (BUS-BRA ZERO-SHOT) ---")
    header = f"{'Model':25s} | {'FG IoU':18s} | {'FG Dice':18s} | {'FG Recall':18s} | {'FG Precision':18s} | {'Volume Ratio':18s}"
    print(header)
    print("-" * len(header))
    for _, r in canonical_summary_df.iterrows():
        print(
            f"{r['Model']:25s} | "
            f"{r['Foreground_IoU_Mean']*100:6.2f} ± {r['Foreground_IoU_Std']*100:5.2f}% | "
            f"{r['Foreground_Dice_Mean']*100:6.2f} ± {r['Foreground_Dice_Std']*100:5.2f}% | "
            f"{r['Foreground_Recall_Mean']*100:6.2f} ± {r['Foreground_Recall_Std']*100:5.2f}% | "
            f"{r['Foreground_Precision_Mean']*100:6.2f} ± {r['Foreground_Precision_Std']*100:5.2f}% | "
            f"{r['Volume_Ratio_Mean']:6.4f} ± {r['Volume_Ratio_Std']:6.4f} "
        )

    # Volume Ratio Comparison: Per-seed mean vs Pooled
    print("\n--- 3. VOLUME RATIO FORMULATION AUDIT ---")
    vr_audit_rows = []
    json_paths = {
        "Vanilla Swin-Tiny": (p9_json_path, "vanilla"),
        "Standard-MLP Swin-MoE": (p9_json_path, "moe"),
        "Standard U-Net": (p10_json_path, None),
        "Attention U-Net": (p11_json_path, None),
        "ResNet-50 U-Net": (p12_json_path, None)
    }

    for name, (j_path, filter_k) in json_paths.items():
        with open(j_path) as f:
            j_data = json.load(f)
        if filter_k:
            keys = [k for k in j_data.keys() if filter_k in k]
        else:
            keys = list(j_data.keys())
            
        sum_pred = 0
        sum_gt = 0
        per_seed_ratios = []
        for k in sorted(keys):
            m = j_data[k]
            tp = m["class_tp"][1]
            fp = m["class_fp"][1]
            fn = m["class_fn"][1]
            pred = tp + fp
            gt = tp + fn
            per_seed_ratios.append(pred / gt)
            sum_pred += pred
            sum_gt += gt
            
        mean_per_seed = np.mean(per_seed_ratios)
        std_per_seed = np.std(per_seed_ratios, ddof=1)
        pooled_ratio = sum_pred / sum_gt
        
        vr_audit_rows.append({
            "Model": name,
            "Mean_Per_Seed_Ratio": mean_per_seed,
            "Std_Per_Seed_Ratio": std_per_seed,
            "Pooled_Ratio": pooled_ratio,
            "Total_Pred_Pixels": sum_pred,
            "Total_GT_Pixels": sum_gt
        })
        print(f"  {name:25s} | Mean Per-Seed: {mean_per_seed:.4f} ± {std_per_seed:.4f} | Pooled: {pooled_ratio:.4f} | Pred: {sum_pred:,} | GT: {sum_gt:,}")

    # In-Domain BUSI Test Audit
    print("\n--- 4. IN-DOMAIN BUSI TEST AUDIT ($N = 78$) ---")
    in_domain_records = {
        "Standard U-Net": unet_in_df["Foreground_IoU"].values,
        "Attention U-Net": (att_in_df["Test_Foreground_IoU"].values if "Test_Foreground_IoU" in att_in_df.columns else att_in_df["Foreground_IoU"].values),
        "ResNet-50 U-Net": res_in_df["Foreground_IoU"].values
    }
    for name, vals in in_domain_records.items():
        zero_shot_vals = models_dict[name]["Foreground_IoU"].values
        penalty = vals - zero_shot_vals
        print(f"  {name:20s} | In-Domain FG IoU: {np.mean(vals)*100:5.2f} ± {np.std(vals, ddof=1)*100:5.2f}% | Zero-Shot: {np.mean(zero_shot_vals)*100:5.2f} ± {np.std(zero_shot_vals, ddof=1)*100:5.2f}% | Domain Drop: -{np.mean(penalty)*100:5.2f} ± {np.std(penalty, ddof=1)*100:5.2f}%")

    # 5. Paired Statistical Comparisons with Holm-Bonferroni Correction
    print("\n--- 5. PAIRED STATISTICAL COMPARISONS (ALL 6 PLANNED PAIRS) ---")
    planned_comparisons = [
        ("Vanilla Swin-Tiny", swin_df, "ResNet-50 U-Net", res_df),
        ("Standard U-Net", unet_df, "ResNet-50 U-Net", res_df),
        ("Vanilla Swin-Tiny", swin_df, "Standard U-Net", unet_df),
        ("Vanilla Swin-Tiny", swin_df, "Attention U-Net", att_df),
        ("Standard U-Net", unet_df, "Attention U-Net", att_df),
        ("Standard-MLP Swin-MoE", moe_df, "Vanilla Swin-Tiny", swin_df)
    ]

    primary_7_metrics = [
        "Foreground_IoU", "Foreground_Dice", "Foreground_Recall", "Foreground_Precision",
        "mIoU", "mDice", "Pixel_Accuracy"
    ]

    paired_results_all = []

    for name_a, df_a, name_b, df_b in planned_comparisons:
        comp_key = f"{name_a} vs. {name_b}"
        print(f"\n================================================================================")
        print(f"COMPARISON: {comp_key}")
        print(f"================================================================================")
        
        comp_rows = []
        for met in primary_7_metrics:
            vals_a = df_a[met].values
            vals_b = df_b[met].values
            diff = vals_a - vals_b
            n = len(diff)
            mean_diff = np.mean(diff)
            std_diff = np.std(diff, ddof=1)
            sem = std_diff / np.sqrt(n)
            cohen_dz = mean_diff / std_diff if std_diff > 0 else 0.0
            
            t_stat, p_t = stats.ttest_rel(vals_a, vals_b)
            ci_low, ci_high = stats.t.interval(0.95, df=n-1, loc=mean_diff, scale=sem)
            
            try:
                w_stat, p_w = stats.wilcoxon(vals_a, vals_b)
            except Exception:
                w_stat, p_w = np.nan, np.nan
                
            comp_rows.append({
                "Comparison": comp_key,
                "Model_A": name_a,
                "Model_B": name_b,
                "Metric": met,
                "Mean_A": np.mean(vals_a),
                "Mean_B": np.mean(vals_b),
                "Mean_Diff (A - B)": mean_diff,
                "Std_Diff": std_diff,
                "Cohen_dz": cohen_dz,
                "95%_CI_Lower": ci_low,
                "95%_CI_Upper": ci_high,
                "t_statistic": t_stat,
                "p_value_raw": p_t,
                "wilcoxon_stat": w_stat,
                "wilcoxon_p": p_w
            })
            
        df_comp = pd.DataFrame(comp_rows)
        # Apply Holm-Bonferroni correction within the 7-metric family
        df_comp = df_comp.sort_values("p_value_raw").reset_index(drop=True)
        m_tests = len(df_comp)
        adjusted_p = []
        running_max = 0.0
        for i, row in df_comp.iterrows():
            k = i + 1
            adj = row["p_value_raw"] * (m_tests - k + 1)
            adj = min(adj, 1.0)
            running_max = max(running_max, adj)
            adjusted_p.append(running_max)
            
        df_comp["p_value_holm"] = adjusted_p
        df_comp["Significant_05"] = df_comp["p_value_holm"] < 0.05
        
        display_cols = ["Metric", "Mean_Diff (A - B)", "Cohen_dz", "95%_CI_Lower", "95%_CI_Upper", "p_value_raw", "p_value_holm", "Significant_05"]
        print(df_comp[display_cols].to_string(index=False))
        
        paired_results_all.append(df_comp)

    all_paired_df = pd.concat(paired_results_all, ignore_index=True)
    paired_csv = os.path.join(OUTPUT_DIR, "CANONICAL_PAIRED_STATISTICAL_RESULTS.csv")
    all_paired_df.to_csv(paired_csv, index=False)
    print(f"\n[Saved] Canonical paired statistical results -> {paired_csv}")
    print("\nAudit calculations finished successfully.")

if __name__ == "__main__":
    run_audit()
