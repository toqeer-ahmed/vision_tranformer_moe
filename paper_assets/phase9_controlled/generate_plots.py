import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def generate_paper_assets(metrics_json_path="results/phase9/reproducibility_metrics.json", output_dir="paper_assets/phase9_controlled"):
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(metrics_json_path):
        print(f"Metrics file not found: {metrics_json_path}")
        print("Please run the Reproducibility Notebook on Kaggle first and download the results!")
        return
        
    with open(metrics_json_path, "r") as f:
        results = json.load(f)
        
    seeds = [42, 123, 2026]
    models = ["vanilla", "standard_moe"]
    
    data = []
    for model_type in models:
        for seed in seeds:
            name = f"swin_{model_type}_seed{seed}"
            if name in results:
                m = results[name]
                data.append({
                    "Model": "Vanilla Swin" if model_type == "vanilla" else "Standard-MLP MoE",
                    "Seed": seed,
                    "mIoU": m["mean_iou"],
                    "Foreground_IoU": m["class_iou"][1],
                    "mDice": m["mean_dice"],
                    "Foreground_Dice": m["class_dice"][1],
                    "Foreground_Precision": m["class_precision"][1],
                    "Foreground_Recall": m["class_recall"][1]
                })
                
    if not data:
        print("No valid data found in metrics file.")
        return
        
    df = pd.DataFrame(data)
    
    # 1 & 2. Save Tables
    df.to_csv(os.path.join(output_dir, "PHASE9_CONTROLLED_RESULTS.csv"), index=False)
    
    summary = df.groupby("Model").agg(["mean", "std"]).round(4)
    summary.to_csv(os.path.join(output_dir, "PHASE9_CONTROLLED_SUMMARY.csv"))
    
    print("Tables saved.")
    
    # Plotting helper
    def plot_comparison(metric, title, filename):
        plt.figure(figsize=(8, 6))
        
        vanilla_data = df[df["Model"] == "Vanilla Swin"][metric].values
        moe_data = df[df["Model"] == "Standard-MLP MoE"][metric].values
        
        x = np.arange(len(seeds))
        width = 0.35
        
        plt.bar(x - width/2, vanilla_data, width, label='Vanilla Swin', color='skyblue')
        plt.bar(x + width/2, moe_data, width, label='Standard-MLP MoE', color='coral')
        
        plt.ylabel(metric)
        plt.title(title)
        plt.xticks(x, [f"Seed {s}" for s in seeds])
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        
    # 3, 4, 5. Plots
    plot_comparison("mIoU", "mIoU Comparison (BUSI -> BUSBRA Zero-Shot)", "miou_comparison.png")
    plot_comparison("Foreground_IoU", "Foreground IoU Comparison", "fg_iou_comparison.png")
    
    # 6. Precision-Recall Scatter
    plt.figure(figsize=(8, 6))
    for model_type in df["Model"].unique():
        subset = df[df["Model"] == model_type]
        plt.scatter(subset["Foreground_Recall"], subset["Foreground_Precision"], 
                    label=model_type, s=100, alpha=0.8)
        
    plt.xlabel("Foreground Recall")
    plt.ylabel("Foreground Precision")
    plt.title("Precision-Recall Distribution (Zero-Shot)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_recall_scatter.png"))
    plt.close()
    
    print(f"Paper assets generated in {output_dir}/")

if __name__ == "__main__":
    generate_paper_assets()
