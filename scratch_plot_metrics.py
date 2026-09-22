import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import seaborn as sns

def get_metrics(log_dir, tag):
    events = []
    if not os.path.exists(log_dir):
        return events
    for file in os.listdir(log_dir):
        if file.startswith("events.out.tfevents"):
            ea = EventAccumulator(os.path.join(log_dir, file))
            ea.Reload()
            if tag in ea.Tags()['scalars']:
                for e in ea.Scalars(tag):
                    events.append((e.step, e.value))
    events.sort(key=lambda x: x[0])
    return [e[0] for e in events], [e[1] for e in events]

def plot_combined_metrics():
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Get SegMoTE data
    seg_steps, seg_vals = get_metrics('notebook_results/extracted_results/medical_segmentation/logs', 'Metrics/mIoU')
    if seg_vals:
        plt.plot(seg_steps, seg_vals, label='SegMoTE (94.2M params)', color='#2ca02c', linewidth=2.5, marker='o')

    # Get MoE-Segformer data
    moe_steps, moe_vals = get_metrics('notebook_results/extracted_results/moe_segmentation/logs', 'Metrics/mIoU')
    if moe_vals:
        plt.plot(moe_steps, moe_vals, label='MoE-Segformer (8.07M params)', color='#ff7f0e', linewidth=2.5, linestyle='--', marker='s')

    plt.title('Validation mIoU over Training Epochs\n(ISIC 2018 Dataset)', fontsize=16, pad=15)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Mean Intersection over Union (mIoU)', fontsize=14)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs('notebook_results/plots', exist_ok=True)
    plt.savefig('notebook_results/plots/combined_miou_plot.png', dpi=300)
    print("Plot saved to notebook_results/plots/combined_miou_plot.png")

if __name__ == '__main__':
    plot_combined_metrics()
