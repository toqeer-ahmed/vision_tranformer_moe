# Phase 9 Metric Verification & Mathematical Audit

**Audit Date**: November 12, 2026  
**Auditor**: Senior Computer Vision Research Engineer & Scientific Auditor  
**Repository**: `D:\Study Material\Research\CV\vision_transformer_research`  
**Reference Document**: `PHASE9_FINAL_REPORT.md`  

---

## 1. Metric Mathematical Definitions & Accumulator Audit

The evaluation script relies on `evaluation/metrics.py` via `SegmentationMetricAccumulator(num_classes=2)`. In zero-shot evaluation on BUS-BRA, binary prediction is performed:
- **Class 0**: Background
- **Class 1**: Foreground (Tumor / Lesion)

Let $TP_c, FP_c, FN_c$ denote the True Positives, False Positives, and False Negatives for class $c \in \{0, 1\}$.

### A. Class-Wise Metrics
1. **Intersection over Union (Jaccard Index)**:
   $$\text{IoU}_c = \frac{TP_c}{TP_c + FP_c + FN_c}$$
   - **Background IoU** ($\text{class\_iou}[0]$): Measures background pixel agreement (typically $94\% - 96\%$).
   - **Foreground IoU** ($\text{class\_iou}[1]$): **The true clinical lesion segmentation metric**. Evaluates tumor boundary accuracy without background dominance.

2. **Dice Similarity Coefficient (F1 Score)**:
   $$\text{Dice}_c = \frac{2 \cdot TP_c}{2 \cdot TP_c + FP_c + FN_c}$$
   - **Background Dice** ($\text{class\_dice}[0]$): Typically $97\% - 98\%$.
   - **Foreground Dice** ($\text{class\_dice}[1]$): Evaluates overlap with ground truth tumor mask.

3. **Precision & Recall**:
   $$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}$$
   - Evaluated for foreground class $c=1$ (`class_precision[1]` and `class_recall[1]`).

### B. Macro-Averaged Metrics
$$\text{mIoU} = \frac{\text{IoU}_0 + \text{IoU}_1}{2}, \quad \text{mDice} = \frac{\text{Dice}_0 + \text{Dice}_1}{2}$$

> **Critical Scientific Distiction**:  
> In ultrasound images, lesions occupy only $5\% - 15\%$ of total image area. Because the background class is enormous, $\text{IoU}_0 \approx 0.95$. When averaged into $\text{mIoU}$, this pulls the metric up to $\sim 75\%$, whereas the true tumor boundary IoU ($\text{Foreground IoU}$) is $\sim 56\% - 58\%$. **Foreground IoU must never be conflated with mIoU.**

---

## 2. Independent Recalculation of Vanilla Swin Runs

The exact raw per-seed dictionaries for the three Vanilla Swin runs were extracted from the Kaggle execution log in turn 110:

```python
combined_results = {
    "swin_vanilla_seed42": {
        "mean_iou": 0.76882,
        "class_iou": [0.94849, 0.58916],
        "mean_dice": 0.85752,
        "class_dice": [0.97356, 0.74148],
        "class_precision": [0.98024, 0.69515],
        "class_recall": [0.96697, 0.79442],
    },
    "swin_vanilla_seed123": {
        "mean_iou": 0.77403,
        "class_iou": [0.95309, 0.59496],
        "mean_dice": 0.86102,
        "class_dice": [0.97598, 0.74605],
        "class_precision": [0.97578, 0.74768],
        "class_recall": [0.97618, 0.74443],
    },
    "swin_vanilla_seed2026": {
        "mean_iou": 0.72140,
        "class_iou": [0.94440, 0.49839],
        "mean_dice": 0.81832,
        "class_dice": [0.97140, 0.66524],
        "class_precision": [0.96315, 0.73940],
        "class_recall": [0.97980, 0.60460],
    },
}
```

### Statistical Verification:
Sample statistics calculated with Bessel's correction ($N=3, df=2$):
$$\bar{x} = \frac{1}{3} \sum_{i=1}^3 x_i, \quad s = \sqrt{\frac{1}{2} \sum_{i=1}^3 (x_i - \bar{x})^2}$$

| Metric | Seed 42 | Seed 123 | Seed 2026 | Calculated Mean | Calculated SD | Reported Table | Delta / Match |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **mIoU** | 0.76882 | 0.77403 | 0.72140 | **0.75475** (75.48%) | **0.02900** (2.90%) | $75.48\% \pm 2.90\%$ | **EXACT MATCH** |
| **Foreground IoU** | 0.58916 | 0.59496 | 0.49839 | **0.56084** (56.08%) | **0.05416** (5.42%) | $56.08\% \pm 5.42\%$ | **EXACT MATCH** |
| **mDice** | 0.85752 | 0.86102 | 0.81832 | **0.84562** (84.56%) | **0.02368** (2.37%) | $84.56\% \pm 2.37\%$ | **EXACT MATCH** |
| **Foreground Dice**| 0.74148 | 0.74605 | 0.66524 | **0.71759** (71.76%) | **0.04539** (4.54%) | $71.76\% \pm 4.54\%$ | **EXACT MATCH** |
| **Precision (FG)** | 0.69515 | 0.74768 | 0.73940 | **0.72741** (72.74%) | **0.02824** (2.82%) | $72.74\% \pm 2.82\%$ | **EXACT MATCH** |
| **Recall (FG)** | 0.79442 | 0.74443 | 0.60460 | **0.71448** (71.45%) | **0.09841** (9.84%) | $71.45\% \pm 9.84\%$ | **EXACT MATCH** |

The Vanilla Swin reported results in `PHASE9_FINAL_REPORT.md` are **100% verified** against the underlying per-seed numbers down to four decimal places.

---

## 3. Standard-MLP Swin-MoE Metric Audit

In turn 111 (step 3915), the Kaggle evaluation script computed the metrics for the three MoE models from their checkpoint directories:
- `outputs/swin_standard_moe_20261008_220211` (Seed 42)
- `outputs/swin_standard_moe_20261008_221522` (Seed 123)
- `outputs/swin_standard_moe_20261008_222535` (Seed 2026)

The output printed by pandas `df.groupby("Model").agg(["mean", "std"]).round(4)` was:

```text
=== FINAL CORRECTED SUMMARY ===
                      Seed               mIoU         Foreground_IoU          \
                      mean        std    mean     std           mean     std   
Model                                                                          
Standard-MLP MoE  730.3333  1122.8109  0.7672  0.0258         0.5811  0.0486   
Vanilla Swin      730.3333  1122.8109  0.7548  0.0290         0.5608  0.0542   

                   mDice         Foreground_Dice         Foreground_Precision  \
                    mean     std            mean     std                 mean   
Model                                                                           
Standard-MLP MoE  0.8552  0.0204          0.7342  0.0393               0.7736   
Vanilla Swin      0.8456  0.0237          0.7176  0.0454               0.7274   

                         Foreground_Recall          
                     std              mean     std  
Model                                               
Standard-MLP MoE  0.0319            0.7049  0.0928  
Vanilla Swin      0.0282            0.7145  0.0984  
```

### Side-by-Side Comparison:

| Metric | Reported Table in `PHASE9_FINAL_REPORT.md` | Terminal Summary in Kaggle Log | Verification Status |
| :--- | :---: | :---: | :---: |
| **mIoU** | $76.72\% \pm 2.58\%$ | `mean: 0.7672, std: 0.0258` | **VERIFIED** |
| **Foreground IoU** | $58.11\% \pm 4.86\%$ | `mean: 0.5811, std: 0.0486` | **VERIFIED** |
| **mDice** | $85.52\% \pm 2.04\%$ | `mean: 0.8552, std: 0.0204` | **VERIFIED** |
| **Foreground Dice** | $73.42\% \pm 3.93\%$ | `mean: 0.7342, std: 0.0393` | **VERIFIED** |
| **Foreground Precision** | $77.36\% \pm 3.19\%$ | `mean: 0.7736, std: 0.0319` | **VERIFIED** |
| **Foreground Recall** | $70.49\% \pm 9.28\%$ | `mean: 0.7049, std: 0.0928` | **VERIFIED** |

---

## 4. Audit Finding on Raw Artifact Residency

- **Verified Fact**: The reported mean and standard deviation numbers in `PHASE9_FINAL_REPORT.md` are not fabricated; they directly transcribe the exact terminal output from Kaggle.
- **Missing Local Artifact**: The raw per-seed breakdown JSON (`results/phase9/reproducibility_metrics.json`) and the individual per-seed rows in `PHASE9_CONTROLLED_RESULTS.csv` were generated into `/kaggle/working/results/phase9/paper_assets/`, but were **never copied or downloaded** into the local workspace repository on disk (`d:\Study Material\Research\CV\vision_transformer_research\results\phase9\`).
- **Audit Verdict**: The summary statistics are verified through execution transcripts, but the individual seed breakdowns for Standard-MLP MoE remain **partially verified (cloud-resident)** until the output files are synced locally from Kaggle.

