# Standard U-Net Zero-Shot Cross-Dataset Evaluation Protocol

**Experiment Identifier**: `phase10_unet`  
**Auditor**: Senior Computer Vision Research Engineer & Scientific Auditor  
**Repository**: `D:\Study Material\Research\CV\vision_transformer_research`  
**Target Benchmark**: External BUS-BRA Dataset ($N = 1,063$)  

---

## 1. Zero-Shot Benchmark Specification

The zero-shot evaluation protocol measures how well models trained on the held-in BUSI dataset ($N=780$) transfer to an external, unseen clinical cohort (BUS-BRA, $N=1,063$):

1. **Complete Domain Shift**:
   - Different clinical institution, patient population, sonographer handling, and acoustic machine brands.
   - Varied speckle noise patterns, depth settings, and gain levels.
2. **Binary Clinical Lesion Formulation**:
   - BUS-BRA ground truth masks annotate tumor boundaries (lesions vs. normal breast tissue).
   - Because U-Net is trained on 3 classes on BUSI (0: background, 1: benign, 2: malignant), predictions are mapped to binary space via:
     $$\hat{y}_{\text{binary}} = (\operatorname{argmax}_{c} z_c > 0)$$
   - Ground truth masks are mapped identically: $y_{\text{binary}} = (y > 0)$.

---

## 2. Evaluation Metrics Protocol

To ensure exact parity with Phase 9B evaluation, the following seven metrics will be recorded for every seed:

1. **Foreground IoU ($\text{class\_iou}[1]$)**:
   $$\text{IoU}_{\text{lesion}} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$$
2. **Foreground Dice ($\text{class\_dice}[1]$)**:
   $$\text{Dice}_{\text{lesion}} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$$
3. **Macro mIoU**:
   $$\text{mIoU} = \frac{1}{2} (\text{IoU}_{\text{background}} + \text{IoU}_{\text{lesion}})$$
4. **Macro mDice**:
   $$\text{mDice} = \frac{1}{2} (\text{Dice}_{\text{background}} + \text{Dice}_{\text{lesion}})$$
5. **Foreground Precision**:
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
6. **Foreground Recall**:
   $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
7. **Pixel Accuracy**:
   $$\text{Acc} = \frac{\text{TP} + \text{TN}}{\text{Total Pixels}}$$

---

## 3. Scientific Hypotheses & Target Benchmark Thresholds

The table below outlines the scientific thresholds established by the audited Phase 9B experiments and the competing hypotheses for Standard U-Net:

| Metric | Phase 9B Vanilla Swin-Tiny (29.29M) | Phase 9B Standard Swin-MoE (81.15M) | Target U-Net Hypothesis (A: CNN Dominance) | Target U-Net Hypothesis (B: ViT Dominance) |
| :--- | :---: | :---: | :--- | :--- |
| **Foreground IoU** | $57.50\% \pm 4.74\%$ | $57.85\% \pm 2.15\%$ | $\ge 58.0\%$ (Translational equivariance preserves lesion boundaries) | $< 54.0\%$ (Local receptive fields fail to adapt to scanner shift) |
| **Foreground Dice** | $72.91\% \pm 3.98\%$ | $73.28\% \pm 1.73\%$ | $\ge 73.5\%$ (Sharp contour reconstruction via skip connections) | $< 70.0\%$ (Reduced global context causes false positives) |
| **Macro mIoU** | $76.10\% \pm 2.82\%$ | $76.30\% \pm 1.31\%$ | $\ge 76.5\%$ | $< 74.0\%$ |
| **Macro mDice** | $85.09\% \pm 2.23\%$ | $85.29\% \pm 0.99\%$ | $\ge 85.5\%$ | $< 83.0\%$ |
| **Volatiliy / Seed Range** | $14.00\%$ ($47.33\% \dots 61.33\%$) | $6.24\%$ ($54.18\% \dots 60.42\%$) | $s \le 3.0\%$ (Convolutions avoid transformer initialization collapse) | $s \ge 5.0\%$ (Small cohort causes severe convolutional seed volatility) |
| **Parameters** | $29,290,365$ | $81,148,989$ | **$31,037,763$** (~100% match to Vanilla Swin) | **$31,037,763$** |

---

## 4. Execution Output Artifacts

Upon running `Phase10_UNet_Baseline_Reproducibility.ipynb` on Kaggle GPU:
- `results/phase10_unet/UNET_PER_SEED_RESULTS.csv` will store all 10 seed evaluations.
- `results/phase10_unet/UNET_SUMMARY_RESULTS.csv` will store mean and standard deviation.
- `results/phase10_unet/unet_reproducibility_metrics.json` will store full confusion matrix tensors.
- All results will be packaged into `results_phase10_unet.zip`.
