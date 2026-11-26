# Phase 11 Precision-Recall & Volume Under-Prediction Analysis

**Document Identifier**: `research_audit/phase11_attention_unet_final/PHASE11_PRECISION_RECALL_ANALYSIS.md`  
**Workspace**: `D:\Study Material\Research\CV\vision_transformer_research`  
**Auditor**: Senior Computer Vision Research Engineer & Scientific Auditor  
**Date**: 2026-11-26  
**Primary Artifact**: [FOUR_MODEL_BENCHMARK_MASTER.csv](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/phase11_attention_unet_final/FOUR_MODEL_BENCHMARK_MASTER.csv)  

---

## 1. The Hypothesis and the Empirical Verdict

### The Phase 11 Research Hypothesis:
> *"Additive attention gates on skip connections filter background noise and suppress irrelevant activations, enabling the decoder to recover subtle peripheral tumor margins and resolve the volume under-prediction pathology ($0.74\times$ volume ratio, $58.57\%$ recall) observed in Standard U-Net."*

### Empirical Verdict: Hypothesis Not Supported by Empirical Evidence
*(Non-Significant Paired Difference: $t(9) = -1.71, p = 0.1225$, Holm $p = 0.7168$)*

Across 10 matched random seeds evaluated on $1,063$ unseen zero-shot BUS-BRA images:
1. **Recall Failed to Improve**:
   - Standard U-Net Recall: **$58.57\% \pm 6.49\%$**
   - Attention U-Net Recall: **$51.33\% \pm 11.32\%$**
   - Paired Delta: **$-7.24\%$** (95% CI: $[-17.41\%, +2.93\%]$, $d_z = -0.51$, $p = 0.1416$, Holm $p = 0.7168$).
   - *Statistical Conclusion*: The empirical data fail to demonstrate any recall benefit from attention gates. Because the confidence interval encompasses zero, this represents failure to establish an improvement rather than statistically proven deterioration.
2. **Volume Under-Prediction Remained Severe**:
   - Total GT Foreground Volume: **$8,146,649$ pixels** ($8.15\text{ MP}$).
   - Standard U-Net Predicted: **$6.00\text{ MP}$** ($0.736\times \pm 0.121\times$).
   - Attention U-Net Predicted: **$5.19\text{ MP}$** ($0.638\times \pm 0.162\times$).
   - Delta in Volume Ratio: **$-0.099\times$** ($-9.88\%$, $p = 0.1808$).
3. **Precision Remained at Parity**:
   - Standard U-Net Precision: **$80.19\% \pm 4.73\%$**
   - Attention U-Net Precision: **$81.38\% \pm 4.38\%$** ($\Delta = +1.19\%$, $95\%$ CI: $[-3.17\%, +5.55\%]$, $p = 0.5518$).

---

## 2. Seed-by-Seed Trajectory and The Catastrophic Failure Mode

The table below breaks down the precision, recall, and tumor volume metrics for Attention U-Net vs Standard U-Net across all 10 matched seeds:

| Seed | Standard U-Net Recall (%) | Attention U-Net Recall (%) | Delta Recall (% pts) | Standard U-Net Vol Ratio | Attention U-Net Vol Ratio | Attention U-Net Pred Pixels | Status / Failure Mode |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **42** | $67.41$ | **$29.56$** | **$-37.85$** | $0.892\times$ | **$0.336\times$** | $2,733,742$ | **Severe Skip-Gate Suppression** |
| **123** | $55.37$ | **$32.41$** | **$-22.96$** | $0.675\times$ | **$0.373\times$** | $3,040,914$ | **Severe Skip-Gate Suppression** |
| **314** | $63.88$ | $51.73$ | $-12.15$ | $0.819\times$ | $0.625\times$ | $5,095,475$ | Moderate Under-Prediction |
| **777** | $55.69$ | $57.23$ | $+1.54$ | $0.683\times$ | $0.690\times$ | $5,623,258$ | Near Parity |
| **999** | $45.15$ | $57.14$ | $+11.99$ | $0.524\times$ | $0.740\times$ | $6,031,195$ | Gain (U-Net failed on 999) |
| **1001** | $65.07$ | $60.33$ | $-4.74$ | $0.926\times$ | $0.816\times$ | $6,645,469$ | Moderate Under-Prediction |
| **2026** | $57.17$ | $61.07$ | $+3.90$ | $0.692\times$ | $0.789\times$ | $6,425,380$ | Slight Gain |
| **2027** | $57.65$ | $52.34$ | $-5.31$ | $0.716\times$ | $0.657\times$ | $5,354,299$ | Moderate Under-Prediction |
| **2718** | $62.90$ | $60.02$ | $-2.88$ | $0.791\times$ | $0.725\times$ | $5,906,446$ | Near Parity |
| **9001** | $55.41$ | $51.43$ | $-3.98$ | $0.648\times$ | $0.625\times$ | $5,091,801$ | Moderate Under-Prediction |

### Investigation of Seeds 42 and 123:
On seeds 42 and 123, Attention U-Net experienced catastrophic volume collapse:
- Seed 42 predicted only **$2.73\text{ MP}$** against $8.15\text{ MP}$ ground truth ($0.336\times$), missing **$70.44\%$** of actual tumor pixels.
- Seed 123 predicted only **$3.04\text{ MP}$** against $8.15\text{ MP}$ ground truth ($0.373\times$), missing **$67.59\%$** of actual tumor pixels.

Crucially, **this was not a training convergence failure**:
- On the in-domain BUSI test split, Seed 42 achieved **$57.50\%$ IoU** and **$64.19\%$ Recall** ($95.48\%$ pixel accuracy).
- On the in-domain BUSI test split, Seed 123 achieved **$55.61\%$ IoU** and **$65.26\%$ Recall** ($96.57\%$ pixel accuracy).
Both models learned healthy representations on the source dataset (BUSI). The collapse occurred **exclusively under cross-dataset domain shift** to BUS-BRA.

---

## 3. Sensitivity Analysis: Excluding the Two Outlier Seeds

To ensure conclusions are not driven solely by the extreme suppression on seeds 42 and 123, we perform a sensitivity re-analysis across the **8 stable seeds** ($\mathcal{S}_{\text{stable}} = \{314, 777, 999, 1001, 2026, 2027, 2718, 9001\}$):

| Metric | Attention U-Net (10 Seeds) | Attention U-Net (8 Stable Seeds) | Standard U-Net (10 Seeds) | Vanilla Swin-Tiny (10 Seeds) |
| :--- | :---: | :---: | :---: | :---: |
| **Foreground IoU (%)** | $45.35 \pm 8.64$ | **$49.38 \pm 2.58$** | $50.79 \pm 3.68$ | **$57.50 \pm 4.74$** |
| **Foreground Dice (%)** | $61.93 \pm 8.84$ | **$66.08 \pm 2.37$** | $67.29 \pm 3.33$ | **$72.91 \pm 3.98$** |
| **Foreground Precision (%)** | $81.38 \pm 4.38$ | **$79.99 \pm 3.25$** | $80.19 \pm 4.73$ | $70.44 \pm 6.61$ |
| **Foreground Recall (%)** | $51.33 \pm 11.32$ | **$56.41 \pm 4.14$** | $58.57 \pm 6.49$ | **$76.51 \pm 6.79$** |
| **Tumor Volume Ratio** | $0.638\times \pm 0.162\times$ | **$0.708\times \pm 0.071\times$** | $0.736\times \pm 0.121\times$ | **$1.101\times \pm 0.181\times$** |

### Critical Takeaway:
Even after excluding the two collapsed seeds, Attention U-Net's mean IoU ($49.38\%$), Dice ($66.08\%$), Recall ($56.41\%$), and Volume Ratio ($0.708\times$) **remain inferior to Standard U-Net** ($50.79\%$ IoU, $58.57\%$ Recall, $0.736\times$ Volume Ratio). At no point do Attention Gates provide a net benefit over un-gated skip connections.

---

## 4. In-Domain vs Out-of-Domain Generalization Gap

Comparing in-domain BUSI test performance against zero-shot BUS-BRA performance reveals the magnitude of the domain-shift penalty:

| Architecture | In-Domain BUSI Test IoU (%) | Zero-Shot BUS-BRA Fg IoU (%) | Absolute Domain-Shift Drop | Relative Generalization Drop |
| :--- | :---: | :---: | :---: | :---: |
| **Standard U-Net** | $56.06 \pm 3.88$ | $50.79 \pm 3.68$ | **$-5.27\%$** | $-9.40\%$ |
| **Attention U-Net** | $56.10 \pm 2.53$ | $45.35 \pm 8.64$ | **$-10.75\%$** | **$-19.16\%$** |

*(Reconciliation Note: Standard U-Net in-domain BUSI test IoU is verified at $56.06\% \pm 3.88\%$ in `UNET_INDOMAIN_TEST_RESULTS.csv`, reconciling the typographical $55.33\%$ entry from early drafts).*

```
[In-Domain BUSI Test (N=78)]
  Attention U-Net:  56.10% ──────┐
  Standard U-Net:   56.06% ──┐   │
                             │   │ Cross-Domain Transfer (BUS-BRA, N=1,063)
[Zero-Shot BUS-BRA]          │   │
  Standard U-Net:   50.79% ◄─┘ (-5.27% drop)
  Attention U-Net:  45.35% ◄─────┘ (-10.75% drop: ~2.04x greater degradation)
```

---

## 5. Mechanistic Investigation: Activation Profiles vs. Sigmoid Saturation

To understand why Attention U-Net under-performed under domain shift, we conducted a direct programmatic inspection of intermediate activations across saved checkpoints (`best_model.pth`) on both BUSI and BUS-BRA (documented in [PHASE11_ATTENTION_GATE_MECHANISM_ANALYSIS.md](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/phase11_attention_unet_final/PHASE11_ATTENTION_GATE_MECHANISM_ANALYSIS.md)).

### 1. Refutation of the Sigmoid Saturation Hypothesis:
- It was previously conjectured that gates suffered extreme sigmoid saturation near zero ($\alpha \to 0$), severing skip connections.
- **Empirical Measurement**: Across all 4 gates, the fraction of activations with $\alpha < 0.10$ is **$< 1.0\%$** for both collapsed seeds (Seed 42: $0.08\% - 0.66\%$) and healthy seeds (Seed 2026: $0.04\% - 0.75\%$).
- The feature transmission ratio ($\|\hat{x}\|_1 / \|x_{\text{skip}}\|_1$) remains virtually invariant at $\approx 50\%$ between in-domain BUSI and out-of-domain BUS-BRA.
- The skip pathways are therefore **not physically disconnected** by the attention gates.

### 2. The True Empirical Behavior: Diffuse Attenuation:
- Rather than acting as high-contrast spatial selectors that isolate lesion cores, the attention gates exhibit narrow dynamic ranges centered tightly around $0.50$ ($Q25 \approx 0.45, Q75 \approx 0.58$, fraction $> 0.90$ is $< 2\%$).
- The gates effectively act as a non-spatial scalar regularizer, halving feature magnitudes entering the decoder.

### 3. Source of Volume Under-Prediction:
- The volume under-prediction on seeds 42 and 123 is driven by **decoder double-convolution and classification-head sensitivity** to acoustic covariate shifts (transducer frequency, dynamic range, and acoustic shadowing), which depressed foreground logit margins below the classification threshold.
- Attention gates failed to provide the global contextual invariance necessary to stabilize these logits across domain shifts.
