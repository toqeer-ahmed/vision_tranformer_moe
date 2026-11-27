# Final Source-Level Reconciliation: Five-Model Breast Ultrasound Study

**Audit Date:** November 27, 2026  
**Auditor:** Independent ML Research Auditor & Biostatistician  
**Workspace:** `D:\Study Material\Research\CV\vision_transformer_research`  
**Associated Canonical Artifacts:**
- [FINAL_VERIFIED_PER_SEED_VALUES.csv](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/final_cross_phase_integrity_audit/FINAL_VERIFIED_PER_SEED_VALUES.csv)
- [FINAL_VERIFIED_INDOMAIN_RESULTS.csv](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/final_cross_phase_integrity_audit/FINAL_VERIFIED_INDOMAIN_RESULTS.csv)
- [FINAL_VERIFIED_PAIRED_TESTS.csv](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/final_cross_phase_integrity_audit/FINAL_VERIFIED_PAIRED_TESTS.csv)
- [FINAL_MULTIPLICITY_FAMILIES.md](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/final_cross_phase_integrity_audit/FINAL_MULTIPLICITY_FAMILIES.md)
- [FINAL_MANUSCRIPT_NUMBERS.md](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/final_cross_phase_integrity_audit/FINAL_MANUSCRIPT_NUMBERS.md)

---

## 1. Actual Source of Truth Inventory

Every experimental source file in the repository was inspected at the byte and record level. No values were assumed, interpolated, or copied from summary documents.

| Audit Source ID | File Path | Number of Rows | Seed Sequence | Seed Status | Metric Columns Audited | Value Format | Observation Level |
|---|---|:---:|---|:---:|---|:---:|:---:|
| **SRC-01** | `results/phase9_auxloss_corrected/CORRECTED_PER_SEED_RESULTS.csv` | 20 | `[42, 123, 2026, 777, 999, 314, 2718, 1001, 2027, 9001]` $\times 2$ models | 10 matched seeds, 0 missing | `Model`, `Seed`, `mIoU`, `Foreground_IoU`, `Background_IoU`, `mDice`, `Foreground_Dice`, `Background_Dice`, `Foreground_Precision`, `Foreground_Recall`, `Pixel_Accuracy` | Fractions ($0.0 \dots 1.0$) | Seed-level aggregate (BUS-BRA zero-shot, $1,875$ images) |
| **SRC-02** | `results/phase10_unet/UNET_PER_SEED_RESULTS.csv` | 10 | `[42, 123, 314, 777, 999, 1001, 2026, 2027, 2718, 9001]` | 10 matched seeds, 0 missing | Same 11 columns as above | Fractions ($0.0 \dots 1.0$) | Seed-level aggregate (BUS-BRA zero-shot, $1,875$ images) |
| **SRC-03** | `results/phase11_attention_unet/ATTENTION_UNET_PER_SEED_RESULTS.csv` | 10 | `[42, 123, 314, 777, 999, 1001, 2026, 2027, 2718, 9001]` | 10 matched seeds, 0 missing | Same 11 columns + `Predicted_Volume_Pixels`, `GroundTruth_Volume_Pixels`, `Volume_Ratio` | Fractions ($0.0 \dots 1.0$) | Seed-level aggregate (BUS-BRA zero-shot, $1,875$ images) |
| **SRC-04** | `results/phase12_resnet50_unet/RESNET50_UNET_PER_SEED_RESULTS.csv` | 10 | `[42, 123, 314, 777, 999, 1001, 2026, 2027, 2718, 9001]` | 10 matched seeds, 0 missing | Same 14 columns as above | Fractions ($0.0 \dots 1.0$) | Seed-level aggregate (BUS-BRA zero-shot, $1,875$ images) |
| **SRC-05** | `outputs/results_phase9_auxloss_corrected/outputs/swin_vanilla_*/logs/test_metrics.json` | 10 files | `[42, 123, 314, 777, 999, 1001, 2026, 2027, 2718, 9001]` | 10 matched seeds, 0 missing | `pixel_accuracy`, `mean_iou`, `mean_dice`, `class_iou`, `class_dice`, `class_precision`, `class_recall`, `class_tp`, `class_fp`, `class_fn` | Fractions ($0.0 \dots 1.0$) | Seed-level aggregate (BUSI test, $78$ images, $3,913,728$ pixels) |
| **SRC-06** | `outputs/results_phase9_auxloss_corrected/outputs/swin_standard_moe_*/logs/test_metrics.json` | 10 files | `[42, 123, 314, 777, 999, 1001, 2026, 2027, 2718, 9001]` | 10 matched seeds, 0 missing | Same JSON schema as above | Fractions ($0.0 \dots 1.0$) | Seed-level aggregate (BUSI test, $78$ images, $3,913,728$ pixels) |
| **SRC-07** | `results/phase10_unet/UNET_INDOMAIN_TEST_RESULTS.csv` | 10 | `[42, 123, 314, 777, 999, 1001, 2026, 2027, 2718, 9001]` | 10 matched seeds, 0 missing | Same 11 columns as SRC-02 | Fractions ($0.0 \dots 1.0$) | Seed-level aggregate (BUSI test, $78$ images, $3,913,728$ pixels) |
| **SRC-08** | `results/phase11_attention_unet/ATTENTION_UNET_INDOMAIN_TEST_RESULTS.csv` | 10 | `[42, 123, 314, 777, 999, 1001, 2026, 2027, 2718, 9001]` | 10 matched seeds, 0 missing | `Model`, `Seed`, `Test_Pixel_Accuracy`, `Test_mIoU`, `Test_Foreground_IoU`, `Test_Background_IoU`, `Test_mDice`, etc. | Fractions ($0.0 \dots 1.0$) | Seed-level aggregate (BUSI test, $78$ images, $3,913,728$ pixels) |
| **SRC-09** | `results/phase12_resnet50_unet/RESNET50_UNET_INDOMAIN_TEST_RESULTS.csv` | 10 | `[42, 123, 314, 777, 999, 1001, 2026, 2027, 2718, 9001]` | 10 matched seeds, 0 missing | Same 11 columns as SRC-02 | Fractions ($0.0 \dots 1.0$) | Seed-level aggregate (BUSI test, $78$ images, $3,913,728$ pixels) |

---

## 2. In-Domain Metric Reconciliation & Domain Drop Audit

### A. Raw In-Domain BUSI Test Metrics ($N=10$ Seeds, $78$ Images per Seed)
All statistics are computed directly from the canonical per-seed records using sample standard deviation ($ddof=1$).

| Model Architecture | Canonical Source File & Column | Mean FG IoU | Sample SD ($ddof=1$) | Min FG IoU | Max FG IoU | Exact Per-Seed FG IoU Values (Seeds 42, 123, 314, 777, 999, 1001, 2026, 2027, 2718, 9001) |
|---|---|:---:|:---:|:---:|:---:|---|
| **Vanilla Swin-Tiny** | `SRC-05`: `class_iou[1]` in `test_metrics.json` | **$60.24\%$** | $6.11\%$ | $52.17\%$ | $71.33\%$ | `[56.23%, 52.17%, 60.60%, 67.02%, 61.13%, 62.32%, 53.21%, 55.38%, 71.33%, 62.97%]` |
| **Standard-MLP Swin-MoE** | `SRC-06`: `class_iou[1]` in `test_metrics.json` | **$60.58\%$** | $5.28\%$ | $53.14\%$ | $69.12\%$ | `[57.23%, 53.14%, 55.17%, 64.81%, 63.42%, 64.03%, 55.02%, 59.78%, 69.12%, 64.08%]` |
| **Standard U-Net** | `SRC-07`: `Foreground_IoU` | **$56.06\%$** | $3.88\%$ | $48.70\%$ | $62.04\%$ | `[52.14%, 60.00%, 62.04%, 57.90%, 55.71%, 57.86%, 55.11%, 48.70%, 57.37%, 53.76%]` |
| **Attention U-Net** | `SRC-08`: `Test_Foreground_IoU` | **$56.10\%$** | $2.53\%$ | $51.88Binding%` | $60.41\%$ | `[57.50%, 55.61%, 56.20%, 57.74%, 53.21%, 53.98%, 60.41%, 57.85%, 56.63%, 51.88%]` |
| **ResNet-50 U-Net** | `SRC-09`: `Foreground_IoU` | **$59.26\%$** | $4.35\%$ | $54.23\%$ | $65.55\%$ | `[56.17%, 54.23%, 55.84%, 57.80%, 60.79%, 54.74%, 58.25%, 64.19%, 65.06%, 65.55%]` |

### B. Explicit Sign Convention and Domain Drop Recomputation

#### Definition of Sign Convention:
$$\Delta_{\text{transfer}} = \text{BUS-BRA Zero-Shot FG IoU} - \text{BUSI In-Domain FG IoU}$$
- A **negative** value indicates a performance decline under hardware/scanner distribution shift.
- A **positive** value indicates an increase in zero-shot performance over in-domain performance.

| Model Architecture | In-Domain BUSI FG IoU | Zero-Shot BUS-BRA FG IoU | Mean Domain Drop ($\Delta_{\text{transfer}}$) | SD of Paired Drop ($ddof=1$) | Per-Seed Paired Domain Drops ($\text{pp}$) |
|---|:---:|:---:|:---:|:---:|---|
| **Vanilla Swin-Tiny** | $60.24\% \pm 6.11\%$ | $57.50\% \pm 4.74\%$ | **$-2.74\text{ pp}$** | $4.17\text{ pp}$ | `[+0.55, -4.84, +0.01, -6.86, -0.15, -3.24, -2.52, +2.86, -11.55, -1.64]` |
| **Standard-MLP Swin-MoE** | $60.58\% \pm 5.28\%$ | $57.85\% \pm 2.15\%$ | **$-2.73\text{ pp}$** | $4.89\text{ pp}$ | `[+2.01, +1.04, +0.95, -6.88, -6.85, -4.52, +4.24, +0.11, -8.70, -8.67]` |
| **Standard U-Net** | $56.06\% \pm 3.88\%$ | $50.79\% \pm 3.68\%$ | **$-5.27\text{ pp}$** | $5.28\text{ pp}$ | `[+3.20, -10.62, -7.91, -8.44, -13.59, -6.83, -4.08, +1.92, -3.26, -3.09]` |
| **Attention U-Net** | $56.10\% \pm 2.53\%$ | $45.35\% \pm 8.64\%$ | **$-10.75\text{ pp}$** | $8.96\text{ pp}$ | `[-29.08, -24.72, -9.52, -6.54, -4.33, -4.23, -8.57, -11.68, -3.27, -5.57]` |
| **ResNet-50 U-Net** | $59.26\% \pm 4.35\%$ | $50.36\% \pm 6.98\%$ | **$-8.91\text{ pp}$** | $8.00\text{ pp}$ | `[+0.97, +2.06, -6.08, -8.64, -6.79, -18.52, -5.58, -23.32, -14.80, -8.35]` |

---

## 3. Resolution of the Swin-Tiny vs. Standard U-Net Discrepancy

### A. The Conflict
- **Latest Verified Report**: Mean Diff = $+6.71\text{ pp}$, Paired $t = 3.467$, Raw $p = 0.007077$.
- **Earlier Draft Report**: Mean Diff = $+6.71\text{ pp}$, Paired $t = 4.220$, Raw $p = 0.0022$.

### B. Ten Paired Observations Joined by Seed ID

| Seed ID | Vanilla Swin-Tiny FG IoU | Standard U-Net FG IoU | Paired Difference ($\Delta = \text{Swin} - \text{UNet}$) |
|:---:|:---:|:---:|:---:|
| **42** | $0.567834$ ($56.78\%$) | $0.553332$ ($55.33\%$) | $+0.014502$ ($+1.45\text{ pp}$) |
| **123** | $0.473300$ ($47.33\%$) | $0.493782$ ($49.38\%$) | $-0.020482$ ($-2.05\text{ pp}$) |
| **314** | $0.606076$ ($60.61\%$) | $0.541295$ ($54.13\%$) | $+0.064782$ ($+6.48\text{ pp}$) |
| **777** | $0.601599$ ($60.16\%$) | $0.494657$ ($49.47\%$) | $+0.106942$ ($+10.69\text{ pp}$) |
| **999** | $0.609777$ ($60.98\%$) | $0.421151$ ($42.12\%$) | $+0.188626$ ($+18.86\text{ pp}$) |
| **1001** | $0.590842$ ($59.08\%$) | $0.510308$ ($51.03\%$) | $+0.080534$ ($+8.05\text{ pp}$) |
| **2026** | $0.506894$ ($50.69\%$) | $0.510338$ ($51.03\%$) | $-0.003444$ ($-0.34\text{ pp}$) |
| **2027** | $0.582407$ ($58.24\%$) | $0.506118$ ($50.61\%$) | $+0.076289$ ($+7.63\text{ pp}$) |
| **2718** | $0.597847$ ($59.78\%$) | $0.541145$ ($54.11\%$) | $+0.056702$ ($+5.67\text{ pp}$) |
| **9001** | $0.613296$ ($61.33\%$) | $0.506655$ ($50.67\%$) | $+0.106641$ ($+10.66\text{ pp}$) |

### C. Recomputation and Mathematical Proof

- **Mean Paired Difference**:
  $$\bar{d} = \frac{1}{10} \sum_{i=1}^{10} d_i = \mathbf{+0.067109} \quad (+6.7109\text{ pp})$$
- **Sample Standard Deviation of Differences ($ddof=1$)**:
  $$s_d = \sqrt{\frac{1}{9} \sum_{i=1}^{10} (d_i - \bar{d})^2} = \mathbf{0.061202} \quad (6.1202\text{ pp})$$
- **Standard Error of the Mean Difference**:
  $$\text{SEM} = \frac{s_d}{\sqrt{10}} = \frac{0.061202}{3.162278} = \mathbf{0.019354} \quad (1.9354\text{ pp})$$
- **Degrees of Freedom**: $df = 10 - 1 = 9$
- **Paired Student's $t$-statistic**:
  $$t = \frac{\bar{d}}{\text{SEM}} = \frac{0.067109}{0.019354} = \mathbf{+3.467473}$$
- **Two-Tailed $p$-value ($t$-distribution, $df=9$)**:
  $$p = 2 \cdot (1 - F_t(3.467473, 9)) = \mathbf{0.007077}$$
- **95% Confidence Interval for $\bar{d}$**:
  $$\bar{d} \pm t_{0.975, 9} \cdot \text{SEM} = 0.067109 \pm 2.262157 \cdot 0.019354 = \mathbf{[+0.023328, +0.110891]} \quad ([+2.33\text{ pp}, +11.09\text{ pp}])$$
- **Wilcoxon Signed-Rank Test**:
  Rank sums: $W = \mathbf{4.0}$, Two-tailed exact $p$-value = $\mathbf{0.013672}$
- **Cohen's $d_z$**:
  $$d_z = \frac{\bar{d}}{s_d} = \frac{0.067109}{0.061202} = \mathbf{+1.096511}$$

### D. Root Cause Analysis of the Discrepancy
The earlier draft value ($t = 4.220, p = 0.0022$) did **not** derive from any calculation on these ten paired observations. An audit of git history and intermediate scripts reveals that $t = 4.220$ was a transcription error from an unverified draft summary that used an assumed lower difference standard deviation ($s_d \approx 0.0503$, which produces $t = \frac{0.0671}{0.0159} = 4.220$ and $p = 0.0022$). The true per-seed standard deviation across the ten matched seeds is strictly **$s_d = 0.0612$**, which yields strictly **$t = 3.467$** and **$p = 0.007077$**. This value is identical to the verified historical record in [research_audit/phase10_unet_final/PHASE10_STATISTICAL_VALIDATION.md#L53](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/phase10_unet_final/PHASE10_STATISTICAL_VALIDATION.md#L53).

---

## 4. Verification of All Six Primary BUS-BRA Paired Comparisons

All six planned comparisons on the primary endpoint (Zero-Shot BUS-BRA Foreground IoU, $N=10$ matched seeds) were recomputed directly from `SRC-01`, `SRC-02`, `SRC-03`, and `SRC-04`:

| Comparison Title | Model A | Model B | Mean Diff ($\text{pp}$) | SD Diff ($\text{pp}$) | SEM ($\text{pp}$) | 95% Confidence Interval | Paired $t$ ($df=9$) | Two-Tailed Raw $p_t$ | Wilcoxon $W$ | Wilcoxon Raw $p_W$ | Cohen's $d_z$ |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Swin-Tiny vs. Attention U-Net** | Swin-Tiny | Attn U-Net | **$+12.15\text{ pp}$** | $7.61$ | $2.40$ | $[+6.71\text{ pp}, +17.59\text{ pp}]$ | $+5.052$ | **$0.000688$** | $1.0$ | $0.003906$ | $+1.60$ |
| **2. Swin-Tiny vs. Standard U-Net** | Swin-Tiny | Std U-Net | **$+6.71\text{ pp}$** | $6.12$ | $1.94$ | $[+2.33\text{ pp}, +11.09\text{ pp}]$ | $+3.467$ | **$0.007077$** | $4.0$ | $0.013672$ | $+1.10$ |
| **3. Swin-Tiny vs. ResNet-50 U-Net** | Swin-Tiny | ResNet-50 | **$+7.14\text{ pp}$** | $9.39$ | $2.97$ | $[+0.42\text{ pp}, +13.86\text{ pp}]$ | $+2.405$ | **$0.039569$** | $8.0$ | $0.048828$ | $+0.76$ |
| **4. Standard U-Net vs. Attention U-Net** | Std U-Net | Attn U-Net | **$+5.44\text{ pp}$** | $10.09$ | $3.19$ | $[-1.78\text{ pp}, +12.66\text{ pp}]$ | $+1.704$ | **$0.122497$** | $13.0$ | $0.160156$ | $+0.54$ |
| **5. Swin-MoE vs. Swin-Tiny** | Swin-MoE | Swin-Tiny | **$+0.35\text{ pp}$** | $4.81$ | $1.52$ | $[-3.09\text{ pp}, +3.79\text{ pp}]$ | $+0.232$ | **$0.821822$** | $25.0$ | $0.845703$ | $+0.07$ |
| **6. Standard U-Net vs. ResNet-50 U-Net** | Std U-Net | ResNet-50 | **$+0.43\text{ pp}$** | $8.06$ | $2.55$ | $[-5.33\text{ pp}, +6.19\text{ pp}]$ | $+0.169$ | **$0.869227$** | $27.0$ | $1.000000$ | $+0.05$ |

---

## 5. Verification of the In-Domain ResNet-50 vs. Standard U-Net Comparison

- **Source Observations**: `SRC-07` (`UNET_INDOMAIN_TEST_RESULTS.csv`) vs `SRC-09` (`RESNET50_UNET_INDOMAIN_TEST_RESULTS.csv`).
- **ResNet-50 U-Net Mean**: $59.2616\% \pm 4.3468\%$
- **Standard U-Net Mean**: $56.0582\% \pm 3.8811\%$
- **Mean Paired Difference**: $\bar{d} = \mathbf{+3.2034\text{ pp}}$ ($+0.032034$)
- **Sample SD of Differences ($ddof=1$)**: $s_d = \mathbf{7.2201\text{ pp}}$ ($0.072201$)
- **Standard Error of Mean Difference**: $\text{SEM} = \mathbf{2.2832\text{ pp}}$ ($0.022832$)
- **Cohen's $d_z$**: $\mathbf{+0.4437}$
- **95% Confidence Interval**: $\mathbf{[-1.9602\text{ pp}, +8.3669\text{ pp}]}$ (Crosses zero)
- **Paired $t$-statistic**: $t = \mathbf{1.403046}$ ($df = 9$)
- **Two-Tailed Raw $p$-value**: $p = \mathbf{0.194148}$
- **Wilcoxon Signed-Rank Test**: $W = \mathbf{16.0}$, Two-Tailed $p = \mathbf{0.275391}$
- **Holm-Adjusted $p$-value**: Under the post-hoc in-domain contrast family ($m=3$), this test is rank 2 (multiplier 2), yielding $p_{\text{adj}} = 0.194148 \times 2 = \mathbf{0.388296}$.

> [!IMPORTANT] Pre-Specification Audit Finding
> The family of $m=3$ in-domain model contrasts was **not pre-specified** in Phase 10 or Phase 12 training configs; it was formulated post hoc during cross-phase synthesis. Furthermore, because the raw $p$-value is $0.1941 > 0.05$, the difference is **statistically non-significant under every family definition** (unadjusted or adjusted).

---

## 6. Comprehensive Discrepancy and Audit Resolution Table

The table below documents every discrepancy identified between earlier reports, the previous turn draft, and the verified canonical raw data:

| Metric / Comparison | Earlier Value | Latest Turn Draft | Raw-Data Verified Value | Root Cause & Auditor Explanation |
|---|---|---|---|---|
| **Swin-MoE Total Parameters** | $81.15\text{M}$ | $33.64\text{M}$ ($28.32\text{M}$) | **$81.15\text{M}$ ($81,148,989$) total, $46.64\text{M}$ active** | Single-stage exploratory configuration (Stage 3 only) was mistakenly transcribed into Phase 12 text. The actual Phase 9B model placed 4-expert MoE across all 12 blocks ($12 \times 4 = 48$ experts). |
| **Swin-Tiny vs Std U-Net $t$-statistic** | $4.220$ ($p=0.0022$) | $3.467$ ($p=0.0071$) | **$t = 3.467473$, $p = 0.007077$** | The $4.220$ value was an unverified draft calculation that assumed an artificially low difference SD ($s_d = 5.03$ vs true $6.12$). Recomputed raw observations yield strictly $t = 3.467$. |
| **Swin-Tiny vs ResNet-50 Recall $p$-value** | $0.0083$ | $0.0094$ | **$p = 0.009425$ ($t = 3.287, df=9$)** | The value $0.0083$ was a transcription error from row 37 of `CANONICAL_PAIRED_STATISTICAL_RESULTS.csv`, which documented the *mean difference* between MoE and Swin-Tiny on Recall ($+0.00833$). |
| **Swin-Tiny vs ResNet-50 Recall Holm $p$** | $0.0498$ | $0.0498 \dots 0.0660$ | **$p_{\text{adj}} = 0.0377$ (Family 2, $m=6$), $0.0660$ (Family 3, $m=7$)** | $0.0498$ was an invalid multiplication of the erroneous $p=0.0083 \times 6$. The range was an artifact of mixing families. |
| **Swin-Tiny vs ResNet-50 FG IoU $p$-value** | $0.0394$ | $0.0396$ | **$p = 0.039569$ ($t = 2.405, df=9$)** | Minor four-decimal rounding error ($0.0394$ vs true $0.0396$). |
| **Swin-Tiny In-Domain FG IoU** | $60.52 \pm 5.43\%$ | $61.50 \pm 3.50\%$ | **$60.24 \pm 6.11\%$** | Early drafts used placeholder or rounded values. The verified value is the exact arithmetic mean of the 10 raw `test_metrics.json` files in `outputs/results_phase9_auxloss_corrected/outputs/`. |
| **Swin-MoE In-Domain FG IoU** | $60.91 \pm 4.12\%$ | $62.00 \pm 2.80\%$ | **$60.58 \pm 5.28\%$** | Same as above. Computed directly from the 10 raw `test_metrics.json` files. |
| **Standard U-Net In-Domain FG IoU SD** | $56.06 \pm 4.96\%$ | $56.06 \pm 3.88\%$ | **$56.06 \pm 3.88\%$** | $4.96\%$ was transcribed from an earlier exploratory seed subset. True sample SD ($ddof=1$) across the 10 seeds in `UNET_INDOMAIN_TEST_RESULTS.csv` is strictly $3.88\%$. |
| **Attention U-Net In-Domain FG IoU** | $55.33 \pm 3.79\%$ | $56.10 \pm 2.53\%$ | **$56.10 \pm 2.53\%$** | $55.33\%$ was an erroneous transcription of Standard U-Net's Seed 42 zero-shot IoU. Recomputed from `ATTENTION_UNET_INDOMAIN_TEST_RESULTS.csv` (`Test_Foreground_IoU`). |
| **ResNet-50 In-Domain Pretraining Claim** | "Statistically significant improvement" | "Supported by evidence" | **Not statistically significant ($p = 0.1941$)** | Overstatement based on training/validation convergence. Paired test on held-out BUSI test FG IoU yields $t = 1.403, p = 0.1941, 95\%\text{ CI}: [-1.96\%, +8.37\%]$, crossing zero. |
| **BUS-BRA Evaluation Cohort Size** | $N = 1,063$ | $N = 1,063$ | **$N = 1,875$ images ($1,063$ patients)** | The BUS-BRA publication cites $1,063$ patients, but the dataset contains $1,875$ slice images. All $1,875$ images were evaluated per seed ($94,080,000$ evaluated pixels per seed). |
| **2D Volume Ratio Formulation** | $0.97\times$ | $1.10\times$ | **$1.1005 \pm 0.1805$ (2D Pixel-Count Ratio)** | $0.97\times$ was an inverted ratio ($\text{GT}/\text{Pred}$) from early notes. Terminology corrected to 2D pixel-count ratio, never clinical volume. |

---
*All source-level discrepancies are resolved and backed by raw data. This document represents the permanent audit record.*
