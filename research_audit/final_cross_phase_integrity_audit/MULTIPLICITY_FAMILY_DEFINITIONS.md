# Multiplicity Family Definitions and Statistical Correction Protocol

**Audit Protocol:** Five-Model Medical Image Segmentation Benchmark (Phases 9B, 10, 11, and 12)  
**Lead Biostatistician & Auditor:** Independent ML Research Auditor & Biostatistician  
**Workspace:** `D:\Study Material\Research\CV\vision_transformer_research`  
**Associated Canonical Dataset:** [FINAL_CANONICAL_PAIRED_TESTS.csv](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/final_cross_phase_integrity_audit/FINAL_CANONICAL_PAIRED_TESTS.csv)

---

## 1. Context and Problem Statement

When evaluating multiple machine learning architectures across multiple segmentation metrics and datasets, unadjusted hypothesis testing inflates the Family-Wise Error Rate (FWER). If $\alpha = 0.05$ is maintained per test without correction, the probability of at least one false positive across $m$ independent null hypotheses is $1 - (1 - \alpha)^m$ (e.g., $30.2\%$ for $m=7$, $26.5\%$ for $m=6$, and $88.8\%$ for $m=42$).

In earlier project phases, different reports applied multiplicity corrections across different sets of hypotheses:
- Phase 12 statistical audit applied Holm-Bonferroni across the 7 segmentation metrics for a single model pair ($m=7$).
- Other informal summaries attempted to correct across model comparisons ($m=6$), or cited unadjusted $p$-values alongside adjusted ranges ($0.0498$ to $0.0660$).

To eliminate ambiguity, this document explicitly defines every multiplicity family, specifies whether it was pre-specified/planned or exploratory, details the exact step-down Holm-Bonferroni calculation, and provides definitive adjusted $p$-values.

---

## 2. Multiplicity Correction Formulation (Holm-Bonferroni Step-Down)

For a family of $m$ null hypotheses $H_{(1)}, H_{(2)}, \dots, H_{(m)}$ with corresponding ordered two-tailed raw $p$-values:
$$p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$$

The Holm-Bonferroni adjusted $p$-values $p_{\text{adj},(k)}$ are calculated sequentially as:
$$p_{\text{adj},(k)} = \max_{j \le k} \left( \min \left[ (m - j + 1) \cdot p_{(j)}, 1.0 \right] \right)$$

A hypothesis $H_{(k)}$ is rejected at family-wise significance level $\alpha = 0.05$ if and only if:
$$p_{\text{adj},(k)} < 0.05$$

The step-down procedure stops at the first index $k$ where $(m - k + 1) \cdot p_{(k)} \ge \alpha$. All subsequent hypotheses are retained.

---

## 3. Explicit Multiplicity Families

### Family 1: Planned Model Contrast Family on the Primary Endpoint (Zero-Shot BUS-BRA Foreground IoU)
- **Status:** **Planned / Confirmatory Hypothesis Family** (Defined before inspecting Phase 12 results based on the 6 planned architecture pairs).
- **Primary Endpoint:** Foreground IoU ($\text{FG IoU}$) on the external zero-shot BUS-BRA cohort ($N=1,875$ images, 10 matched seeds).
- **Family Size:** $m = 6$ planned pairwise comparisons.

#### Hypotheses Included:
1. $H_{0,1}$: Swin-Tiny = ResNet-50 U-Net (Primary test of ViT vs Pretrained CNN under matched ImageNet-1K pretraining).
2. $H_{0,2}$: Standard U-Net = ResNet-50 U-Net (Isolated effect of ImageNet-1K pretraining within CNN U-Net).
3. $H_{0,3}$: Swin-Tiny = Standard U-Net (Pretrained ViT vs Canonical scratch CNN baseline).
4. $H_{0,4}$: Swin-Tiny = Attention U-Net (Pretrained ViT vs Attention-gated scratch CNN).
5. $H_{0,5}$: Standard U-Net = Attention U-Net (Isolated effect of Oktay et al. skip attention gates).
6. $H_{0,6}$: Standard-MLP Swin-MoE = Vanilla Swin-Tiny (Isolated effect of sparsely-gated MoE routing).

#### Canonical Step-Down Holm Adjustment ($m=6$):

| Rank ($k$) | Comparison | Mean Diff ($\Delta$) | Paired $t$ ($df=9$) | Raw $p$ ($p_{(k)}$) | Multiplier ($m - k + 1$) | Uncapped Adjusted Step | Exact Holm $p_{\text{adj}}$ | Significant at $\alpha = 0.05$? |
|---|---|---|---|---|---|---|---|---|
| 1 | Swin-Tiny vs. Attention U-Net | $+12.15\%$ | $5.052$ | $0.000688$ | $6$ | $0.004131$ | **$0.004131$** | **YES ($p < 0.01$)** |
| 2 | Swin-Tiny vs. Standard U-Net | $+6.71\%$ | $3.467$ | $0.007077$ | $5$ | $0.035385$ | **$0.035385$** | **YES ($p < 0.05$)** |
| 3 | Swin-Tiny vs. ResNet-50 U-Net | $+7.14\%$ | $2.405$ | $0.039569$ | $4$ | $0.158274$ | **$0.158274$** | **NO (Non-Significant)** |
| 4 | Standard U-Net vs. Attention U-Net | $+5.44\%$ | $1.704$ | $0.122497$ | $3$ | $0.367492$ | **$0.367492$** | **NO (Non-Significant)** |
| 5 | Swin-MoE vs. Swin-Tiny | $+0.35\%$ | $0.232$ | $0.821822$ | $2$ | $1.000000$ | **$1.000000$** | **NO (Non-Significant)** |
| 6 | Standard U-Net vs. ResNet-50 U-Net | $+0.43\%$ | $0.169$ | $0.869227$ | $1$ | $0.869227$ | **$1.000000$** | **NO (Non-Significant)** |

> [!IMPORTANT] Takeaway on Primary Hypothesis
> When multiplicity is controlled across the 6 planned model contrasts on the primary endpoint (FG IoU), **Swin-Tiny vs. ResNet-50 U-Net fails to achieve statistical significance ($p_{\text{adj}} = 0.1583 > 0.05$)**.

---

### Family 2: Planned Model Contrast Family on Key Secondary Endpoints ($m=6$ Each)
- **Status:** **Planned Secondary Families** (Evaluated across the same 6 model pairs for individual secondary metrics).

#### A. Foreground Recall Family ($m=6$):
| Rank ($k$) | Comparison | Mean Diff ($\Delta$) | Paired $t$ ($df=9$) | Raw $p$ | Multiplier | Exact Holm $p_{\text{adj}}$ | Significant at $\alpha = 0.05$? |
|---|---|---|---|---|---|---|---|
| 1 | Swin-Tiny vs. Standard U-Net | $+17.94\%$ | $6.218$ | $0.000155$ | $6$ | **$0.000932$** | **YES ($p < 0.001$)** |
| 2 | Swin-Tiny vs. Attention U-Net | $+25.19\%$ | $5.391$ | $0.000438$ | $5$ | **$0.002192$** | **YES ($p < 0.01$)** |
| 3 | Swin-Tiny vs. ResNet-50 U-Net | $+13.68\%$ | $3.287$ | $0.009425$ | $4$ | **$0.037700$** | **YES ($p < 0.05$)** |
| 4 | Standard U-Net vs. Attention U-Net | $+7.24\%$ | $1.611$ | $0.141600$ | $3$ | **$0.424801$** | **NO (Non-Significant)** |
| 5 | Standard U-Net vs. ResNet-50 U-Net | $-4.27\%$ | $-1.047$ | $0.322507$ | $2$ | **$0.645015$** | **NO (Non-Significant)** |
| 6 | Swin-MoE vs. Swin-Tiny | $+0.83\%$ | $0.428$ | $0.678486$ | $1$ | **$0.678486$** | **NO (Non-Significant)** |

#### B. Foreground Precision Family ($m=6$):
| Rank ($k$) | Comparison | Mean Diff ($\Delta$) | Paired $t$ ($df=9$) | Raw $p$ | Multiplier | Exact Holm $p_{\text{adj}}$ | Significant at $\alpha = 0.05$? |
|---|---|---|---|---|---|---|---|
| 1 | Standard U-Net vs. Swin-Tiny | $+9.75\%$ | $3.958$ | $0.003315$ | $6$ | **$0.019888$** | **YES ($p < 0.05$)** |
| 2 | Standard U-Net vs. ResNet-50 U-Net | $+7.57\%$ | $3.884$ | $0.003716$ | $5$ | **$0.019888$** | **YES ($p < 0.05$)** |
| 3 | Attention U-Net vs. Swin-Tiny | $+10.94\%$ | $3.457$ | $0.007197$ | $4$ | **$0.028789$** | **YES ($p < 0.05$)** |
| 4 | Swin-MoE vs. Swin-Tiny | $-0.52\%$ | $-0.320$ | $0.756281$ | $3$ | **$1.000000$** | **NO (Non-Significant)** |
| 5 | Attention U-Net vs. Standard U-Net | $+1.19\%$ | $0.618$ | $0.551842$ | $2$ | **$1.000000$** | **NO (Non-Significant)** |
| 6 | Swin-Tiny vs. ResNet-50 U-Net | $-2.18\%$ | $-0.696$ | $0.503823$ | $1$ | **$1.000000$** | **NO (Non-Significant)** |

---

### Family 3: Per-Contrast Metric Battery Family ($m=7$)
- **Status:** **Secondary Protocol** (Applied within each pairwise comparison across all 7 segmentation metrics: Foreground IoU, Foreground Dice, Foreground Recall, Foreground Precision, mIoU, mDice, Pixel Accuracy).
- **Rationale:** Controls error inflation when examining multiple summary endpoints for a single architecture pair.

#### For Comparison: Vanilla Swin-Tiny vs. ResNet-50 U-Net ($m=7$):
| Rank ($k$) | Metric | Mean Diff ($\Delta$) | Paired $t$ ($df=9$) | Raw $p$ | Multiplier ($7 - k + 1$) | Exact Holm $p_{\text{adj}}$ | Status |
|---|---|---|---|---|---|---|---|
| 1 | Foreground Recall | $+13.68\%$ | $3.287$ | $0.009425$ | $7$ | **$0.065975$** | **Non-Significant Trend ($p < 0.10$)** |
| 2 | Foreground IoU | $+7.14\%$ | $2.405$ | $0.039569$ | $6$ | **$0.237411$** | **Non-Significant ($p = 0.24$)** |
| 3 | Foreground Dice | $+6.19\%$ | $2.324$ | $0.045175$ | $5$ | **$0.237411$** | **Non-Significant ($p = 0.24$)** |
| 4 | Macro mIoU | $+3.75\%$ | $2.231$ | $0.052571$ | $4$ | **$0.237411$** | **Non-Significant ($p = 0.24$)** |
| 5 | Macro mDice | $+3.19\%$ | $2.223$ | $0.053321$ | $3$ | **$0.237411$** | **Non-Significant ($p = 0.24$)** |
| 6 | Pixel Accuracy | $+0.41\%$ | $0.933$ | $0.375071$ | $2$ | **$0.750142$** | **Non-Significant** |
| 7 | Foreground Precision | $-2.18\%$ | $-0.696$ | $0.503823$ | $1$ | **$0.750142$** | **Non-Significant** |

> [!NOTE] Resolution of Adjusted Recall Range
> Under Family 2 (contrast family, $m=6$), Recall achieves significance at $p_{\text{adj}} = 0.0377$. Under Family 3 (metric battery family, $m=7$), Recall is a trend at $p_{\text{adj}} = 0.0660$. The earlier cited range "$0.0498$ to $0.0660$" was caused by an intermediate calculation error where $p=0.0083$ was multiplied by 6 ($0.0498$). The definitive values are strictly **$0.0377$** (Family 2) and **$0.0660$** (Family 3).

---

### Family 4: In-Domain BUSI Test Contrast Family ($m=3$)
- **Status:** **Planned In-Domain Family** (Comparisons on held-out in-domain test set, $N=78$ images, 10 seeds).
- **Primary Endpoint:** In-domain Foreground IoU.

| Rank ($k$) | Comparison | Mean Diff ($\Delta$) | Paired $t$ ($df=9$) | Raw $p$ | Multiplier ($3 - k + 1$) | Exact Holm $p_{\text{adj}}$ | Significant at $\alpha = 0.05$? |
|---|---|---|---|---|---|---|---|
| 1 | ResNet-50 U-Net vs. Attention U-Net | $+3.16\%$ | $1.842$ | $0.098534$ | $3$ | **$0.295603$** | **NO (Non-Significant)** |
| 2 | ResNet-50 U-Net vs. Standard U-Net | $+3.20\%$ | $1.403$ | $0.194148$ | $2$ | **$0.388296$** | **NO (Non-Significant)** |
| 3 | Standard U-Net vs. Attention U-Net | $-0.04\%$ | $-0.028$ | $0.978650$ | $1$ | **$0.978650$** | **NO (Non-Significant)** |

---

### Family 5: Global Exploratory Omnibus Family ($m=42$)
- **Status:** **Post-Hoc Exploratory Family** (Fully unconstrained two-way family across all 6 model pairs and 7 metrics).
- **Family Size:** $m = 6 \times 7 = 42$ tests.

Under this highly conservative exploratory benchmark:
- Swin-Tiny vs. Standard U-Net Recall ($p_{\text{raw}} = 0.000155$) remains **significant** ($p_{\text{adj}} = 0.006527 < 0.01$).
- Swin-Tiny vs. Attention U-Net Recall ($p_{\text{raw}} = 0.000438$) remains **significant** ($p_{\text{adj}} = 0.017975 < 0.05$).
- Swin-Tiny vs. Attention U-Net FG IoU ($p_{\text{raw}} = 0.000688$) remains **significant** ($p_{\text{adj}} = 0.027539 < 0.05$).
- Swin-Tiny vs. Standard U-Net FG IoU ($p_{\text{raw}} = 0.007077$) becomes **non-significant** ($p_{\text{adj}} = 0.240615$).
- Swin-Tiny vs. ResNet-50 U-Net Recall ($p_{\text{raw}} = 0.009425$) becomes **non-significant** ($p_{\text{adj}} = 0.292176$).
- Swin-Tiny vs. ResNet-50 U-Net FG IoU ($p_{\text{raw}} = 0.039569$) becomes **non-significant** ($p_{\text{adj}} = 1.000000$).

---

## 4. Summary Matrix of Corrected Multiplicity Results

| Contrast Pair | Metric | Raw $p$ | Family 1 / 2 ($m=6$) | Family 3 ($m=7$) | Family 5 ($m=42$) | Definitive Statistical Decision |
|---|---|---|---|---|---|---|
| **Swin-Tiny vs. ResNet-50 U-Net** | Foreground IoU | $0.0396$ | $0.1583$ | $0.2374$ | $1.0000$ | **Descriptive difference only; Non-significant** |
| **Swin-Tiny vs. ResNet-50 U-Net** | Foreground Recall | $0.0094$ | $0.0377$* | $0.0660$ | $0.2922$ | **Directional trend ($d_z = 1.04$); Non-significant under primary battery** |
| **Swin-Tiny vs. Standard U-Net** | Foreground IoU | $0.0071$ | $0.0354$ | $0.0354$ | $0.2406$ | **Statistically Significant ($p < 0.05$)** |
| **Swin-Tiny vs. Standard U-Net** | Foreground Recall | $0.0002$ | $0.0009$ | $0.0011$ | $0.0065$ | **Statistically Significant ($p < 0.01$)** |
| **Standard U-Net vs. ResNet-50 U-Net** | Foreground Precision | $0.0037$ | $0.0199$ | $0.0260$ | $0.1449$ | **Statistically Significant ($p < 0.05$)** |
| **Standard U-Net vs. ResNet-50 U-Net** | Foreground IoU | $0.8692$ | $1.0000$ | $1.0000$ | $1.0000$ | **Null hypothesis not rejected ($p = 0.87$)** |
| **ResNet-50 vs. Standard U-Net (BUSI)**| In-Domain FG IoU | $0.1941$ | $0.3883$ ($m=3$) | $0.7766$ ($m=7$) | — | **Null hypothesis not rejected ($p = 0.19$)** |
| **Standard U-Net vs. Attention U-Net** | Foreground IoU | $0.1225$ | $0.3675$ | $0.7168$ | $1.0000$ | **Null hypothesis not rejected ($p = 0.12$)** |
| **Swin-MoE vs. Vanilla Swin-Tiny** | Foreground IoU | $0.8218$ | $1.0000$ | $1.0000$ | $1.0000$ | **Null hypothesis not rejected ($p = 0.82$)** |

*\*Note on Primary Manuscript Guidance:* Because the primary endpoint (FG IoU) fails to achieve significance under any multiplicity family ($p_{\text{adj}} \ge 0.158$), the conservative and rigorous reporting standard requires that the secondary Recall endpoint be reported as a strong directional trend ($d_z = 1.04$), rather than an isolated confirmatory discovery.
