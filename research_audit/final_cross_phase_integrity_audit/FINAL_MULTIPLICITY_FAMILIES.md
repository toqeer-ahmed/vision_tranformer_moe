# Final Multiplicity Families and Statistical Correction Protocol

**Audit Protocol:** Five-Model Medical Image Segmentation Benchmark (Phases 9B, 10, 11, and 12)  
**Lead Biostatistician & Auditor:** Independent ML Research Auditor & Biostatistician  
**Workspace:** `D:\Study Material\Research\CV\vision_transformer_research`  
**Associated Canonical Dataset:** [FINAL_VERIFIED_PAIRED_TESTS.csv](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/final_cross_phase_integrity_audit/FINAL_VERIFIED_PAIRED_TESTS.csv)

---

## 1. Problem Statement & Multiplicity Discipline

Evaluating 5 machine learning architectures across multiple segmentation metrics and datasets generates a substantial multiplicity burden. If $\alpha = 0.05$ is maintained per individual hypothesis test without adjustment, the Family-Wise Error Rate (FWER) inflates dramatically:
$$\text{FWER} = 1 - (1 - \alpha)^m$$
- For $m = 3$ tests: $\text{FWER} = 14.3\%$
- For $m = 6$ tests: $\text{FWER} = 26.5\%$
- For $m = 7$ tests: $\text{FWER} = 30.2\%$
- For $m = 42$ tests: $\text{FWER} = 88.8\%$

In earlier audit notes, different reports cited conflicting multiplicity-adjusted numbers (e.g., $p_{\text{adj}} = 0.0498$ to $0.0660$ for Swin-Tiny vs. ResNet-50 Recall). This document resolves all ambiguity by establishing:
1. The mathematical definition of the Holm-Bonferroni step-down procedure.
2. The exact pre-specification status of each multiplicity family (pre-specified vs. post-hoc exploratory).
3. The definitive, verified step-down calculations for all families.

---

## 2. Mathematical Formulation: Holm-Bonferroni Step-Down

For a family of $m$ null hypotheses $H_{(1)}, H_{(2)}, \dots, H_{(m)}$ with corresponding ordered two-tailed raw $p$-values:
$$p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$$

The Holm-Bonferroni adjusted $p$-value at rank $k$ ($1 \le k \le m$) is computed as:
$$p_{\text{adj},(k)} = \max_{j \le k} \left( \min \left[ (m - j + 1) \cdot p_{(j)}, 1.0 \right] \right)$$

A hypothesis $H_{(k)}$ is rejected at family-wise significance level $\alpha = 0.05$ if and only if $p_{\text{adj},(k)} < 0.05$.  
The step-down procedure terminates at the first index $k$ where $(m - k + 1) \cdot p_{(k)} \ge \alpha$; all subsequent hypotheses in the family are retained as non-significant.

---

## 3. Pre-Specification Audit and Git Commit Evidence

To adhere to open-science and biostatistical standards, an analysis protocol cannot be deemed "pre-specified" simply because it appears in a retrospective analysis script. A truly pre-specified or confirmatory analysis requires an explicit protocol or pre-registration frozen in a versioned repository *before* any experimental outcomes on the evaluated cohort are observed.

### Audit of Earliest Git Commits Defining Families A–E:

| Family | Analysis Scope | Earliest Defining Artifact | Earliest Git Commit & Date | Were Results Examined Prior to Definition? | Formal Classification |
|---|---|---|---|:---:|---|
| **Family A** | Per-Contrast 7-Metric Battery ($m=7$) | `PHASE10_STATISTICAL_VALIDATION.md` | Commit `d31a183` (2026-11-25) | **YES** (Phase 10 raw U-Net results had already been inspected) | **Exploratory / Post-Training Metric Battery** |
| **Family B** | Primary Model Contrasts on FG IoU ($m=6$) | `MULTIPLICITY_FAMILY_DEFINITIONS.md` | Commit `0d47103` (2026-11-26) | **YES** (Formulated after Phase 12 results were computed) | **Exploratory / Post-Hoc Synthesis Family** |
| **Family C** | Secondary Metric Contrasts ($m=6$ each) | `MULTIPLICITY_FAMILY_DEFINITIONS.md` | Commit `0d47103` (2026-11-26) | **YES** (Formulated after Phase 12 results were computed) | **Exploratory / Post-Hoc Secondary Family** |
| **Family D** | In-Domain BUSI Contrasts ($m=3$) | `MULTIPLICITY_FAMILY_DEFINITIONS.md` | Commit `0d47103` (2026-11-26) | **YES** (Formulated during cross-phase synthesis) | **Exploratory / Post-Hoc In-Domain Family** |
| **Family E** | Global Omnibus ($m=42$) | `MULTIPLICITY_FAMILY_DEFINITIONS.md` | Commit `0d47103` (2026-11-26) | **YES** (Formulated during cross-phase synthesis) | **Exploratory / Post-Hoc Omnibus Benchmark** |

> [!IMPORTANT] Biostatistical Standard on Pre-Specification
> No prospective pre-registration document was committed prior to running the training pipelines or evaluating test sets. Consequently, **all multiplicity families in this study are formally exploratory / post hoc**. Historical analysis scripts (`audit_phase12_statistical_analysis.py`, `PHASE10_STATISTICAL_VALIDATION.md`) represent retrospective analytical protocols rather than a priori pre-specifications.

### Why Six Contrasts Were Selected in Family B (Instead of All Ten Possible Pairs):

Among five architectures, there are $\binom{5}{2} = \frac{5 \times 4}{2} = 10$ possible pairwise contrasts. Family B evaluates exactly **six**:
1. **Vanilla Swin-Tiny vs. ResNet-50 U-Net** (Phase 12: ViT vs. CNN under matched ImageNet-1K pretraining).
2. **Vanilla Swin-Tiny vs. Standard U-Net** (Phase 10: Pretrained ViT vs. canonical scratch CNN baseline).
3. **Vanilla Swin-Tiny vs. Attention U-Net** (Phase 11: Dense self-attention ViT vs. skip-attention scratch CNN).
4. **Standard U-Net vs. Attention U-Net** (Phase 11: Isolated effect of adding Oktay et al. skip attention gates to canonical U-Net).
5. **Standard U-Net vs. ResNet-50 U-Net** (Phase 12: Scratch CNN baseline vs. pretrained residual CNN).
6. **Standard-MLP Swin-MoE vs. Vanilla Swin-Tiny** (Phase 9B: Isolated effect of replacing dense MLPs with Top-2 MoE routing).

**The Four Omitted Pairwise Contrasts:**
1. *Standard-MLP Swin-MoE vs. Standard U-Net*
2. *Standard-MLP Swin-MoE vs. Attention U-Net*
3. *Standard-MLP Swin-MoE vs. ResNet-50 U-Net*
4. *Attention U-Net vs. ResNet-50 U-Net*

**Rationale for Omission:**
The six included contrasts map directly to the sequential, single-factor experimental questions posed chronologically in Phases 9B through 12. Each of the four omitted comparisons alters multiple major architectural dimensions simultaneously (for example, comparing Swin-MoE to Standard U-Net confounds vision transformer self-attention, sparse Top-2 mixture-of-experts routing, and ImageNet pretraining against a scratch convolutional baseline). Because these four pairs do not isolate any single architectural mechanism, they were not formulated as primary contrasts. Nonetheless, because this restriction to 6 contrasts was established during cross-phase synthesis rather than in an a priori pre-registered statistical analysis plan, Family B must be reported as a **planned scientific contrast subset within a post-hoc synthesis**, not a pre-registered confirmatory family.

---

## 4. Family A: Per-Contrast Metric Battery ($m=7$)

- **Pre-Specification Status:** **Exploratory / Retrospective Metric Battery** (First implemented in Phase 10 audit script `PHASE10_STATISTICAL_VALIDATION.md` at `d31a183`).
- **Scientific Purpose:** Controls error inflation when evaluating a single architecture comparison across all 7 segmentation metrics: Foreground IoU, Foreground Dice, Foreground Recall, Foreground Precision, Macro mIoU, Macro mDice, and Pixel Accuracy.
- **Family Size:** $m = 7$.

### Primary Evaluation Contrast: Vanilla Swin-Tiny vs. ResNet-50 U-Net ($m=7$)

| Rank ($k$) | Metric | Mean Diff (pp) | Paired $t$ ($df=9$) | Raw $p_{(k)}$ | Multiplier ($7 - k + 1$) | Uncapped Step | Exact Holm $p_{\text{adj}}$ | Significance Decision ($\alpha=0.05$) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | Foreground Recall | $+13.68$ | $+3.287$ | $0.009425$ | $7$ | $0.065975$ | **$0.065975$** | **Non-Significant Trend ($p < 0.10$)** |
| **2** | Foreground IoU | $+7.14$ | $+2.405$ | $0.039569$ | $6$ | $0.237411$ | **$0.237411$** | **Non-Significant ($p > 0.05$)** |
| **3** | Foreground Dice | $+6.19$ | $+2.324$ | $0.045175$ | $5$ | $0.225875$ | **$0.237411$** | **Non-Significant ($p > 0.05$)** |
| **4** | Macro mIoU | $+3.75$ | $+2.231$ | $0.052571$ | $4$ | $0.210284$ | **$0.237411$** | **Non-Significant ($p > 0.05$)** |
| **5** | Macro mDice | $+3.19$ | $+2.223$ | $0.053321$ | $3$ | $0.159963$ | **$0.237411$** | **Non-Significant ($p > 0.05$)** |
| **6** | Pixel Accuracy | $+0.41$ | $+0.933$ | $0.375071$ | $2$ | $0.750142$ | **$0.750142$** | **Non-Significant ($p > 0.05$)** |
| **7** | Foreground Precision | $-2.18$ | $-0.696$ | $0.503823$ | $1$ | $0.503823$ | **$0.750142$** | **Non-Significant ($p > 0.05$)** |

> [!NOTE] Monotonicity Enforcement in Step-Down
> At rank 3 (Dice), $(7 - 3 + 1) \cdot 0.045175 = 0.225875$, which is less than rank 2 ($0.237411$). The step-down definition enforces monotonicity via $\max_{j \le k}$, holding $p_{\text{adj}} = 0.237411$.

---

## 5. Family B: Planned Primary Model Contrast Family on FG IoU ($m=6$)

- **Pre-Specification Status:** **Exploratory / Post-Hoc Analytical Synthesis Family** (Primary model contrasts across the 5 architectures, formulated post-experimentally during cross-phase synthesis at commit `0d47103`).
- **Primary Endpoint:** Foreground IoU on the external zero-shot BUS-BRA cohort ($N=1,875$ images, 10 matched seeds).
- **Family Size:** $m = 6$ model comparisons.

| Rank ($k$) | Comparison | Mean Diff (pp) | Paired $t$ ($df=9$) | Raw $p_{(k)}$ | Multiplier ($6 - k + 1$) | Uncapped Step | Exact Holm $p_{\text{adj}}$ | Significant at $\alpha=0.05$? |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | Swin-Tiny vs. Attention U-Net | $+12.15$ | $+5.052$ | $0.000688$ | $6$ | $0.004131$ | **$0.004131$** | **YES ($p < 0.01$)** |
| **2** | Swin-Tiny vs. Standard U-Net | $+6.71$ | $+3.467$ | $0.007077$ | $5$ | $0.035385$ | **$0.035385$** | **YES ($p < 0.05$)** |
| **3** | Swin-Tiny vs. ResNet-50 U-Net | $+7.14$ | $+2.405$ | $0.039569$ | $4$ | $0.158274$ | **$0.158274$** | **NO (Non-Significant)** |
| **4** | Standard U-Net vs. Attention U-Net | $+5.44$ | $+1.704$ | $0.122497$ | $3$ | $0.367492$ | **$0.367492$** | **NO (Non-Significant)** |
| **5** | Swin-MoE vs. Swin-Tiny | $+0.35$ | $+0.232$ | $0.821822$ | $2$ | $1.000000$ | **$1.000000$** | **NO (Non-Significant)** |
| **6** | Standard U-Net vs. ResNet-50 U-Net | $+0.43$ | $+0.169$ | $0.869227$ | $1$ | $0.869227$ | **$1.000000$** | **NO (Non-Significant)** |

> [!IMPORTANT] Biostatistical Takeaway on Primary ViT vs. Pretrained CNN Contrast
> When multiplicity is controlled across the 6 planned model contrasts on the primary endpoint (FG IoU), **Vanilla Swin-Tiny vs. ResNet-50 U-Net fails to achieve statistical significance ($p_{\text{adj}} = 0.1583 > 0.05$)**.

---

## 6. Family C: Secondary Model Contrast Families ($m=6$ Each)

- **Pre-Specification Status:** **Exploratory / Post-Hoc Secondary Families** (Evaluated across the 6 model contrasts for key secondary metrics, formulated during cross-phase synthesis at commit `0d47103`).

### A. Foreground Recall Family ($m=6$)

| Rank ($k$) | Comparison | Mean Diff (pp) | Paired $t$ ($df=9$) | Raw $p_{(k)}$ | Multiplier | Uncapped Step | Exact Holm $p_{\text{adj}}$ | Significant at $\alpha=0.05$? |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | Swin-Tiny vs. Standard U-Net | $+17.94$ | $+6.218$ | $0.000155$ | $6$ | $0.000932$ | **$0.000932$** | **YES ($p < 0.001$)** |
| **2** | Swin-Tiny vs. Attention U-Net | $+25.19$ | $+5.391$ | $0.000438$ | $5$ | $0.002192$ | **$0.002192$** | **YES ($p < 0.01$)** |
| **3** | Swin-Tiny vs. ResNet-50 U-Net | $+13.68$ | $+3.287$ | $0.009425$ | $4$ | $0.037700$ | **$0.037700$** | **YES ($p < 0.05$)** |
| **4** | Standard U-Net vs. Attention U-Net | $+7.24$ | $+1.611$ | $0.141600$ | $3$ | $0.424801$ | **$0.424801$** | **NO (Non-Significant)** |
| **5** | Standard U-Net vs. ResNet-50 U-Net | $-4.27$ | $-1.047$ | $0.322507$ | $2$ | $0.645015$ | **$0.645015$** | **NO (Non-Significant)** |
| **6** | Swin-MoE vs. Swin-Tiny | $+0.83$ | $+0.428$ | $0.678486$ | $1$ | $0.678486$ | **$0.678486$** | **NO (Non-Significant)** |

### B. Foreground Precision Family ($m=6$)

| Rank ($k$) | Comparison | Mean Diff (pp) | Paired $t$ ($df=9$) | Raw $p_{(k)}$ | Multiplier | Uncapped Step | Exact Holm $p_{\text{adj}}$ | Significant at $\alpha=0.05$? |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | Standard U-Net vs. Swin-Tiny | $+9.75$ | $+3.958$ | $0.003315$ | $6$ | $0.019888$ | **$0.019888$** | **YES ($p < 0.05$)** |
| **2** | Standard U-Net vs. ResNet-50 U-Net | $+7.57$ | $+3.884$ | $0.003716$ | $5$ | $0.018580$ | **$0.019888$** | **YES ($p < 0.05$)** |
| **3** | Attention U-Net vs. Swin-Tiny | $+10.94$ | $+3.457$ | $0.007197$ | $4$ | $0.028789$ | **$0.028789$** | **YES ($p < 0.05$)** |
| **4** | Swin-Tiny vs. ResNet-50 U-Net | $-2.18$ | $-0.696$ | $0.503823$ | $3$ | $1.000000$ | **$1.000000$** | **NO (Non-Significant)** |
| **5** | Attention U-Net vs. Standard U-Net | $+1.19$ | $+0.618$ | $0.551842$ | $2$ | $1.000000$ | **$1.000000$** | **NO (Non-Significant)** |
| **6** | Swin-MoE vs. Swin-Tiny | $-0.52$ | $-0.320$ | $0.756281$ | $1$ | $0.756281$ | **$1.000000$** | **NO (Non-Significant)** |

### C. Foreground Dice Family ($m=6$)

| Rank ($k$) | Comparison | Mean Diff (pp) | Paired $t$ ($df=9$) | Raw $p_{(k)}$ | Multiplier | Uncapped Step | Exact Holm $p_{\text{adj}}$ | Significant at $\alpha=0.05$? |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | Swin-Tiny vs. Attention U-Net | $+10.98$ | $+4.661$ | $0.001198$ | $6$ | $0.007188$ | **$0.007188$** | **YES ($p < 0.01$)** |
| **2** | Swin-Tiny vs. Standard U-Net | $+5.62$ | $+3.393$ | $0.007954$ | $5$ | $0.039770$ | **$0.039770$** | **YES ($p < 0.05$)** |
| **3** | Swin-Tiny vs. ResNet-50 U-Net | $+6.19$ | $+2.324$ | $0.045175$ | $4$ | $0.180700$ | **$0.180700$** | **NO (Non-Significant)** |
| **4** | Standard U-Net vs. Attention U-Net | $+5.36$ | $+1.745$ | $0.114949$ | $3$ | $0.344847$ | **$0.344847$** | **NO (Non-Significant)** |
| **5** | Swin-MoE vs. Swin-Tiny | $+0.37$ | $+0.312$ | $0.762410$ | $2$ | $1.000000$ | **$1.000000$** | **NO (Non-Significant)** |
| **6** | Standard U-Net vs. ResNet-50 U-Net | $+0.57$ | $+0.252$ | $0.806657$ | $1$ | $0.806657$ | **$1.000000$** | **NO (Non-Significant)** |

---

## 7. Family D: In-Domain BUSI Contrast Family ($m=3$)

- **Pre-Specification Status:** **Exploratory / Post-Hoc In-Domain Family** (Comparisons on held-out in-domain test set, $N=78$ images, 10 matched seeds, formulated during cross-phase synthesis at commit `0d47103`).
- **Endpoint:** In-Domain BUSI Test Foreground IoU.
- **Family Size:** $m = 3$ comparisons.

| Rank ($k$) | Comparison | Mean Diff (pp) | Paired $t$ ($df=9$) | Raw $p_{(k)}$ | Multiplier ($3 - k + 1$) | Uncapped Step | Exact Holm $p_{\text{adj}}$ | Significant at $\alpha=0.05$? |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | ResNet-50 U-Net vs. Attention U-Net | $+3.16$ | $+1.842$ | $0.098534$ | $3$ | $0.295603$ | **$0.295603$** | **NO (Non-Significant)** |
| **2** | ResNet-50 U-Net vs. Standard U-Net | $+3.20$ | $+1.403$ | $0.194148$ | $2$ | $0.388296$ | **$0.388296$** | **NO (Non-Significant)** |
| **3** | Standard U-Net vs. Attention U-Net | $-0.04$ | $-0.028$ | $0.978650$ | $1$ | $0.978650$ | **$0.978650$** | **NO (Non-Significant)** |

---

## 8. Family E: Global Exploratory Omnibus Family ($m=42$)

- **Pre-Specification Status:** **Post-Hoc Exploratory Omnibus Family** (Unconstrained across all 6 zero-shot model pairs and all 7 metrics: $6 \times 7 = 42$).
- **Scientific Rationale:** Identifies the ultra-conservative boundary if a reader or reviewer demands FWER control across every computed test simultaneously.

### Top 10 Ranked Tests in the Global Omnibus Family ($m=42$)

| Rank ($k$) | Comparison | Metric | Mean Diff (pp) | Paired $t$ | Raw $p_{(k)}$ | Multiplier ($42 - k + 1$) | Exact Holm $p_{\text{adj}}$ | Significant at $\alpha=0.05$? |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | Swin-Tiny vs. Standard U-Net | Foreground Recall | $+17.94$ | $+6.218$ | $0.000155$ | $42$ | **$0.006527$** | **YES ($p < 0.01$)** |
| **2** | Swin-Tiny vs. Attention U-Net | Foreground Recall | $+25.19$ | $+5.391$ | $0.000438$ | $41$ | **$0.017975$** | **YES ($p < 0.05$)** |
| **3** | Swin-Tiny vs. Attention U-Net | Foreground IoU | $+12.15$ | $+5.052$ | $0.000688$ | $40$ | **$0.027539$** | **YES ($p < 0.05$)** |
| **4** | Swin-Tiny vs. Attention U-Net | Foreground Dice | $+10.98$ | $+4.661$ | $0.001198$ | $39$ | **$0.046714$** | **YES ($p < 0.05$)** |
| **5** | Swin-Tiny vs. Attention U-Net | Macro mDice | $+6.18$ | $+4.148$ | $0.002497$ | $38$ | **$0.094883$** | **NO (Trend, $p < 0.10$)** |
| **6** | Standard U-Net vs. Swin-Tiny | Foreground Precision | $+9.75$ | $+3.958$ | $0.003315$ | $37$ | **$0.122646$** | **NO (Non-Significant)** |
| **7** | Standard U-Net vs. ResNet-50 U-Net | Foreground Precision | $+7.57$ | $+3.884$ | $0.003716$ | $36$ | **$0.133777$** | **NO (Non-Significant)** |
| **8** | Swin-Tiny vs. Standard U-Net | Foreground IoU | $+6.71$ | $+3.467$ | $0.007077$ | $35$ | **$0.247702$** | **NO (Non-Significant)** |
| **9** | Attention U-Net vs. Swin-Tiny | Foreground Precision | $+10.94$ | $+3.457$ | $0.007197$ | $34$ | **$0.247702$** | **NO (Non-Significant)** |
| **10** | Swin-Tiny vs. Standard U-Net | Foreground Dice | $+5.62$ | $+3.393$ | $0.007954$ | $33$ | **$0.262482$** | **NO (Non-Significant)** |

> [!NOTE] Omnibus Threshold Findings
> Even under the ultra-conservative 42-hypothesis omnibus correction, the zero-shot Recall advantage of Swin-Tiny over Standard U-Net ($p_{\text{adj}} = 0.0065$) and Attention U-Net ($p_{\text{adj}} = 0.0180$), and the FG IoU advantage of Swin-Tiny over Attention U-Net ($p_{\text{adj}} = 0.0275$), remain **statistically significant at $\alpha = 0.05$**.

---

## 9. Comprehensive Multiplicity Summary Matrix

The table below cross-references all key pairwise comparisons across every defined multiplicity family:

| Architecture Comparison | Evaluated Metric | Raw $p$ | Family A ($m=7$, Metric Battery) | Family B/C ($m=6$, Model Contrast) | Family E ($m=42$, Global Omnibus) | Definitive Statistical Finding |
|---|---|:---:|:---:|:---:|:---:|---|
| **Swin-Tiny vs. ResNet-50 U-Net** | Foreground IoU | $0.0396$ | $0.2374$ | $0.1583$ | $1.0000$ | **Non-Significant under all corrected families** (Descriptive $+7.14\text{ pp}$) |
| **Swin-Tiny vs. ResNet-50 U-Net** | Foreground Recall | $0.0094$ | $0.0660$ | $0.0377$* | $0.2922$ | **Directional trend ($d_z = 1.04$)**; Significant under Family C, Trend under Family A |
| **Swin-Tiny vs. Standard U-Net** | Foreground IoU | $0.0071$ | $0.0354$* | $0.0354$* | $0.2477$ | **Statistically Significant ($p < 0.05$)** under Families A and B |
| **Swin-Tiny vs. Standard U-Net** | Foreground Recall | $0.0002$ | $0.0011$** | $0.0009$*** | $0.0065$** | **Statistically Significant ($p < 0.01$)** across ALL families including Omnibus |
| **Swin-Tiny vs. Attention U-Net** | Foreground IoU | $0.0007$ | $0.0041$** | $0.0041$** | $0.0275$* | **Statistically Significant ($p < 0.05$)** across ALL families including Omnibus |
| **Standard U-Net vs. ResNet-50 U-Net** | Foreground Precision | $0.0037$ | $0.0260$* | $0.0199$* | $0.1338$ | **Statistically Significant ($p < 0.05$)** (Scratch U-Net $+7.57\text{ pp}$ higher precision) |
| **Standard U-Net vs. ResNet-50 U-Net** | Foreground IoU | $0.8692$ | $1.0000$ | $1.0000$ | $1.0000$ | **Null hypothesis not rejected** (Indistinguishable zero-shot performance) |
| **Standard U-Net vs. Attention U-Net** | Foreground IoU | $0.1225$ | $0.7168$ | $0.3675$ | $1.0000$ | **Null hypothesis not rejected** (Attention gates provide no benefit) |
| **Swin-MoE vs. Vanilla Swin-Tiny** | Foreground IoU | $0.8218$ | $1.0000$ | $1.0000$ | $1.0000$ | **Null hypothesis not rejected** (Top-2 MoE routing provides no benefit) |
| **ResNet-50 vs. Standard U-Net (BUSI)** | In-Domain FG IoU | $0.1941$ | $0.7766$ | $0.3883$ ($m=3$) | — | **Null hypothesis not rejected** (In-domain test difference is non-significant) |

---
*All calculations derive strictly from the verified canonical dataset [FINAL_VERIFIED_PAIRED_TESTS.csv](file:///D:/Study%20Material/Research/CV/vision_transformer_research/research_audit/final_cross_phase_integrity_audit/FINAL_VERIFIED_PAIRED_TESTS.csv).*
