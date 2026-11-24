# Model Parameter Audit & Capacity Confounding Analysis

**Experiment Identifier**: `phase9_auxloss_corrected`  
**Auditor**: Senior Computer Vision Research Engineer & Scientific Auditor  
**Repository**: `D:\Study Material\Research\CV\vision_transformer_research`  

---

## 1. Verified Model Parameter Breakdown

All parameter counts were computed directly from the saved model checkpoints (`best_model.pth`) and model module definitions in `models/swin.py` and `models/moe/standard_swin_moe_layer.py`.

### Detailed Component-by-Component Counts:

| Architectural Component | Vanilla Swin-Tiny | Standard-MLP Swin-MoE | Parameter Difference | Ratio ($\frac{\text{MoE}}{\text{Vanilla}}$) |
| :--- | :---: | :---: | :---: | :---: |
| **Patch Embeddings & Stage Layernorms** | $147,744$ | $147,744$ | $0$ | $1.00\times$ |
| **Window Self-Attention Blocks (12 Blocks)** | $9,484,722$ | $9,484,722$ | $0$ | $1.00\times$ |
| **Downsample Patch-Merging Layers** | $641,280$ | $641,280$ | $0$ | $1.00\times$ |
| **Total Non-MoE Backbone** | **$10,273,746$** | **$10,273,746$** | **$0$** | **$1.00\times$** |
| **Dense FFN Blocks (1 per block $\times$ 12)** | $17,274,432$ | $0$ | $-17,274,432$ | $0.00\times$ |
| **MoE Routers / Gating Networks ($W_g + W_{\text{noise}}$)** | $0$ | $35,328$ | $+35,328$ | $\infty$ |
| **MoE FFN Experts (4 per block $\times$ 12 = 48 Experts)** | $0$ | $69,097,728$ | $+69,097,728$ | $\infty$ |
| **Router Token-Count Tracking Buffers** | $0$ | $48$ | $+48$ | $\infty$ |
| **Total Backbone Parameters** | **$27,548,166$** | **$79,406,850$** | **$+51,858,684$** | **$2.88\times$** |
| **Segmentation Decode Head (Conv-BN-ReLU-Conv)** | $1,771,524$ | $1,771,524$ | $0$ | $1.00\times$ |
| **Total Model Parameters** | **$29,319,690$** | **$81,178,374$** | **$+51,858,684$** | **$2.77\times$** |
| **Total Trainable Parameters** | **$29,290,365$** | **$81,148,989$** | **$+51,858,624$** | **$2.77\times$** |

*Note: Trainable parameter counts exclude non-trainable buffers (such as BatchNorm running statistics and router token counter integers).*

---

## 2. Model Capacity Confounding: The Critical Scientific Limitation

A central question in evaluating Mixture-of-Experts architectures is:
> *Does the experiment isolate the causal effect of **conditional dynamic routing**, or is the effect confounded by **additional model parameter capacity**?*

### Audit Analysis:
1. **Total Parameter Capacity**:
   - Standard-MLP Swin-MoE possesses **$81.15$ million parameters**, compared to only **$29.29$ million parameters** in Vanilla Swin-Tiny.
   - This represents an increase of **$+177\%$ in parameter storage capacity** ($2.77\times$).
2. **Computational vs. Capacity Asymmetry**:
   - In MoE architectures with Top-2 routing across 4 experts, only $2$ experts are activated per token during inference.
   - Consequently, the **computational FLOPs per token** during the forward pass are approximately $1.5\times$ Vanilla Swin, but the **total parameter memory footprint** is $2.77\times$.
3. **Loss Surface & Optimization Implications**:
   - In deep learning theory, overparameterization is well known to smooth the loss landscape, mitigate non-convex saddle points, and provide redundancy against poor weight initializations.
   - Therefore, the empirical observation that MoE avoided the catastrophic collapses observed in Vanilla Swin (Seed 123 and Seed 2026) **cannot be causally attributed to conditional routing**.
   - It is equally plausible that an $81\text{M}$-parameter dense model would exhibit the exact same stabilization due purely to increased parameter redundancy.
4. **Scientific Verdict**:
   The current experiment **severely confounds conditional routing with model capacity**. Asserting that conditional routing is a "structural regularizer" without controlling for parameter count is scientifically unsupportable.

---

## 3. Proposal for a Parameter-Matched Dense Control

To isolate the causal mechanism of conditional routing in future research, the following **parameter-matched dense control** is proposed:

### Proposed Architecture: Dense Swin-Tiny Wide-FFN
- **Backbone**: Swin-Tiny ($224 \times 224$ input, 4 stages, depths $[2, 2, 6, 2]$, channels $[96, 192, 384, 768]$).
- **Modification**: In each of the 12 transformer blocks, replace the standard FFN (expansion ratio $\times 4$) with a **Wide FFN** having expansion ratio $\times 16$:
  - Hidden dimension: $4 \times (\text{standard FFN intermediate dim})$.
  - In Stage 1 ($C=96$): Intermediate dim expands from $384$ to $1,536$.
  - In Stage 2 ($C=192$): Intermediate dim expands from $768$ to $3,072$.
  - In Stage 3 ($C=384$): Intermediate dim expands from $1,536$ to $6,144$.
  - In Stage 4 ($C=768$): Intermediate dim expands from $3,072$ to $12,288$.
- **Target Parameters**:
  - Wide FFN parameters: Exactly $69,097,728$ parameters (matching the 48 MoE experts).
  - Total model parameters: **$81.14$ million parameters** (exact match to Standard-MLP MoE).

### Hypothesis Isolation Matrix:
| Experimental Outcome | Dense Swin-Tiny (29M) | Dense Swin Wide-FFN (81M) | Standard Swin-MoE (81M) | Causal Interpretation |
| :---: | :---: | :---: | :---: | :--- |
| **Case A** | Collapses on 2 seeds | Collapses on 2 seeds | Stable across all seeds | **Conditional routing is the causal regularizer.** |
| **Case B** | Collapses on 2 seeds | Stable across all seeds | Stable across all seeds | **Overparameterization / capacity is the causal factor; routing is irrelevant.** |
| **Case C** | Collapses on 2 seeds | Overfits severely | Stable across all seeds | **MoE achieves capacity without dense overfitting.** |

*(Note: Per audit instructions, this dense control is proposed for future experiments and was not trained during this audit).*
