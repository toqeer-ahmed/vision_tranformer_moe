# Phase 9: Routing Analysis Plan

**Primary Research Question:** Why does MoE produce opposite precision-recall behavior in SegFormer and Swin under zero-shot dataset shift?

> **STOP:** Do not initiate any new large-scale training until this analysis is reviewed.

## Step 1: Code Audit (SegFormer MoE vs Swin MoE)

Before running experiments, we must acknowledge a critical architectural difference discovered in the codebase:

| Feature | SegFormer MoE | Swin MoE |
|---|---|---|
| **Original Backbone FFN** | Contains a $3 \times 3$ depthwise convolution | Standard 2-layer linear MLP |
| **MoE Expert Architecture** | `SpatiallyAwareExpert` ($3 \times 3$ dwconv) | `SpatiallyAwareExpert` ($3 \times 3$ dwconv) |
| **Result of MoE Injection** | **Preserves** existing spatial inductive bias | **Injects** a new spatial inductive bias |
| **Warm Initialization** | Fully implemented (clones original dwconv weights) | **Missing/Incomplete** (Original FFN had no dwconv to clone) |
| **Router & Gating** | Top-2 Noisy Gating | Top-2 Noisy Gating |
| **Load Balancing Loss** | Yes (0.01 coef) | Yes (0.01 coef) |
| **Stages Replaced** | All transformer blocks | All transformer blocks |

**Key Finding:** Because we reused the `SpatiallyAwareExpert` for Swin, we accidentally added overlapping receptive fields (depthwise convolutions) to an architecture that originally relied entirely on windowed attention for spatial mixing. Furthermore, because Swin's original FFN lacked these convolutions, true "Warm Initialization" was mathematically impossible for those layers, meaning the Swin Warm-MoE was likely trained from random initialization just like the Spatial MoE.

## Step 2: Routing Behavior Audit

We need to understand how the models are making decisions. Currently, our `zero_shot_metrics.json` logs `mean_routing_entropy` and `expert_utilization` for the entire dataset.

**Required Logging Hooks (To be implemented if missing):**
1. **Per-Image Expert Utilization:** Does the router use different experts for BUSI (in-domain) vs BUS-BRA (OOD)?
2. **Spatial Routing Maps:** A hook to save the `top_k_indices` mapped back to the 2D image grid. We need to visualize *where* the router is sending tokens. Do edge tokens go to Expert 1 while center tokens go to Expert 2?

## Step 3: Qualitative Error Analysis

Before training, we must visually inspect the predictions.
We need to generate side-by-side comparison grids for 10-20 random BUS-BRA images:
1. Original Ultrasound Image
2. Ground Truth Mask
3. Vanilla Prediction (SegFormer / Swin)
4. MoE Prediction (SegFormer / Swin)

We are looking for visual evidence of the precision-recall shift:
- Do SegFormer MoE masks look "bloated" (high recall, low precision)?
- Do Swin MoE masks look "shrunken" or overly fragmented (high precision, low recall)?

## Step 4: Controlled Experiment Design

Once the qualitative and code audits are complete, we propose the following minimal, targeted experiment to explain the architecture-dependent behavior:

### Experiment 9A: The Swin Linear Expert Control
**Hypothesis:** The conservative behavior (high precision/low recall) of Swin MoE is caused by injecting the $3 \times 3$ depthwise convolution into the experts, which interferes with Swin's native windowed attention.
- **Independent Variable:** Expert architecture (`SpatiallyAwareExpert` vs standard `MLPExpert`).
- **Controlled Variables:** Swin-Tiny backbone, Top-2 routing, hyperparameters.
- **Dataset:** BUSI (Train) → BUS-BRA (Zero-Shot).
- **Expected Interpretation:** If replacing the `SpatiallyAwareExpert` with a simple `MLPExpert` (which matches Swin's original FFN) restores the baseline behavior or flips the precision/recall shift, we prove the behavior is caused by the expert architecture, not the router.

### Experiment 9B: Routing Distribution Shift
**Hypothesis:** The MoE routers in SegFormer and Swin react differently to OOD data (BUS-BRA).
- **Method:** Extract and plot the expert utilization distribution for BUSI Test vs BUS-BRA for both models.
- **Expected Interpretation:** If SegFormer's routing entropy collapses on BUS-BRA while Swin's remains stable, it indicates the routing mechanism is the source of the behavioral divergence.
