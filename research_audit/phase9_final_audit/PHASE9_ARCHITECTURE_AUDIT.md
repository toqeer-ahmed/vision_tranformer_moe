# Phase 9 Architecture & Implementation Audit

**Audit Date**: November 12, 2026  
**Auditor**: Senior Computer Vision Research Engineer & Scientific Auditor  
**Repository**: `D:\Study Material\Research\CV\vision_transformer_research`  
**Audited Files**:
- `models/swin.py`
- `models/moe/standard_swin_moe_layer.py`
- `models/moe/experts.py`
- `models/moe/gating.py`
- `models/moe/router.py`
- `count_params_phase9.py`

---

## 1. High-Level Architecture Explanation (Beginner-Friendly)

In a standard Swin Transformer, an image is divided into small patches (tokens). These tokens pass through a series of Transformer blocks:
1. **Shifted Window Self-Attention**: Tokens attend to neighboring tokens inside local windows to capture spatial context.
2. **Feed-Forward Network (FFN)**: A standard two-layer Multi-Layer Perceptron (Linear $\to$ GELU $\to$ Linear) that processes each token independently to refine features.

In a **Mixture of Experts (MoE)** architecture, instead of using a single FFN for every token, we create multiple parallel FFNs ("Experts"). A lightweight "Router" network looks at each token and dynamically routes it to the top 2 most relevant experts.

```
                            [Input Token x]
                                   |
                     +-------------+-------------+
                     |                           |
             [Router Network]            [Token Pathways]
             (Calculates gating                  |
              probabilities)                     |
                     |                           |
             [Top-2 Selection]                   |
             (e.g., Expert 1 & 3)                |
                     |                           |
                     +-------+           +-------+
                             |           |
                        +----+-----------+----+
                        |                     |
                   [Expert 1]            [Expert 3]
                   (Standard             (Standard
                    2-layer MLP)          2-layer MLP)
                        |                     |
                        +----+-----------+----+
                             |
                     [Weighted Sum:
                      g1 * E1(x) + g3 * E3(x)]
                             |
                       [Output Token]
```

---

## 2. Vanilla Swin Architecture

- **Backbone**: `microsoft/swin-tiny-patch4-window7-224` (ImageNet-1k pre-trained).
- **Structure**: 4 hierarchical stages:
  - **Stage 0**: 2 Swin Transformer blocks, hidden dimension $C = 96$, intermediate FFN dimension $4C = 384$.
  - **Stage 1**: 2 Swin Transformer blocks, hidden dimension $C = 192$, intermediate FFN dimension $4C = 768$.
  - **Stage 2**: 6 Swin Transformer blocks, hidden dimension $C = 384$, intermediate FFN dimension $4C = 1536$.
  - **Stage 3**: 2 Swin Transformer blocks, hidden dimension $C = 768$, intermediate FFN dimension $4C = 3072$.
  - **Total Blocks**: $2 + 2 + 6 + 2 = 12$ blocks.
- **Decoder Head**: Simple lightweight convolutional head:
  - Conv2D($768 \to 256, 3 \times 3$, padding 1) $\to$ BatchNorm2D $\to$ ReLU $\to$ Dropout(0.1) $\to$ Conv2D($256 \to 3, 1 \times 1$).
  - Bilinear interpolation upsample to original resolution ($224 \times 224$).
- **Total Parameters**: **29,290,365** (Verified).

---

## 3. Standard-MLP Swin-MoE Architecture

The Standard-MLP MoE modification replaces the native FFN in all 12 Swin blocks:

### A. Expert Implementation (`SimpleMLPBlock`)
- Implemented in `models/moe/experts.py#L28-L44`.
- **Layers**:
  - `dense1`: `nn.Linear(hidden_dim, 4 * hidden_dim)`
  - `act`: `nn.GELU()`
  - `dense2`: `nn.Linear(4 * hidden_dim, hidden_dim)`
  - `dropout`: `nn.Dropout(0.1)`
- **Crucial Architectural Property**: This **strictly preserves** the exact layer sequence, activation, and dimensional expansion ($4\times$) of Swin's native FFN. Unlike Phase 8 (`SpatiallyAwareExpert`), there is **no depthwise convolution**, isolating pure conditional routing from convolutional inductive biases.

### B. Routing Mechanism (`GatingNetwork` & `TopKRouter`)
1. **Gating Network** (`models/moe/gating.py`):
   - Computes linear projections: $H(x) = x \cdot W_g$, where $W_g \in \mathbb{R}^{C \times E}$ ($E=4$ experts).
   - Training-time exploration noise:
     $$\text{Logits}(x) = x \cdot W_g + \epsilon \cdot \text{Softplus}(x \cdot W_{\text{noise}}), \quad \epsilon \sim \mathcal{N}(0, I)$$
2. **Top-K Selection** (`models/moe/router.py`):
   - Softmax over experts: $P = \text{Softmax}(\text{Logits}(x))$.
   - Selects top $k=2$ experts with highest probability.
   - Re-normalizes gate weights:
     $$g_i = \frac{P_i}{\sum_{j \in \text{Top-2}} P_j}$$

### C. FFN Replacement & Warm Initialization (`models/swin.py`)
- Function `replace_swin_ffn_with_standard_moe()`:
  - Replaces `block.intermediate` with `SwinMoEWrapper`.
  - Replaces `block.output` with `nn.Identity()` (since the expert already projects back to $C$).
- **Warm Initialization Protocol** (`models/swin.py#L101-L113`):
  ```python
  if moe_cfg.get("warm_init", False):
      with torch.no_grad():
          for expert in moe_layer.experts:
              expert.dense1.weight.copy_(block.intermediate.dense.weight)
              expert.dense1.bias.copy_(block.intermediate.dense.bias)
              expert.dense2.weight.copy_(block.output.dense.weight)
              expert.dense2.bias.copy_(block.output.dense.bias)
  ```
  Every expert starts training as an exact replica of the pre-trained ImageNet FFN. The router begins randomly initialized, gradually differentiating token traffic without suffering from random-initialization amnesia.

---

## 4. Parameter Accounting Verification

Parameter counts verified via offline model instantiations in `count_params_phase9.py`:

| Architecture | Expert Type | Number of Experts | Total Parameters | Trainable Parameters | Parameter Ratio vs. Vanilla |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Vanilla Swin-Tiny** | Native FFN | 1 | **29,290,365** | 29,290,365 | $1.00\times$ (Baseline) |
| **Spatial Swin-MoE** (Phase 8) | `SpatiallyAwareExpert` ($3\times3$ DWConv) | 4 | **81,855,549** | 81,855,549 | $2.80\times$ |
| **Standard-MLP Swin-MoE** (Phase 9) | `SimpleMLPBlock` (Pure Linear) | 4 | **81,148,989** | 81,148,989 | $2.77\times$ |

### Mathematical Breakdown of Parameter Delta:
$$\Delta_{\text{Spatial} - \text{Standard}} = 81,855,549 - 81,148,989 = 706,560\text{ parameters}$$
Across the 12 replaced blocks:
- Stage 0 ($2 \text{ blocks} \times 4 \text{ experts}$): $8 \times (3 \times 3 \times 384 + 384) = 30,720$
- Stage 1 ($2 \text{ blocks} \times 4 \text{ experts}$): $8 \times (3 \times 3 \times 768 + 768) = 61,440$
- Stage 2 ($6 \text{ blocks} \times 4 \text{ experts}$): $24 \times (3 \times 3 \times 1536 + 1536) = 368,640$
- Stage 3 ($2 \text{ blocks} \times 4 \text{ experts}$): $8 \times (3 \times 3 \times 3072 + 3072) = 245,760$
- **Total Depthwise Conv Parameters**: $30,720 + 61,440 + 368,640 + 245,760 = \mathbf{706,560}$.
The parameter accounting is mathematically exact.

---

## 5. Critical Implementation Concerns

1. **Load Balancing Loss Disconnection**:
   In `StandardSwinMoELayer`, `self.aux_loss` is calculated and saved as a buffer, but neither `SwinSegmentation.forward()` nor `train_swin_standard_moe.py` ever aggregates or backpropagates this term. The router optimization was driven exclusively by the task loss gradient.
2. **Dense Evaluation of Experts**:
   In `models/moe/standard_swin_moe_layer.py#L62`, when any token routes to an expert, the implementation executes `expert(hidden_states)` across the entire batch grid before masking. This is mathematically correct but computationally redundant compared to a true sparse scatter-gather dispatch. For Swin-Tiny batch size 16, GPU memory was sufficient, but it does not represent optimal throughput.
3. **Identity Layer Replacement**:
   Setting `block.output = nn.Identity()` replaces the layer norm and residual connections in Hugging Face's Swin implementation correctly because the MoE wrapper sits inside the intermediate residual stream.

