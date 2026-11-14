# Auxiliary Loss Implementation Audit: Standard-MLP Swin-MoE

**Audit Date**: November 12, 2026  
**Auditor**: Senior Computer Vision Research Engineer & Scientific Auditor  
**Repository**: `D:\Study Material\Research\CV\vision_transformer_research`  
**Target Architecture**: `SwinSegmentation` with `StandardSwinMoELayer`  

---

## 1. Executive Summary

This audit addresses the critical defect identified during the Phase 9 reproducibility audit: the Mixture-of-Experts auxiliary load-balancing loss was configured in YAML (`balance_loss_coef: 0.01`) and calculated inside each layer's `TopKRouter`, but was **completely omitted from the backward pass** in `training/train_swin_standard_moe.py`.

This document details the forensic inspection of the original codebase, answers the five mandatory audit questions, maps the autograd computational graph, and documents the repaired integration.

---

## 2. Answers to the Five Core Architectural Questions

### Question 1: Where does each MoE layer calculate its auxiliary load-balancing loss?
**Answer**:  
Inside `models/moe/router.py` in the `TopKRouter.forward()` method (lines 46–60):
```python
if self.training:
    assignments = torch.zeros(num_tokens, self.num_experts, device=gating_logits.device)
    assignments.scatter_(1, top_k_indices, 1.0)
    f = assignments.mean(dim=0) # shape: (num_experts) - dispatch frequency
    P = gates.mean(dim=0)       # shape: (num_experts) - gating probabilities
    balance_loss = self.num_experts * torch.sum(f * P)
else:
    balance_loss = torch.tensor(0.0, device=gating_logits.device)

return top_k_gates, top_k_indices, balance_loss * self.balance_loss_coef
```
In `models/moe/standard_swin_moe_layer.py#L50-L51`, this return value is captured and stored:
```python
top_k_gates, top_k_indices, aux_loss = self.router(gating_logits)
self.aux_loss = aux_loss
```

### Question 2: Is that loss updated on every forward pass?
**Answer**:  
**YES**. On every forward pass through `StandardSwinMoELayer.forward()`, `self.router(gating_logits)` is executed. This computes a fresh `aux_loss` tensor derived from the current batch's gating logits and overwrites `self.aux_loss`. When `model.train()` is active, `aux_loss` has `requires_grad=True` and is connected to the autograd graph via `gates`. When `model.eval()` is active, `TopKRouter` returns `torch.tensor(0.0)`.

### Question 3: How can the loss be accessed safely by the training script?
**Answer**:  
The model hierarchy is:
`SwinSegmentation` $\to$ `self.backbone (SwinModel)` $\to$ `encoder.layers[stage_idx]` $\to$ `blocks[block_idx]` $\to$ `intermediate (SwinMoEWrapper)` $\to$ `moe_layer (StandardSwinMoELayer)`.

Accessing this directly in training scripts via long nested loops creates fragile, error-prone code. The clean, safe implementation encapsulates this within a dedicated method on `SwinSegmentation` in `models/swin.py`:
```python
def get_aux_loss(self):
    total_aux = torch.tensor(0.0, device=next(self.parameters()).device)
    for stage in self.backbone.encoder.layers:
        for block in stage.blocks:
            if hasattr(block, "intermediate") and hasattr(block.intermediate, "moe_layer"):
                moe_l = block.intermediate.moe_layer
                if hasattr(moe_l, "aux_loss") and moe_l.aux_loss is not None:
                    total_aux = total_aux + moe_l.aux_loss
    return total_aux
```
If called on Vanilla Swin (where `intermediate` is standard `SwinIntermediate`), the `hasattr` check cleanly returns `0.0`.

### Question 4: Do all 12 MoE layers contribute to the training objective?
**Answer**:  
- **In Original Phase 9**: **ZERO (0) layers contributed**. The training loop only computed `loss = criterion(logits, targets)`.
- **In Repaired Implementation**: **ALL TWELVE (12) layers contribute**. In `SwinModel` (`microsoft/swin-tiny-patch4-window7-224`), there are:
  - Stage 0: 2 blocks
  - Stage 1: 2 blocks
  - Stage 2: 6 blocks
  - Stage 3: 2 blocks
  - **Total: 12 blocks**, all of which have their FFN replaced by `StandardSwinMoELayer`. `get_aux_loss()` sums across all 12 layers.

### Question 5: Is the configured `balance_loss_coef` actually applied?
**Answer**:  
**YES, but care was required to avoid double-scaling**.  
In `TopKRouter.forward()#L60`, the return statement is:
```python
return top_k_gates, top_k_indices, balance_loss * self.balance_loss_coef
```
The router **already multiplies** the balance loss by `balance_loss_coef` (0.01). Therefore, in the training script, the loss must be formulated as:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{segmentation}} + \mathcal{L}_{\text{aux}}$$
If the training script had multiplied by `balance_loss_coef` again, the effective coefficient would have been $0.01 \times 0.01 = 0.0001$, rendering the load-balancing penalty 100 times too weak.

---

## 3. Autograd Computational Graph Flow

```
                                [Input Batch Images]
                                         |
                                         v
                            [Swin Transformer Backbone]
                                         |
        +--------------------------------+--------------------------------+
        |                                                                 |
        v                                                                 v
  [Block 0..11 Attention]                                        [Decode Head Conv2D]
        |                                                                 |
        v                                                                 v
  [Tokens x] ---> [GatingNetwork: W_g]                                [Logits]
        |                  |                                              |
        |                  v                                              v
        |           [Routing Logits]                           [Segmentation Loss:
        |                  |                                    Focal Tversky]
        |                  v                                              |
        |           [TopKRouter]                                          |
        |            /         \                                          |
        |           v           v                                         |
        |    [Top-2 Gates]   [Aux Loss: 0.01 * N * sum(f*P)]              |
        |          |                     |                                |
        v          v                     |                                |
   [Selected Experts: E_i(x)]            |                                |
        |          |                     |                                |
        +----+-----+                     |                                |
             |                           |                                |
             v                           v                                |
     [Token Output]               [get_aux_loss()]                        |
             |                           |                                |
             |                           v                                v
             +--------------------> [Total Loss = Seg Loss + Agg Aux Loss]
                                                 |
                                                 v
                                       [total_loss.backward()]
                                                 |
                         +-----------------------+-----------------------+
                         |                                               |
                         v                                               v
              [Task Gradients to Experts]                   [Balancing Gradients to
               and Attention Layers]                         Router Weights W_g]
```

---

## 4. Verification Checkpoint

The repaired autograd flow guarantees:
1. Every MoE block updates its router weights based on both expert task specialization and load-balancing penalty.
2. The auxiliary loss is evaluated in real time for every batch without retaining stale graphs.
3. Vanilla Swin is unaffected and produces strictly zero auxiliary loss.

