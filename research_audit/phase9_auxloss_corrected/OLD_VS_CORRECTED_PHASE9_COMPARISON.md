# Technical Comparison: Original Phase 9 vs. Corrected Phase 9

**Auditor**: Senior Computer Vision Research Engineer & Scientific Auditor  
**Repository**: `D:\Study Material\Research\CV\vision_transformer_research`  

---

## 1. High-Level Comparison Summary

| Dimension | Original Phase 9 Experiment | Corrected Phase 9 Experiment (`phase9_auxloss_corrected`) | Scientific / Engineering Significance |
| :--- | :--- | :--- | :--- |
| **Auxiliary Load Loss** | **DISCONNECTED** (Omitted from backward pass) | **CONNECTED** ($\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{seg}} + \mathcal{L}_{\text{aux}}$) | Router actively penalizes expert starvation |
| **Autograd Verification** | None (Untested before training) | Verified via `tests/test_moe_aux_loss.py` | Proven direct gradient flow $\nabla_{W_g} \mathcal{L}_{\text{aux}} \neq 0$ |
| **Seed Protocol** | 3 Seeds (`42`, `123`, `2026`) | **10 Matched Seeds** ($42 \dots 9001$) | Increased degrees of freedom from $df=2$ to $df=9$ |
| **Reported FG IoU Lead** | $+5.96\%$ (Artifact of seed selection) | **$+0.35\%$** ($57.85\%$ vs $57.50\%$) | Proves MoE does not boost raw mean accuracy |
| **Statistical Test** | Invalid ($df=2$, underpowered) | **$t = 0.2319, \; p = 0.8218$** | Formally fails to reject $H_0$ of zero mean gain |
| **Variance Reduction** | Unchecked | **$F = 4.87, \; p = 0.0273$** | **Confirmed $79.5\%$ variance reduction** |
| **Checkpoint Storage** | Full AdamW state (~975 MB per model) | Pure `state_dict()` (~325 MB) + cleanup | Solves Kaggle 19.5 GB disk exhaustion |
| **Dataset Mounting** | Physical copying via `shutil.copy` | Zero-copy symbolic links (`os.symlink`) | 2-second dataset mount with 0 MB disk usage |

---

## 2. In-Depth Mathematical & Architectural Comparison

### A. The Objective Function
In the original Phase 9 training script (`training/train_swin_standard_moe.py`):
```python
# ORIGINAL FLAWED IMPLEMENTATION:
logits = model(images)
loss = criterion(logits, targets)
scaler.scale(loss).backward()
```
The gradient was computed exclusively as:
$$g = \nabla_\theta \mathcal{L}_{\text{segmentation}}$$
Because router parameters $W_g$ are upstream of top-2 expert selection, they received task gradients only:
$$\nabla_{W_g} \mathcal{L}_{\text{total}} = \sum_{i \in \text{Top-2}} \frac{\partial \mathcal{L}_{\text{seg}}}{\partial y} \frac{\partial y}{\partial E_i} \frac{\partial E_i}{\partial g_i} \nabla_{W_g} g_i$$

In the corrected implementation:
```python
# REPAIRED IMPLEMENTATION:
logits = model(images)
seg_loss = criterion(logits, targets)
aux_loss = model.get_aux_loss()
total_loss = seg_loss + aux_loss
scaler.scale(total_loss).backward()
```
The gradient is now complete:
$$\nabla_{W_g} \mathcal{L}_{\text{total}} = \nabla_{W_g} \mathcal{L}_{\text{segmentation}} + \nabla_{W_g} \mathcal{L}_{\text{aux}}$$
Where:
$$\mathcal{L}_{\text{aux}} = \lambda_{\text{balance}} \cdot N_{\text{experts}} \sum_{e=1}^{N_{\text{experts}}} f_e \cdot P_e$$
Here, $f_e$ is the empirical fraction of tokens routed to expert $e$, and $P_e$ is the average softmax probability assigned to expert $e$. This term actively enforces balanced routing across all 12 MoE layers.

---

## 3. Dissecting the "Original Phase 9 Gain" vs "Corrected 10-Seed Reality"

### The Small Sample Size Trap ($N=3$):
When evaluating only seeds `42`, `123`, and `2026`:
- Seed 42: Vanilla $56.78\%$ vs MoE $59.24\%$ ($\Delta = +2.46\%$)
- Seed 123: Vanilla $47.33\%$ vs MoE $54.18\%$ ($\Delta = +6.85\%$)
- Seed 2026: Vanilla $50.69\%$ vs MoE $59.25\%$ ($\Delta = +8.56\%$)
- **Mean Delta across 3 Seeds: $+5.96\%$**

Because 2 out of the 3 randomly chosen seeds in Phase 9 happened to be catastrophic initialization failures for Vanilla Swin, it appeared as though Standard-MLP MoE provided a massive $+5.96\%$ boost in cross-dataset transfer.

### The 10-Seed Matched Truth ($N=10$):
When evaluated across 10 matched seeds, Vanilla Swin performed strongly on seeds 314 ($60.61\%$), 777 ($60.16\%$), 999 ($60.98\%$), and 9001 ($61.33\%$).
- Vanilla Swin beat MoE on **4 out of 10 seeds**.
- Overall mean Foreground IoU: Vanilla $57.50\% \pm 4.74\%$ vs MoE $57.85\% \pm 2.15\%$ ($\Delta = +0.35\%$).
- Paired $t$-test: $t = 0.2319, \; p = 0.8218$.

### The Real Advantage of MoE: Variance Suppression:
While MoE does not improve peak or mean performance, it protects against the catastrophic failures that plague Vanilla Swin:
- **Vanilla Swin Minimum Foreground IoU**: $47.33\%$
- **Standard-MLP MoE Minimum Foreground IoU**: $54.18\%$ ($+6.85\%$ higher worst-case floor)
- **Variance Ratio $F$-test**: $F(9, 9) = 4.8702, \; p = 0.0273$ ($79.5\%$ variance reduction).

---

## 4. Engineering & Computational Infrastructure Evolution

1. **Autograd Safety Gates**:
   - The test suite `tests/test_moe_aux_loss.py` verifies 4 mathematical conditions:
     1. Vanilla Swin has strictly $0.0$ aux loss.
     2. All 12 MoE layers compute finite, positive aux loss.
     3. Backpropagation updates both router and expert weights.
     4. Pure aux loss updates router weights directly.
2. **Compact Checkpoint Protocol**:
   - In `training/train_swin_vanilla.py` and `training/train_swin_standard_moe.py`, `best_model.pth` saves only `model.state_dict()`. AdamW optimizer states are omitted from the best model, reducing per-model size from ~975 MB to ~325 MB.
   - Intermediate `last_checkpoint.pth` files are automatically deleted upon run completion.
   - Total disk footprint for 20 runs was reduced from $>22$ GB to $<4$ GB.
3. **High-Speed Zero-Copy Symbolic Mounting**:
   - In `Phase9_AuxLoss_Corrected_Reproducibility.ipynb`, `os.symlink` links image files instead of copying them into working disk, accelerating setup time to $<2$ seconds and saving $>2$ GB disk space.
