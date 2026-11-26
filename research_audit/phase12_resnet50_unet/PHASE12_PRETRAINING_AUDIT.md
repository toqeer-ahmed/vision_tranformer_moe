# Phase 12 Pretraining Audit and Weight Provenance

## 1. Executive Summary

A critical scientific vulnerability identified in prior literature is the **pretraining confound**: comparing a pretrained transformer against a randomly initialized CNN and attributing the transformer's superior out-of-distribution performance solely to its architectural inductive biases.

To address this confound rigorously, **Phase 12** introduces an **ImageNet-1K pretrained ResNet-50 backbone**. This document audits the exact weight provenance, initialization protocol, gradient flow controls, and multi-seed randomization mechanics.

---

## 2. Pretrained Weight Provenance and Verification

The encoder weights are sourced directly from PyTorch's official TorchVision weight repository:

- **Architecture**: ResNet-50 (He et al., 2016)
- **Weight Specification**: `torchvision.models.ResNet50_Weights.IMAGENET1K_V1`
- **Official URL**: `https://download.pytorch.org/models/resnet50-0676ba61.pth`
- **Cached Local Path**: `C:\Users\Toqeer Ahmed\.cache\torch\hub\checkpoints\resnet50-0676ba61.pth`
- **File Size**: 102,530,333 bytes (~97.78 MB)
- **SHA-256 Hash**: `0676ba610034604e578c2e6840d7c71e2e9c20a44cb25ad7a6fa4d528cb6ae4e`
- **Verification Status**: Validated in `tests/test_resnet50_unet.py::test_01_pretrained_weights_loading_and_verification` against official torchvision tensor values with absolute tolerance $10^{-7}$.

### Strict Initialization Safeguards
The initialization code in `models/resnet50_unet.py` explicitly forbids silent fallback:
```python
if pretrained:
    try:
        if weight_enum is not None:
            resnet = models.resnet50(weights=weight_enum)
        else:
            resnet = models.resnet50(weights=None)
            state = torch.load(weights, map_location="cpu")
            resnet.load_state_dict(state)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load ImageNet-1K pretrained weights for ResNet-50. "
            f"Pretraining is strictly required by the research protocol. Error: {e}"
        )
```
If weights cannot be located or downloaded, an explicit `RuntimeError` terminates execution immediately.

---

## 3. Fine-Tuning Strategy: Full End-to-End Optimization

To match the exact fine-tuning conditions under which Swin-Tiny was audited in Phase 9B:
1. **Unfrozen Backbone**: All 23,508,032 encoder parameters have `requires_grad = True` by default (`freeze_encoder = False`).
2. **Identical Learning Rate**: The entire model (encoder, decoder, and segmentation head) is updated using AdamW with $lr = 1\times 10^{-4}$ and weight decay $wd = 0.05$.
3. **BatchNorm Dynamics**: Batch normalization layers in the ResNet-50 backbone are fine-tuned dynamically (`freeze_bn = False`), allowing the running mean and variance to adapt to the acoustic speckle and intensity distributions of breast ultrasound.

---

## 4. Multi-Seed Randomization Controls

Phase 12 executes across **10 matched seeds**:
`[42, 123, 314, 777, 999, 1001, 2026, 2027, 2718, 9001]`

### Separation of Determinism and Randomness:
1. **Deterministic Backbone Initialization**: Across all 10 seeds, the ResNet-50 encoder is initialized from the exact same ImageNet-1K weights (`resnet50-0676ba61.pth`). This matches the Swin-Tiny baseline, where the backbone weights were identical across all seeds prior to fine-tuning.
2. **Seed-Dependent Decoder Initialization**: The 5 decoder blocks and $1\times 1$ segmentation head are initialized randomly using PyTorch's default Kaiming uniform initialization (`torch.manual_seed(seed)`).
3. **Seed-Dependent Data Partitioning**: The 80/10/10 split of the BUSI dataset is partitioned deterministically per seed, matching the exact image subsets used for Phase 9B (Swin-Tiny and Swin-MoE), Phase 10 (Standard U-Net), and Phase 11 (Attention U-Net).
4. **Seed-Dependent Augmentations**: Online data augmentations (horizontal flips, affine rotations, elastic deformations) use the seed-specific pseudo-random number generator stream.
