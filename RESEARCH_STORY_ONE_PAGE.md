# The One-Page Research Story

### Problem
In medical diagnostics, doctors need to know the exact boundaries of a tumor, not just its presence. This requires semantic segmentation (pixel-perfect classification). Small medical datasets make training large AI models difficult because they tend to overfit and fail on unseen data from different clinics.

### Baseline
We started with **SegFormer**, a highly efficient Vision Transformer. Its hierarchical structure and pre-trained weights made it a strong baseline for breast ultrasound (BUSI) segmentation. 

### MoE (Mixture of Experts)
To increase the model's capacity without increasing computational cost, we replaced the standard Feed-Forward Networks with a **Mixture of Experts**. A router network dynamically assigns image tokens to specialized "experts," meaning only a fraction of the network is active at a given time.

### Spatial Awareness
Standard MoE experts are linear and lose local boundary information. Because precise boundaries are critical in medical segmentation, we designed a `SpatiallyAwareExpert` containing a $3 \times 3$ depthwise convolution to preserve local spatial relationships.

### Warm Initialization
When we replaced the pre-trained FFNs with randomly initialized experts, the model suffered "amnesia" and performance dropped. We introduced **Warm Initialization**, copying the original pre-trained weights into the experts before training. This successfully recovered the lost in-domain performance.

### Generalization
Testing on the same dataset (BUSI) doesn't prove clinical reliability. We performed **Zero-Shot Evaluation** on **BUS-BRA**, an entirely unseen dataset from a different country. This "dataset shift" revealed that the SegFormer MoE model traded precision for recall—it became hyper-sensitive, finding more tumors but generating more false positives compared to the Vanilla baseline.

### Swin
To determine if this precision-recall shift was a universal property of MoEs, we migrated the pipeline to Kaggle GPUs and implemented a completely different architecture: the **Swin Transformer**, which uses shifted window attention instead of SegFormer's dense global attention.

### Main Finding
We discovered a surprising reversal. While the SegFormer MoE sacrificed precision to boost recall, the **Swin MoE sacrificed recall to boost precision**. The MoE router's adaptation to unseen medical data is highly dependent on the spatial inductive biases of the underlying backbone architecture.

### Current Question
Why does the MoE router behave conservatively in Swin but aggressively in SegFormer when faced with out-of-distribution data? 

---

### Supervisor Explanation (60–90 Seconds)
"Hi, I'd like to update you on our research. We are trying to improve breast ultrasound segmentation using Mixture of Experts (MoE) to increase model capacity without blowing up computation. 

We initially used SegFormer. Because segmentation requires tight boundaries, we designed 'Spatially-Aware Experts' using depthwise convolutions, and used 'Warm Initialization' from pre-trained weights so the experts wouldn't start from scratch.

When we evaluated the model zero-shot on an entirely new dataset (BUS-BRA), we found something fascinating. The MoE model drastically increased its recall, catching far more tumors than the baseline, but it dropped in precision, creating more false positives. 

We wanted to know if this was a universal MoE property, so we rebuilt the entire pipeline using a different architecture—the Swin Transformer. To our surprise, the behavior completely reversed! Swin MoE dropped in recall but spiked in precision, becoming highly conservative. 

This is our main finding: MoE routing behavior under dataset shift is fundamentally architecture-dependent. Our next phase is to crack open the routing mechanisms of both models to prove exactly why they diverge."

---

### Research Summary Checklist
- **Completed:** Baseline setups, MoE injection, Spatial Experts, Warm Initialization, Zero-Shot pipeline, Swin Transformer integration, Kaggle GPU migration.
- **Learned:** MoE does not universally improve in-domain metrics on small datasets. It alters prediction behavior under domain shift, but the direction (Precision vs Recall) is architecture-dependent.
- **Uncertain:** The exact mathematical reason the routers diverge in behavior, and whether Swin MoE experts were properly warm-initialized given Swin's lack of native spatial convolutions.
- **Why Swin:** To test architecture generalization (whether the SegFormer findings were universal).
- **The Comparison:** SegFormer MoE becomes aggressive (high recall); Swin MoE becomes conservative (high precision).
- **Phase 9 Goal:** Audit the routing statistics and expert architectures to explain the divergence.
- **NOT to do yet:** Do not start large-scale training until the code audit and qualitative visualization of the routing distribution are completed.
