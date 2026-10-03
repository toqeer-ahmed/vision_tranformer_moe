# The Complete Research Journey: Vision Transformers and Mixture of Experts in Medical Segmentation

## Part A & B: The Real-World Problem
Imagine a doctor examining a breast ultrasound image. The image is noisy and complex, containing normal tissue, shadows, and potentially a lesion (a tumor). 

The doctor doesn't just need to know, "Is there a tumor in this image?" (Classification). 
The doctor needs to know, "**Exactly which pixels belong to the tumor?**" (Segmentation).

Think of segmentation like giving the AI a digital highlighter. We aren't just categorizing the image; we are drawing a precise, pixel-perfect boundary around the lesion. This is critical for medical diagnostics to determine the exact size and shape of a tumor for biopsy or surgery.

To solve this, we use an AI model that takes an Image, processes it, and outputs a Segmentation Mask (a black-and-white image where white pixels represent the tumor).

---

## Part C & D: The Modeling Journey and Vision Transformers

Historically, Convolutional Neural Networks (CNNs) were used for this. CNNs slide small filters over an image to detect edges and textures. However, they struggle to understand the "big picture" (long-range dependencies).

To solve this, researchers introduced **Vision Transformers (ViTs)**.
Unlike CNNs, Transformers don't slide filters. Here is how they work:
1. Take an **Image**.
2. Cut it into small, non-overlapping squares called **patches**.
3. Flatten each patch into a list of numbers called a **token** (an embedding).
4. Add **positional information** so the model knows where the patch came from (top-left, bottom-right).
5. Pass the tokens into a **Transformer block**. 

Inside the block, a mechanism called **Self-Attention** allows every token to "look" at every other token in the image simultaneously. A patch of tissue on the left can directly communicate with a patch on the right. Finally, a Feed-Forward Network (**FFN**) processes this combined information to create a rich **feature map**. 

This global understanding makes Transformers incredibly powerful for segmentation.

---

## Part E: Why SegFormer?

Standard Vision Transformers output tokens at a single, coarse resolution. But tumors come in all sizes. 

We selected **SegFormer** because it uses a hierarchical architecture (the `mit-b0` Mix Vision Transformer backbone). It processes the image at multiple resolutions (from fine details to coarse global features). A lightweight MLP decoder then fuses these multi-scale features together to predict the final high-resolution segmentation mask. 

"B0" refers to the smallest, most lightweight version of this backbone (only ~3.7 million parameters), making it ideal for small-scale medical datasets where massive models would overfit.

---

## Part F: Why Mixture of Experts (MoE)?

Imagine a hospital with four specialists (Experts). Instead of forcing every specialist to examine every single patient, a receptionist (Router) looks at the patient's file and sends them only to the top two most relevant specialists.

In a neural network, **Mixture of Experts (MoE)** works the same way:
- **Experts:** We replace the standard FFN in the Transformer with multiple independent FFNs.
- **Router:** A small neural network looks at an incoming image token and calculates routing probabilities.
- **Top-K Gating:** The router selects the top 2 experts for that specific token.
- **Load Balancing:** An auxiliary loss function forces the router to distribute tokens evenly, preventing it from just sending everything to Expert 1 (which is called "expert collapse").

**Why do this?** 
MoE drastically increases the number of parameters (model capacity) without slowing down the network, because only a fraction of the network is used for any given token.

**Our Implementation:**
We replaced the FFNs in the SegFormer backbone with 4 experts, routed via Top-2 noisy gating, stabilized by a load-balancing loss.

---

## Part G: Why Spatially-Aware MoE?

When predicting exactly where a tumor boundary is, local spatial relationships (neighboring pixels) are critical. 

Standard MoE experts are linear (they look at one token at a time). But the original SegFormer FFN contained a special $3 \times 3$ depthwise convolution—a layer that explicitly mixes information between neighboring tokens to preserve spatial awareness.

If we naively replaced this with linear MoE experts, the model would go "blind" to local boundaries. Therefore, we designed a `SpatiallyAwareExpert`. 
**Our implementation:**
- `dense1` (Linear projection)
- `dwconv` ($3 \times 3$ depthwise convolution to mix spatial neighbors)
- `activation` (GELU)
- `dense2` (Linear projection)

---

## Part H: Why Warm Initialization?

When you initialize a network, its weights are random. But our baseline SegFormer was *pre-trained* on ImageNet; its original FFNs already knew how to recognize shapes and textures.

When we ripped out the original FFN and replaced it with our MoE experts, those new experts were born with "amnesia" (random weights). The model had to relearn how to process images from scratch on a tiny dataset, which caused performance to drop.

**Warm Initialization** solves this. Instead of starting from scratch, we copied the exact pre-trained weights from the original SegFormer FFN and pasted them into all 4 of our MoE experts before training began. The router remained randomly initialized so it could learn how to distribute traffic among the newly identical experts.

---

## Part I & K: The Dataset and Generalization

We used **BUSI** (Breast Ultrasound Images), an open-source dataset. We split it into 80% Training, 10% Validation, and 10% strictly held-out Test data to evaluate how well the model learned.

But testing on the same dataset is like a student studying a teacher's practice exam and then taking that same teacher's final exam. 

To test true **Cross-Dataset Generalization (Zero-Shot)**, we took the models trained *only* on BUSI and tested them on **BUS-BRA**, an entirely different dataset from a different hospital and country. This simulates the real world: a model trained in one clinic must work on ultrasound machines in another clinic without failing (handling Out-of-Distribution/OOD data, or dataset shift).

---

## Part J & L: The SegFormer Story and Key Findings

We ran three stages:
1. **Vanilla SegFormer:** Established the baseline.
2. **Spatially-Aware MoE (Random Init):** Tested if MoE routing improved segmentation. Performance dropped because it lost pre-trained knowledge.
3. **Warm-Init MoE:** Restored the pre-trained knowledge. Performance recovered in-domain, but remained slightly below Vanilla.

**The Crucial Zero-Shot Finding (BUSI → BUS-BRA):**
When tested on the unseen BUS-BRA dataset, we found a massive behavioral shift:

| Model | Precision (False Positive control) | Recall (False Negative control) |
|---|---|---|
| Vanilla SegFormer | **81.88%** | 66.79% |
| Warm-Init MoE | 65.04% | **81.22%** |

*Simple explanation:* The Vanilla model was very conservative (high precision), meaning when it predicted a lesion, it was usually right, but it completely missed a lot of actual lesions (low recall). The MoE model adapted by becoming hyper-sensitive (high recall), finding almost all the lesions, but at the cost of "crying wolf" and predicting lesions where there were none (low precision).

---

## Part M, N, O: The Swin Transformer & Phase 8

The research question became: *"Is this precision-recall trade-off a universal property of MoEs, or is it just because we used SegFormer?"*

To answer this, we introduced the **Swin Transformer**.
Ordinary self-attention calculates relationships between *every* token, which is very slow. Swin groups tokens into **Windows** (like people talking in separate rooms) and performs attention only inside the window. In the next layer, the windows are **Shifted**, allowing people to talk to neighboring rooms. 

We implemented Phase 8 on Kaggle GPUs (using Mixed Precision / AMP to prevent memory crashes and speed up training) and trained:
1. Vanilla Swin-Tiny
2. Spatially-Aware Swin-MoE
3. Warm-Init Swin-MoE

**Phase 8 Zero-Shot Results (BUSI → BUS-BRA):**

| Model | mIoU | mDice | Precision | Recall |
|---|---|---|---|---|
| Vanilla Swin | 0.7072 | 0.8083 | 0.7849 | **0.8378** |
| Swin MoE | 0.6808 | 0.7813 | 0.8530 | 0.7375 |
| Warm Swin MoE | 0.6399 | 0.7393 | **0.8745** | 0.6832 |

---

## Part P: The Surprising Cross-Architecture Result

We discovered a complete reversal in behavior!

- **SegFormer MoE:** Sacrificed Precision (-16%) to boost Recall (+14%). It became hyper-sensitive.
- **Swin MoE:** Sacrificed Recall (-15%) to boost Precision (+9%). It became conservative.

This suggests that MoE routing does not have a single, universal effect on generalization. The way the router adapts to out-of-distribution data is deeply dependent on the underlying spatial inductive biases of the backbone architecture.

---

## Part Q & R: Engineering and Current Research Position

**Engineering Triumphs:**
To achieve this, we solved massive bottlenecks: migrated from slow local CPUs to Kaggle GPUs, implemented Automatic Mixed Precision (AMP) to fix memory crashes, engineered streaming metric accumulators to prevent RAM Out-of-Memory errors during zero-shot evaluation, and dynamically patched Kaggle's read-only filesystem constraints to pair the BUS-BRA datasets correctly.

**Current Research Position:**
- **Strongly Supported:** MoE routing changes prediction behavior under dataset shift, but the direction of this change (Precision vs Recall) is highly dependent on the backbone architecture (SegFormer vs Swin).
- **Not Established:** We cannot claim MoE universally improves generalization or clinical safety, as it worsened overall IoU in these specific small-dataset medical configurations.
