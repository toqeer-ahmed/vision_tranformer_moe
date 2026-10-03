# My Research Learning Guide

Imagine you are a doctor looking at a grainy, black-and-white ultrasound screen. You suspect a patient has breast cancer. Finding the tumor is like finding a specific shadow in a dark room. 

Our AI project acts like a brilliant assistant that takes the doctor's screen and digitally draws a bright line exactly around the tumor. 

Here is the step-by-step breakdown of how our project actually works, from basic concepts to advanced architecture.

---

### 1. Image
- **In one sentence:** A grid of colored dots representing a picture.
- **Imagine:** A mosaic made of tiny square tiles.
- **Technically:** A matrix of numerical values representing pixel intensities.
- **In OUR project:** The raw breast ultrasound scans from the BUSI dataset.

### 2. Pixel
- **In one sentence:** The smallest individual dot in an image.
- **Imagine:** A single tile in the mosaic.
- **Technically:** A discrete value (0-255) representing brightness.
- **In OUR project:** What the model looks at to determine if it is part of a lesion or normal tissue.

### 3. Segmentation
- **In one sentence:** Drawing a precise boundary around an object of interest.
- **Imagine:** Using a highlighter to trace the exact outline of a word on a page, instead of just saying "the word is on this page" (which is Classification).
- **Technically:** Assigning a class label to every single pixel in the image.
- **In OUR project:** Identifying exactly which pixels are a tumor.

### 4. Mask
- **In one sentence:** The final output picture showing only the highlighted object.
- **Imagine:** A stencil where the tumor is cut out, allowing you to see exactly its shape.
- **Technically:** A binary matrix where `1` is foreground (tumor) and `0` is background.
- **In OUR project:** What the AI generates and compares against the doctor's "Ground Truth" mask.

### 5. CNN (Convolutional Neural Network)
- **In one sentence:** An older type of AI that scans an image using a small sliding window to find patterns.
- **Imagine:** Looking at a massive painting through a tiny magnifying glass, moving it inch by inch.
- **Technically:** A network that applies spatial convolution filters to extract local features.
- **In OUR project:** What researchers historically used, but which we replaced with Transformers to better understand the "big picture".

### 6. Transformer
- **In one sentence:** A modern AI architecture that looks at the entire image at once to understand context.
- **Imagine:** Looking at the entire painting at once, noticing how a shadow in the corner relates to a light source in the center.
- **Technically:** A neural network relying entirely on self-attention mechanisms instead of convolutions.
- **In OUR project:** The core brain of our models (SegFormer and Swin).

### 7. Patch
- **In one sentence:** A small, square chunk cut out of the original image.
- **Imagine:** Taking scissors and cutting a photograph into 16 equal square pieces.
- **Technically:** A flattened 2D block of pixels (e.g., 4x4 or 16x16) used as the basic input unit.
- **In OUR project:** How the ultrasound image is initially fed into the Vision Transformer.

### 8. Token
- **In one sentence:** The mathematical translation of a patch that the AI can read.
- **Imagine:** Translating the visual content of a photograph piece into a descriptive sentence written in numbers.
- **Technically:** A continuous vector embedding representing the semantic features of a patch.
- **In OUR project:** The flow of information passing through the layers of the model.

### 9. Attention
- **In one sentence:** The mechanism that lets the AI decide which parts of the image are most important to each other.
- **Imagine:** If you see a dog's tail on the left side of a photo, your brain "pays attention" to the right side to find the body.
- **Technically:** A weighted sum of values based on the similarity between query and key vectors.
- **In OUR project:** How the model connects different parts of the ultrasound scan to understand the shape of a tumor.

### 10. SegFormer
- **In one sentence:** A specific Transformer designed to process images at multiple scales to output a high-quality segmentation mask.
- **Imagine:** An artist who sketches the rough outline first, then zooms in to paint the fine details, and finally blends them together.
- **Technically:** A hierarchical Vision Transformer with a lightweight multi-layer perceptron (MLP) decoder.
- **In OUR project:** Our first baseline model (the `mit-b0` backbone).

### 11. FFN (Feed-Forward Network)
- **In one sentence:** The part of the model that processes the information gathered by Attention.
- **Imagine:** Attention is a detective gathering clues; the FFN is the detective sitting at a desk processing those clues into a conclusion.
- **Technically:** A two-layer MLP applied independently to each token.
- **In OUR project:** The specific blocks in the Transformer that we ripped out to replace with our Mixture of Experts.

### 12. MoE (Mixture of Experts)
- **In one sentence:** Replacing a single general-purpose brain with several highly specialized mini-brains.
- **Imagine:** A hospital with different specialists instead of one general practitioner.
- **Technically:** A conditional computation layer containing multiple sub-networks (experts) and a gating mechanism.
- **In OUR project:** The core experimental modification we made to the Transformers to increase capacity.

### 13. Expert
- **In one sentence:** One of the specialized mini-brains inside the MoE layer.
- **Imagine:** The radiologist who only specializes in looking at highly dense breast tissue.
- **Technically:** An independent neural network branch (usually an FFN) that processes a subset of tokens.
- **In OUR project:** We used 4 experts in our MoE layers.

### 14. Router
- **In one sentence:** The receptionist that decides which expert should look at which part of the image.
- **Imagine:** A triage nurse assigning patients to the right doctors based on their symptoms.
- **Technically:** A linear layer that predicts a probability distribution over the experts.
- **In OUR project:** The mechanism causing the precision-recall behavioral shifts on unseen data.

### 15. Top-K
- **In one sentence:** The rule dictating exactly how many experts get to see a single piece of information.
- **Imagine:** A hospital rule stating every patient must be seen by exactly 2 doctors.
- **Technically:** Setting the gating probabilities of all but the highest `K` values to negative infinity before routing.
- **In OUR project:** We used Top-K = 2 routing.

### 16. Spatially-Aware Expert
- **In one sentence:** An expert upgraded to understand local boundaries and textures by looking at neighboring pixels.
- **Imagine:** Giving a doctor a magnifying glass to look at the edges of a tumor, rather than just looking at a single dot.
- **Technically:** An expert that includes a $3 \times 3$ depthwise convolution.
- **In OUR project:** We designed `SpatiallyAwareExpert` to prevent the MoE from going "blind" to spatial structures when doing segmentation.

### 17. Warm Initialization
- **In one sentence:** Giving the new experts the pre-existing knowledge of the original model so they don't start from scratch.
- **Imagine:** Handing a new doctor the old doctor's notebook of patient histories on their first day.
- **Technically:** Copying pre-trained weights from a dense FFN into the parallel experts before training begins.
- **In OUR project:** Phase 3, where we proved it recovered performance lost during random initialization.

### 18. Dataset Shift
- **In one sentence:** When the AI encounters data in the real world that looks fundamentally different from what it studied.
- **Imagine:** A student who studied for a math test suddenly being given a physics test.
- **Technically:** A difference between the training data distribution and the test data distribution.
- **In OUR project:** The difference between Egyptian ultrasound machines (BUSI) and Brazilian ultrasound machines (BUS-BRA).

### 19. Zero-Shot
- **In one sentence:** Testing the AI on a completely new dataset without giving it any practice examples.
- **Imagine:** Asking a Spanish speaker to translate Italian without ever teaching them Italian.
- **Technically:** Evaluating a model on an Out-of-Distribution dataset without any fine-tuning.
- **In OUR project:** Training purely on BUSI and evaluating purely on BUS-BRA.

### 20. Precision
- **In one sentence:** When the AI says "this is a tumor", how often is it actually right?
- **Imagine:** A cautious doctor who only diagnoses cancer when they are 100% sure. (High Precision).
- **Technically:** True Positives / (True Positives + False Positives).
- **In OUR project:** Swin-MoE became highly precise, minimizing false alarms.

### 21. Recall
- **In one sentence:** Out of all the actual tumors that exist, how many did the AI successfully find?
- **Imagine:** A paranoid doctor who flags everything suspicious to ensure they never miss a cancer case. (High Recall).
- **Technically:** True Positives / (True Positives + False Negatives).
- **In OUR project:** SegFormer-MoE became highly sensitive, maximizing recall but triggering false alarms.

### 22. IoU (Intersection over Union)
- **In one sentence:** A strict grading system for how perfectly the AI's mask overlaps with the doctor's mask.
- **Imagine:** Placing the AI's stencil over the doctor's stencil and seeing exactly how much they overlap vs how much they don't.
- **Technically:** The area of overlap divided by the total area covered by both masks.
- **In OUR project:** The primary metric we used to evaluate overall performance.

### 23. Dice (Dice Coefficient)
- **In one sentence:** A slightly more forgiving version of IoU that gives more credit for overlap.
- **Imagine:** A grading rubric that is a bit more generous to the student.
- **Technically:** $2 \times (\text{Intersection}) / (\text{Area 1} + \text{Area 2})$.
- **In OUR project:** A secondary metric commonly used in medical literature.

### 24. Swin Transformer
- **In one sentence:** An upgraded Vision Transformer that processes the image in local blocks to save memory and capture finer details.
- **Imagine:** Instead of looking at the whole painting at once, scanning it block-by-block.
- **Technically:** A Hierarchical Vision Transformer using Shifted Windows to limit self-attention computation.
- **In OUR project:** The Phase 8 architecture we used to verify our SegFormer findings.

### 25. Window Attention
- **In one sentence:** Forcing the AI to only pay attention to things inside a small local square.
- **Imagine:** People in a building who are only allowed to talk to people inside their own room.
- **Technically:** Self-attention computed locally within non-overlapping $M \times M$ windows.
- **In OUR project:** The core mechanism making Swin highly efficient for dense prediction tasks.

### 26. Shifted Window
- **In one sentence:** Shifting the boundaries of the local squares so information can cross over.
- **Imagine:** After everyone talks in their rooms, the walls shift so people can now talk to their neighbors from the adjacent room.
- **Technically:** Shifting the window partition by $\lfloor M/2 \rfloor$ pixels to introduce cross-window connections in consecutive layers.
- **In OUR project:** How Swin maintains global understanding despite using local windows.

### 27. Architecture Generalization
- **In one sentence:** Testing if a cool trick works on all cars, or just on Hondas.
- **Imagine:** Finding a great new engine part and seeing if it fits in both a sedan and a truck.
- **Technically:** Determining if a specific mechanism (like MoE) produces consistent behaviors across fundamentally different base architectures.
- **In OUR project:** The reason we tested both SegFormer and Swin.

### 28. Routing Behavior
- **In one sentence:** How the receptionist adapts to seeing patients they have never seen before.
- **Imagine:** Does the triage nurse panic and send everyone to the same doctor, or distribute them evenly?
- **Technically:** The probability distribution output by the gating network when encountering Out-of-Distribution tokens.
- **In OUR project:** What we need to study in Phase 9 to understand the differences between Swin and SegFormer!
