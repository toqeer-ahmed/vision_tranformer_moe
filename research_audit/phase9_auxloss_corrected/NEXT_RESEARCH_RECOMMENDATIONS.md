# Next Research Recommendations & Execution Roadmap

**Auditor & Research Architect**: Senior Computer Vision Research Engineer  
**Repository**: `D:\Study Material\Research\CV\vision_transformer_research`  
**Milestone**: Transition from Phase 9 Audit to Controlled Phase 9B Execution  

---

## 1. Immediate Next Step: Phase 9B Cloud Execution

All local preparation, architectural repair, unit testing, configuration generation, and notebook construction are complete. The immediate physical next step is **cloud GPU execution**:

### Execution Protocol:
1. **Commit and Push Codebase**:
   - Push all changes (including `models/swin.py`, `training/train_swin_standard_moe.py`, `tests/test_moe_aux_loss.py`, `configs/phase9_auxloss_corrected/`, and `Phase9_AuxLoss_Corrected_Reproducibility.ipynb`) to the GitHub remote repository.
2. **Launch Kaggle Session**:
   - Open a Kaggle notebook with Dual T4 or P100 GPU accelerator.
   - Attach datasets: `busi-dataset` and `busbra-dataset`.
   - Run `Phase9_AuxLoss_Corrected_Reproducibility.ipynb`.
3. **Execute Primary 5-Seed Tier**:
   - Train Vanilla Swin and Standard-MLP MoE across Seeds `[42, 123, 2026, 777, 999]`.
   - Run zero-shot evaluation on BUS-BRA.
   - Download the generated `results_phase9_auxloss_corrected.zip`.
4. **Populate Verified Results**:
   - Unpack the metrics into `results/phase9_auxloss_corrected/`.
   - Update `CORRECTED_PHASE9_RESULTS.md` with the finalized numerical outcomes.
   - Execute the paired Student's t-test and Wilcoxon signed-rank test.

---

## 2. Medium-Term Roadmap: Benchmarking & Baselines

Following the completion of the corrected Phase 9B multi-seed runs, the research program should advance through three strategic stages:

### Stage 2: Classical CNN Benchmarks (Sanity Check)
- **Scientific Rationale**: A common criticism from reviewers at MICCAI/IEEE TMI is:  
  *"Does a complex 81M-parameter Vision Transformer MoE actually outperform a clean, properly-regularized 8M-parameter U-Net on a small medical dataset ($N=780$)?"*
- **Models to Train on BUSI**:
  1. **Standard U-Net** (Ronneberger et al., 2015): 4-stage encoder-decoder.
  2. **Attention U-Net** (Oktay et al., 2018): Spatial attention gates on skip connections.
- **Evaluation**: Zero-shot on BUS-BRA using identical preprocessing and resolution ($224 \times 224$).

### Stage 3: Medical Foundation Model Comparison
- **Models to Benchmark**:
  1. **MedSAM** (Medical Segment Anything Model): Evaluated under:
     - *Mode A (Oracle Box Prompt)*: Tight bounding box derived from ground truth mask with $\pm 10\text{px}$ jitter.
     - *Mode B (Fully Automatic / Zero-Shot)*: Grid-point prompt with automatic mask selection.
  2. **SAM 2** (Segment Anything Model 2): Test memory-bank feature caching on static 2D ultrasound frames.

### Stage 4: ConvNeXt-MoE (MoE for Pure CNNs)
- Formulate a hybrid architecture that replaces the pointwise $1 \times 1$ conv blocks of a **ConvNeXt-Tiny** backbone with a convolutional Mixture-of-Experts layer.
- Test whether combining convolutional spatial inductive biases with conditional MoE routing outperforms transformer-based MoE on small-scale medical datasets.

