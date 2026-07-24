y# Vision Transformer & Mixture of Experts (MoE) Research Framework

A unified, modular, research-oriented computer vision framework written in Python and PyTorch. This framework progresses from standard Vision Transformer (ViT) classification to SegFormer-based semantic segmentation, and finally to a SegFormer model augmented with a dynamic Mixture of Experts (MoE) layer.

This codebase is designed using software engineering best practices (OOP, strong typing, clean configs) to serve as a prerequisite and launching pad for advanced graduate computer vision research.

---

## Repository Structure

```text
vision_transformer_research/
├── configs/
│   ├── classification.yaml       # Configuration for Phase 1 (ViT classification)
│   ├── segmentation.yaml         # Configuration for Phase 2 (SegFormer segmentation)
│   └── moe_segmentation.yaml     # Configuration for Phase 3 (MoE SegFormer segmentation)
├── datasets/
│   ├── classification_dataset.py # CIFAR-10/100 dataloaders with Albumentations
│   └── segmentation_dataset.py   # Oxford-IIIT Pet segmentation dataloader
├── models/
│   ├── vit_classifier.py         # ViT classification wrapper
│   ├── segformer.py              # SegFormer segmentation wrapper
│   └── moe/
│       ├── experts.py            # MLP Experts definitions
│       ├── gating.py             # Gating networks (Linear/Noisy gating)
│       ├── router.py             # Top-K router & load balancing loss computation
│       └── moe_layer.py          # Unified Mixture of Experts Layer
├── training/
│   ├── train_classifier.py       # Fine-tuning classification pipeline
│   ├── train_segmentation.py     # Fine-tuning segmentation pipeline
│   └── train_moe.py              # Training pipeline with dynamic MoE replacement
├── evaluation/
│   ├── metrics.py                # Dice, IoU, Accuracy, F1, Precision, Recall
│   └── visualize_predictions.py  # Loss/Acc/IoU curves & ground truth-vs-mask plots
├── inference/
│   └── predict.py                # Command-line inference script for classification/segmentation
├── utils/
│   ├── logger.py                 # Structured console and file logger
│   ├── seed.py                   # Global random seed generator
│   └── checkpoint.py             # PyTorch state dictionary saving/loading
├── requirements.txt              # Framework dependencies
└── README.md                     # Main documentation (this file)
```

---

## Installation

### 1. Setup Virtual Environment
Run the following inside your terminal to set up a virtual environment and activate it:
```powershell
# Create virtual environment (if not already created)
python -m venv .venv

# Activate on Windows Powershell
.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip
```

### 2. Install Dependencies
Install all package requirements for this framework:
```powershell
pip install -r vision_transformer_research/requirements.txt
```

---

## Dataset Preparation

* **Phase 1 (CIFAR-10/100):** The dataset downloads automatically to the `./data` directory when the training script is run. No manual setup is required.
* **Phase 2 & 3 (Oxford-IIIT Pet):** The dataset downloads automatically via `torchvision.datasets.OxfordIIITPet` to `./data` during training. If you already have it downloaded, configure `data_dir` in `configs/segmentation.yaml` to point to the directory containing it.

---

## Configuration System

All tasks run based on YAML configuration files in `configs/`. Key parameters include:
* `fast_dev_run`: Set to `true` to perform a quick end-to-end integration dry-run using mock data (bypasses internet downloads and completes in under 10 seconds).
* `pretrained`: Set to `true` to fetch pre-trained weights from Hugging Face Hub (set to `false` for random initializations/offline runs).

---

## Training Commands

Run all training scripts from the workspace root (`d:\Study Material\Research\CV`):

### Phase 1: ViT Image Classification
```powershell
python vision_transformer_research/training/train_classifier.py --config vision_transformer_research/configs/classification.yaml
```

### Phase 2: SegFormer Semantic Segmentation
```powershell
python vision_transformer_research/training/train_segmentation.py --config vision_transformer_research/configs/segmentation.yaml
```

### Phase 3: SegFormer + MoE Semantic Segmentation
```powershell
python vision_transformer_research/training/train_moe.py --config vision_transformer_research/configs/moe_segmentation.yaml
```

---

## Evaluation & Metrics

During training, progress is monitored and logged. Output directories generate:
* **Training Logs:** Written to `outputs/<task>/logs/` (and printed to console).
* **Saved Checkpoints:** Stored in `outputs/<task>/checkpoints/` as `last_checkpoint.pth` and `best_model.pth`.
* **Plots:** Saved to `outputs/<task>/plots/` (includes accuracy/loss curves, segmentation curves, prediction visual maps, and confusion matrices).
* **TensorBoard Visualization:**
  To launch TensorBoard and visualize metrics interactively:
  ```powershell
  tensorboard --logdir=outputs/
  ```

---

## Inference commands

To run prediction on a custom input image:

### Image Classification Prediction
```powershell
python vision_transformer_research/inference/predict.py --task classification --config vision_transformer_research/configs/classification.yaml --checkpoint outputs/classification/checkpoints/best_model.pth --image path/to/image.jpg
```

### Semantic Segmentation Prediction
```powershell
python vision_transformer_research/inference/predict.py --task segmentation --config vision_transformer_research/configs/segmentation.yaml --checkpoint outputs/segmentation/checkpoints/best_model.pth --image path/to/image.jpg --save_path outputs/predictions/segmentation_result.png
```

### MoE Semantic Segmentation Prediction
```powershell
python vision_transformer_research/inference/predict.py --task moe_segmentation --config vision_transformer_research/configs/moe_segmentation.yaml --checkpoint outputs/moe_segmentation/checkpoints/best_model.pth --image path/to/image.jpg --save_path outputs/predictions/moe_segmentation_result.png
```

---

## Streamlit Visualization Dashboard

The framework includes an interactive visual dashboard built with Streamlit. It allows you to:
1. Compare architecture specs and dynamically calculated parameter sizes for Phase 1, Phase 2, and Phase 3 models.
2. Run sample inference sandbox operations using synthetic data previews (perfect for offline runs).
3. Visualize live routing token distribution counts per expert across all 8 MoE layers in real-time.
4. Review the code tour explaining dynamic FFN-to-MoE block substitution.

To launch the dashboard locally, execute:
```powershell
streamlit run vision_transformer_research/app.py
```

---

## Training on Medical Datasets & Google Colab

The framework supports custom medical image segmentation datasets (such as ISIC skin lesion scans or BUSI breast cancer ultrasound datasets). 

### 1. Data Structure Setup
To load your custom medical dataset, arrange your files in the following format:
```text
data/medical_dataset/
├── images/
│   ├── patient_001.jpg
│   ├── patient_002.jpg
└── masks/
    ├── patient_001_segmentation.png
    └── patient_002_segmentation.png
```
*Note: The dataloader matches file stems automatically, so `patient_001.jpg` in `images/` will pair with `patient_001_segmentation.png` in `masks/`.*

### 2. Launch Local Training
Run the training script on the GPU using the medical configurations:
```powershell
python vision_transformer_research/training/train_moe.py --config vision_transformer_research/configs/medical_segmentation.yaml
```

### 3. Launching on Google Colab (Recommended for Free GPU access)
To train using Colab's T4 or A100 GPU:
1. Open Google Colab and select **Upload Notebook**.
2. Upload the provided Jupyter Notebook: [notebooks/train_colab.ipynb](file:///d:/Study%20Material/Research/CV/vision_transformer_research/notebooks/train_colab.ipynb).
3. Change the Colab runtime to a **GPU hardware accelerator** (`Runtime > Change runtime type > GPU`).
4. Execute the cells sequentially to clone your repository, install packages, upload your dataset, and train.

---

## How to Extend the Framework for MoE Research

This framework was built from the ground up for academic and professional research. You can easily plug in new research components by following these interfaces:

### 1. Adding a New Gating Network
Inherit from `nn.Module` in `models/moe/gating.py` and output a tensor of shape `(num_tokens, num_experts)`.
Example: Implementing learning-to-route networks, sigmoid gating, or reinforcement-learning routers.

### 2. Custom Router / Routing Strategy
Modify `models/moe/router.py` to change how indices are chosen and how gating weights are normalized. Here, you can also inject custom auxiliary loss terms such as:
* Load balancing loss
* Expert capacity utilization loss
* Sparsity regularization penalties

### 3. Creating Diverse Experts
Experts do not need to be uniform MLPs. You can replace the uniform `MLPExpert` blocks in `models/moe/experts.py` with:
* Convolutional experts (such as MixFFN experts with different kernel sizes)
* Attention-augmented experts
* Low-rank adaptation (LoRA) experts
