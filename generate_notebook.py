import json

notebook = {
    'cells': [
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '# Phase 9: Standard-MLP Swin-MoE Kaggle Experiment\n',
                '\n',
                'This notebook trains the Phase 9 `swin_standard_moe` model, which uses standard `Linear -> GELU -> Linear` experts instead of injecting spatial depthwise convolutions. This allows us to cleanly isolate the effect of conditional routing on zero-shot generalization.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                '# ==========================================\n',
                '# STEP 1: ENVIRONMENT SETUP\n',
                '# ==========================================\n',
                'import os\n',
                'import sys\n',
                'import shutil\n',
                '\n',
                '# Clone the repository\n',
                '!git clone https://github.com/toqeer-ahmed/vision_tranformer_moe.git vision_transformer_research\n',
                'os.chdir(\'vision_transformer_research\')\n',
                'sys.path.append(os.path.abspath(\'.\'))\n',
                '\n',
                '# Install requirements\n',
                '!pip install -r requirements.txt -q'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                '# ==========================================\n',
                '# STEP 2: RESTRUCTURE DATASETS\n',
                '# ==========================================\n',
                'print("Setting up dataset paths...")\n',
                'os.makedirs("data/medical_dataset/images", exist_ok=True)\n',
                'os.makedirs("data/medical_dataset/masks", exist_ok=True)\n',
                '\n',
                '# Copy BUSI (Adjust the input paths if your Kaggle dataset names differ slightly)\n',
                'for file in os.listdir("/kaggle/input/busi-dataset/images"):\n',
                '    shutil.copy(f"/kaggle/input/busi-dataset/images/{file}", f"data/medical_dataset/images/{file}")\n',
                'for file in os.listdir("/kaggle/input/busi-dataset/masks"):\n',
                '    shutil.copy(f"/kaggle/input/busi-dataset/masks/{file}", f"data/medical_dataset/masks/{file}")'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                '# ==========================================\n',
                '# STEP 3: CONFIGURE EXPERIMENT\n',
                '# ==========================================\n',
                'config_yaml = """\\\n',
                'model:\n',
                '  name: "swin_standard_moe"\n',
                '  backbone: "microsoft/swin-tiny-patch4-window7-224"\n',
                '  num_classes: 3\n',
                '\n',
                'moe:\n',
                '  num_experts: 4\n',
                '  top_k: 2\n',
                '  noisy_gating: true\n',
                '  balance_loss_coef: 0.01\n',
                '  warm_init: true\n',
                '\n',
                'training:\n',
                '  batch_size: 32\n',
                '  learning_rate: 0.0001\n',
                '  weight_decay: 0.01\n',
                '  num_epochs: 50\n',
                '  early_stopping_patience: 10\n',
                '  img_size: 224\n',
                '  seed: 42\n',
                '"""\n',
                'os.makedirs("configs", exist_ok=True)\n',
                'with open("configs/swin_standard_moe_kaggle.yaml", "w") as f:\n',
                '    f.write(config_yaml)\n',
                'print("Configuration saved.")'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                '# ==========================================\n',
                '# STEP 4: SMOKE TEST (1 BATCH)\n',
                '# ==========================================\n',
                'print("Running Smoke Test...")\n',
                'import torch\n',
                'import yaml\n',
                'from models.swin import SwinSegmentation, replace_swin_ffn_with_standard_moe\n',
                'from datasets.medical_dataset import get_medical_dataloaders\n',
                'from evaluation.losses import CombinedSegmentationLoss\n',
                '\n',
                'with open("configs/swin_standard_moe_kaggle.yaml", "r") as f:\n',
                '    config = yaml.safe_load(f)\n',
                '\n',
                '# Initialize Model and Inject MoE\n',
                'model = SwinSegmentation(num_classes=3, pretrained=True)\n',
                'replace_swin_ffn_with_standard_moe(model, config["moe"])\n',
                'model.cuda()\n',
                '\n',
                '# Get 1 Batch\n',
                'train_loader, _, _ = get_medical_dataloaders("data/medical_dataset", batch_size=2, img_size=224, seed=42)\n',
                'images, targets = next(iter(train_loader))\n',
                'images, targets = images.cuda(), targets.cuda()\n',
                '\n',
                '# Forward, Loss, Backward\n',
                'optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)\n',
                'criterion = CombinedSegmentationLoss(loss_type="focal_tversky").cuda()\n',
                '\n',
                'with torch.amp.autocast("cuda"):\n',
                '    logits = model(images)\n',
                '    loss = criterion(logits, targets)\n',
                '\n',
                'loss.backward()\n',
                'optimizer.step()\n',
                '\n',
                'print(f"Smoke Test Passed! Initial Loss: {loss.item():.4f}")\n',
                'print("Gradients successfully flowed through Standard-MLP MoE layers.")\n',
                'del model, optimizer, criterion, images, targets\n',
                'torch.cuda.empty_cache()'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                '# ==========================================\n',
                '# STEP 5: FULL TRAINING\n',
                '# ==========================================\n',
                'print("Starting Full Training...")\n',
                '!python training/train_swin_standard_moe.py --config configs/swin_standard_moe_kaggle.yaml --warm-init'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                '# ==========================================\n',
                '# STEP 6: ZERO-SHOT EVALUATION ON BUS-BRA\n',
                '# ==========================================\n',
                'print("Setting up BUS-BRA for Zero-Shot Evaluation...")\n',
                '# Check your exact BUS-BRA input path. It might be /kaggle/input/busbra-dataset/ or similar\n',
                '!python scripts/evaluate_zero_shot.py --model swin_standard_moe --checkpoint outputs/swin_standard_moe_*/checkpoints/best_model.pth --dataset busbra --data_dir /kaggle/input/busbra-dataset'
            ]
        }
    ],
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.10.12'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

with open('Phase9_Standard_MLP_Swin_MoE.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
