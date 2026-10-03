import json

notebook = {
    'cells': [
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '# Phase 9: Controlled Reproducibility Experiment\n',
                'This notebook trains 3 seeds of Vanilla Swin-Tiny and 3 seeds of Standard-MLP Swin-MoE under **identical** training protocols, then performs zero-shot BUSBRA evaluation.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                '# 1. SETUP ENVIRONMENT & REPO\n',
                'import os, sys, shutil, glob, torch, json\n',
                '\n',
                '!rm -rf vision_transformer_research\n',
                '!git clone https://github.com/toqeer-ahmed/vision_tranformer_moe.git vision_transformer_research\n',
                'os.chdir("vision_transformer_research")\n',
                'sys.path.append(os.path.abspath("."))\n',
                '\n',
                '!pip install -r requirements.txt -q\n',
                '\n',
                '# Print git hash\n',
                '!git rev-parse HEAD\n'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                '# 2. BULLETPROOF DATASET MOUNTING\n',
                'print("Setting up dataset paths...")\n',
                'os.makedirs("data/medical_dataset/images", exist_ok=True)\n',
                'os.makedirs("data/medical_dataset/masks", exist_ok=True)\n',
                'os.makedirs("data/busbra_dataset/images", exist_ok=True)\n',
                'os.makedirs("data/busbra_dataset/masks", exist_ok=True)\n',
                '\n',
                'for root_dir, _, files in os.walk("/kaggle/input", followlinks=True):\n',
                '    for file in files:\n',
                '        if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):\n',
                '            src = os.path.join(root_dir, file)\n',
                '            if "busi" in src.lower() and "checkpoints" not in src.lower():\n',
                '                if "_mask" in file.lower():\n',
                '                    shutil.copy(src, os.path.join("data/medical_dataset/masks", file))\n',
                '                else:\n',
                '                    shutil.copy(src, os.path.join("data/medical_dataset/images", file))\n',
                '            elif "busbra" in src.lower():\n',
                '                if "mask" in src.lower():\n',
                '                    shutil.copy(src, os.path.join("data/busbra_dataset/masks", file))\n',
                '                else:\n',
                '                    shutil.copy(src, os.path.join("data/busbra_dataset/images", file))\n',
                '\n',
                'print(f"Total BUSI images: {len(os.listdir(\'data/medical_dataset/images\'))}")\n',
                'print(f"Total BUSBRA images: {len(os.listdir(\'data/busbra_dataset/images\'))}")'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                '# 3. TRAIN ALL 6 MODELS\n',
                'seeds = [42, 123, 2026]\n',
                '\n',
                'for seed in seeds:\n',
                '    print(f"\\n======================================\\n")\n',
                '    print(f"TRAINING VANILLA SWIN - SEED {seed}")\n',
                '    print(f"\\n======================================\\n")\n',
                '    !python training/train_swin_vanilla.py --config configs/reproducibility/swin_vanilla_seed{seed}.yaml\n',
                '    \n',
                '    print(f"\\n======================================\\n")\n',
                '    print(f"TRAINING STANDARD-MLP MOE - SEED {seed}")\n',
                '    print(f"\\n======================================\\n")\n',
                '    # Note: --warm-init ensures the MoE is properly initialized from the backbone\n',
                '    !python training/train_swin_standard_moe.py --config configs/reproducibility/swin_standard_moe_seed{seed}.yaml --warm-init'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                '# 4. ZERO-SHOT EVALUATION ON BUSBRA\n',
                'from datasets.medical_dataset import MedicalImageMaskDataset, get_medical_transforms\n',
                'from evaluation.metrics import SegmentationMetricAccumulator\n',
                'from torch.utils.data import DataLoader\n',
                'from models.swin import SwinSegmentation, replace_swin_ffn_with_standard_moe\n',
                'import yaml\n',
                '\n',
                '_, transform = get_medical_transforms(224)\n',
                'busbra_dataset = MedicalImageMaskDataset("data/busbra_dataset", transform=transform)\n',
                'busbra_loader = DataLoader(busbra_dataset, batch_size=16, shuffle=False, num_workers=2)\n',
                'device = "cuda" if torch.cuda.is_available() else "cpu"\n',
                '\n',
                'def evaluate_checkpoint(ckpt_path, config_path, is_moe):\n',
                '    with open(config_path, "r") as f: config = yaml.safe_load(f)\n',
                '    model = SwinSegmentation(num_classes=3, pretrained=False)\n',
                '    if is_moe: replace_swin_ffn_with_standard_moe(model, config["moe"])\n',
                '    ckpt = torch.load(ckpt_path, map_location=device)\n',
                '    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt, strict=False)\n',
                '    model.to(device).eval()\n',
                '    \n',
                '    acc = SegmentationMetricAccumulator(num_classes=2)\n',
                '    with torch.no_grad():\n',
                '        for imgs, masks in busbra_loader:\n',
                '            imgs, masks = imgs.to(device), masks.to(device)\n',
                '            with torch.amp.autocast("cuda"):\n',
                '                logits = model(imgs)\n',
                '            preds = (logits.argmax(dim=1) > 0).long() # Map benign/malignant to tumor(1)\n',
                '            acc.update(preds, masks)\n',
                '    return acc.compute()\n',
                '\n',
                'results = {}\n',
                'for model_type in ["vanilla", "standard_moe"]:\n',
                '    for seed in seeds:\n',
                '        name = f"swin_{model_type}_seed{seed}"\n',
                '        # Find the output directory (latest matching)\n',
                '        dirs = sorted(glob.glob(f"outputs/{name}_*"))\n',
                '        if not dirs:\n',
                '            print(f"Skipping {name} (no outputs found)")\n',
                '            continue\n',
                '        ckpt_path = os.path.join(dirs[-1], "checkpoints", "best_model.pth")\n',
                '        if not os.path.exists(ckpt_path):\n',
                '            print(f"Skipping {name} (no checkpoint found)")\n',
                '            continue\n',
                '            \n',
                '        cfg_path = f"configs/reproducibility/{name}.yaml"\n',
                '        print(f"Evaluating {name} zero-shot...")\n',
                '        metrics = evaluate_checkpoint(ckpt_path, cfg_path, is_moe=(model_type=="standard_moe"))\n',
                '        results[name] = metrics\n',
                '\n',
                'os.makedirs("results/phase9", exist_ok=True)\n',
                'with open("results/phase9/reproducibility_metrics.json", "w") as f:\n',
                '    clean_results = {name: {k: (v.tolist() if isinstance(v, torch.Tensor) else v) for k,v in m.items()} for name, m in results.items()}\n',
                '    json.dump(clean_results, f, indent=4)\n',
                'print("Results saved to results/phase9/reproducibility_metrics.json")'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                '# 5. GENERATE STATISTICAL SUMMARY\n',
                'print("\\n=== REPRODUCIBILITY STATISTICAL SUMMARY ===")\n',
                'def print_stats(model_type):\n',
                '    mious, fg_ious, mdices, fg_dices, precs, recs = [], [], [], [], [], []\n',
                '    for seed in seeds:\n',
                '        name = f"swin_{model_type}_seed{seed}"\n',
                '        if name in results:\n',
                '            m = results[name]\n',
                '            mious.append(m["mean_iou"])\n',
                '            fg_ious.append(m["class_iou"][1])\n',
                '            mdices.append(m["mean_dice"])\n',
                '            fg_dices.append(m["class_dice"][1])\n',
                '            precs.append(m["class_precision"][1])\n',
                '            recs.append(m["class_recall"][1])\n',
                '            print(f"{name}: mIoU={m[\'mean_iou\']:.4f}, FG-IoU={m[\'class_iou\'][1]:.4f}")\n',
                '    if mious:\n',
                '        print(f"-> {model_type.upper()} MEAN ± SD:")\n',
                '        print(f"   mIoU: {np.mean(mious):.4f} ± {np.std(mious):.4f}")\n',
                '        print(f"   FG-IoU: {np.mean(fg_ious):.4f} ± {np.std(fg_ious):.4f}\\n")\n',
                '\n',
                'print_stats("vanilla")\n',
                'print_stats("standard_moe")'
            ]
        }
    ],
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

with open('Phase9_Controlled_Reproducibility.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
