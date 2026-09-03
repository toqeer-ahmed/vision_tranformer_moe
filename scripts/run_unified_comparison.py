import os
import shutil
import glob
import yaml
import subprocess
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def setup_kaggle():
    # If running on a Kaggle Kernel, authentication is either handled by Secrets or not needed if datasets are attached.
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
        print("[1/6] Running inside Kaggle Notebook. Skipping manual API token setup.")
        return
        
    os.environ['KAGGLE_API_TOKEN'] = "KGAT_f92b021c2b42601bd960c76192014a55"
    os.system("mkdir -p ~/.kaggle")
    os.system("echo 'KGAT_f92b021c2b42601bd960c76192014a55' > ~/.kaggle/access_token")
    os.system("chmod 600 ~/.kaggle/access_token")
    os.system("chmod 600 ~/.kaggle/access_token")
    print("[1/6] Kaggle authentication configured.")

def download_and_format_dataset():
    print("[2/6] Downloading BUSI dataset from Kaggle...")
    os.system("kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset")
    os.system("unzip -q breast-ultrasound-images-dataset.zip -d busi_temp")

    dest_dir = "data/medical_dataset"
    os.makedirs(os.path.join(dest_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "masks"), exist_ok=True)

    image_files = []
    for root, dirs, files in os.walk('busi_temp'):
        for file in files:
            if file.lower().endswith('.png'):
                if 'mask' not in file.lower() and 'segmentation' not in file.lower():
                    image_files.append(os.path.join(root, file))

    mask_files = []
    for root, dirs, files in os.walk('busi_temp'):
        for file in files:
            if file.lower().endswith('.png'):
                if 'mask' in file.lower() or 'segmentation' in file.lower():
                    mask_files.append(os.path.join(root, file))

    mask_dict = {}
    for m in mask_files:
        base = os.path.basename(m).replace('_mask_1', '').replace('_mask_2', '').replace('_mask', '').replace('.png', '')
        mask_dict[base] = m

    found_pairs = 0
    for img_path in image_files:
        base = os.path.basename(img_path).replace('.png', '')
        if base in mask_dict:
            shutil.copy(img_path, os.path.join(dest_dir, 'images', f'{base}.png'))
            shutil.copy(mask_dict[base], os.path.join(dest_dir, 'masks', f'{base}_mask.png'))
            found_pairs += 1

    print(f"      Successfully paired and formatted {found_pairs} BUSI images.")

def update_config(path, epochs=15):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    config['training']['epochs'] = epochs
    config['dataset']['name'] = 'medical-image-mask'
    config['dataset']['data_dir'] = 'data/medical_dataset'
    config['dataset']['batch_size'] = 2
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def configure_models():
    print("[3/6] Unifying configuration files for identical training conditions...")
    update_config("configs/medical_segmentation.yaml", epochs=20)
    update_config("configs/moe_segmentation.yaml", epochs=20)

def train_models():
    print("[4/6] Training SegMoTE (Foundation Model)...")
    subprocess.run("PYTHONPATH=. python training/train_segmote.py --config configs/medical_segmentation.yaml", shell=True, check=True)
    
    print("[4/6] Training MoE-Segformer (Baseline)...")
    subprocess.run("PYTHONPATH=. python training/train_moe.py --config configs/moe_segmentation.yaml", shell=True, check=True)

def get_best_metric(log_dir, tag='Metrics/mIoU'):
    best_val = 0.0
    if not os.path.exists(log_dir): return best_val
    for file in os.listdir(log_dir):
        if file.startswith("events.out.tfevents"):
            ea = EventAccumulator(os.path.join(log_dir, file))
            ea.Reload()
            if tag in ea.Tags()['scalars']:
                vals = [e.value for e in ea.Scalars(tag)]
                if max(vals) > best_val: best_val = max(vals)
    return best_val

def compare_results():
    print("[5/6] Parsing automated evaluation results...")
    segmote_iou = get_best_metric('outputs/medical_segmentation/logs', 'Metrics/mIoU')
    segmote_dice = get_best_metric('outputs/medical_segmentation/logs', 'Metrics/mDice')
    
    baseline_iou = get_best_metric('outputs/moe_segmentation/logs', 'Metrics/mIoU')
    baseline_dice = get_best_metric('outputs/moe_segmentation/logs', 'Metrics/mDice')

    df = pd.DataFrame({
        'Model': ['MoE-Segformer (Baseline)', 'SegMoTE (Foundation Model)'],
        'BUSI Val mIoU': [f"{baseline_iou*100:.2f}%", f"{segmote_iou*100:.2f}%"],
        'BUSI Val mDice': [f"{baseline_dice*100:.2f}%", f"{segmote_dice*100:.2f}%"],
        'Total Parameters': ['8.07 Million', '94.20 Million']
    })

    print("\n================ BUSI DATASET COMPARISON RESULTS ================")
    print(df.to_string(index=False))
    print("================================================================")

def finalize_and_zip():
    print("[6/6] Zipping results for Kaggle Output...")
    shutil.make_archive('training_results', 'zip', 'outputs')
    
    # If running on Kaggle, move the zip to the main working directory so it's easily downloadable
    # and clean up the dataset so the Kaggle output isn't cluttered.
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
        shutil.move('training_results.zip', '/kaggle/working/training_results.zip')
        print("      Results saved to /kaggle/working/training_results.zip")
        print("      You can now download this zip file directly from the Kaggle 'Output' section!")

if __name__ == "__main__":
    setup_kaggle()
    download_and_format_dataset()
    configure_models()
    train_models()
    compare_results()
    finalize_and_zip()
