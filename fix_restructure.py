import json

with open('Phase8_Swin_MoE_Kaggle.ipynb', 'r') as f:
    nb = json.load(f)

restructure_code = """
import os
import shutil
import glob

# The training scripts hardcode the dataset path to 'data/medical_dataset'
# We must restructure the Kaggle upload into 'images/' and 'masks/' folders
os.makedirs("data/medical_dataset/images", exist_ok=True)
os.makedirs("data/medical_dataset/masks", exist_ok=True)

print("Restructuring BUSI dataset into images/ and masks/ for the dataloader...")
if busi_path and os.path.exists(busi_path):
    all_pngs = glob.glob(os.path.join(busi_path, "**", "*.png"), recursive=True)
    img_count = 0
    mask_count = 0
    
    for file_path in all_pngs:
        filename = os.path.basename(file_path)
        if "mask" in filename.lower():
            dest = os.path.join("data/medical_dataset/masks", filename)
            if not os.path.exists(dest):
                shutil.copy2(file_path, dest)
            mask_count += 1
        else:
            dest = os.path.join("data/medical_dataset/images", filename)
            if not os.path.exists(dest):
                shutil.copy2(file_path, dest)
            img_count += 1
            
    print(f"Copied {img_count} images and {mask_count} masks.")
else:
    print("WARNING: busi_path not found! The training scripts will fail!")
"""

# Find the cell that dynamically locates the dataset paths
target_idx = -1
for i, cell in enumerate(nb['cells']):
    if 'busi_path = find_dataset_path' in ''.join(cell.get('source', [])):
        target_idx = i
        break

if target_idx != -1:
    # Insert the restructure cell right after the path finding cell
    new_cell = {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [line + '\n' for line in restructure_code.strip().split('\n')]
    }
    nb['cells'].insert(target_idx + 1, new_cell)

with open('Phase8_Swin_MoE_Kaggle.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("Injected dataset restructuring code!")
