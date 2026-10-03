import json

with open('Phase8_Swin_MoE_Kaggle.ipynb', 'r') as f:
    nb = json.load(f)

dynamic_path_code = """import os
import yaml

def find_dataset_path(base_dir, target_folder_name):
    print(f"Searching for {target_folder_name} in {base_dir}...")
    for root, dirs, files in os.walk(base_dir):
        if target_folder_name in dirs:
            found_path = os.path.join(root, target_folder_name)
            print(f"Found at: {found_path}")
            return found_path
    print(f"WARNING: Could not find {target_folder_name}!")
    return None

# Dynamically find the datasets regardless of how Kaggle extracted the zip folders (archive / archive 1)
busi_path = find_dataset_path("/kaggle/input", "Dataset_BUSI_with_GT")
busbra_path = find_dataset_path("/kaggle/input", "BUSBRA")

# Fallback just in case
if not busi_path: busi_path = "/kaggle/input/busi-dataset/Dataset_BUSI_with_GT"
if not busbra_path: busbra_path = "/kaggle/input/busbra-dataset/BUSBRA"
"""

for cell in nb['cells']:
    if 'busi_path = "/kaggle/input/busi-dataset/' in ''.join(cell.get('source', [])):
        cell['source'] = [line + '\n' for line in dynamic_path_code.split('\n')]

with open('Phase8_Swin_MoE_Kaggle.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print('Dynamic path finding added to the notebook!')
