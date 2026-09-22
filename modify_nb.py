import json

with open('Phase8_Swin_MoE_Kaggle.ipynb', 'r') as f:
    nb = json.load(f)

# The configuration cell (Cell 3, usually index 5 or 6 depending on markdown)
config_cell_idx = -1
for i, cell in enumerate(nb['cells']):
    if 'busi_path =' in ''.join(cell.get('source', [])):
        config_cell_idx = i
        break

if config_cell_idx != -1:
    # Modify the config cell paths
    new_source = []
    for line in nb['cells'][config_cell_idx]['source']:
        if 'busi_path =' in line:
            new_source.append('busi_path = "/kaggle/working/busi/Dataset_BUSI_with_GT"\\n')
        elif 'busbra_path =' in line:
            new_source.append('busbra_path = "/kaggle/working/busbra"\\n')
        else:
            new_source.append(line)
    nb['cells'][config_cell_idx]['source'] = new_source

# Add a download cell before the config cell
download_cell = {
    'cell_type': 'code',
    'metadata': {},
    'execution_count': None,
    'outputs': [],
    'source': [
        "# Automatically download datasets using Kaggle CLI (works natively in Kaggle Notebooks)\\n",
        "!kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset\\n",
        "!kaggle datasets download -d eugeniobigm/bus-bra-a-breast-ultrasound-dataset\\n",
        "\\n",
        "# Unzip the datasets into the working directory\\n",
        "!unzip -o -q breast-ultrasound-images-dataset.zip -d /kaggle/working/busi\\n",
        "!unzip -o -q bus-bra-a-breast-ultrasound-dataset.zip -d /kaggle/working/busbra\\n",
        "\\n",
        "print('Datasets downloaded and extracted successfully!')\\n"
    ]
}

nb['cells'].insert(config_cell_idx, download_cell)

with open('Phase8_Swin_MoE_Kaggle.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print('Notebook successfully updated to auto-download datasets!')
