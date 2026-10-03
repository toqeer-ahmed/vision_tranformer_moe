import json

with open('Phase8_Swin_MoE_Kaggle.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if '!kaggle datasets download -d aryashah2k' in ''.join(cell.get('source', [])):
        cell['source'] = [
            '# Please manually upload BOTH datasets to Kaggle!\n', 
            '# We have removed the auto-download code because the Kaggle datasets were deleted.\n'
        ]

for cell in nb['cells']:
    if 'busi_path = "/kaggle/working/busi/' in ''.join(cell.get('source', [])):
        new_source = []
        for line in cell['source']:
            if 'busi_path =' in line:
                new_source.append('busi_path = "/kaggle/input/busi-dataset/Dataset_BUSI_with_GT" # Upload manually and name it busi-dataset\n')
            elif 'busbra_path =' in line:
                new_source.append('busbra_path = "/kaggle/input/busbra-dataset/BUSBRA" # Upload manually and name it busbra-dataset\n')
            else:
                new_source.append(line)
        cell['source'] = new_source

with open('Phase8_Swin_MoE_Kaggle.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print('Notebook paths updated successfully!')
