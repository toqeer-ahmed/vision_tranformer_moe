import json

with open('Phase8_Swin_MoE_Kaggle.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if '!kaggle datasets download -d eugeniobigm' in ''.join(cell.get('source', [])):
        new_source = []
        for line in cell['source']:
            if 'bus-bra' in line and '!kaggle' in line:
                new_source.append('# NOTE: Please manually upload your local BUS-BRA dataset to Kaggle!\\n')
            elif 'bus-bra' in line and '!unzip' in line:
                continue
            else:
                new_source.append(line)
        cell['source'] = new_source

for cell in nb['cells']:
    if 'busbra_path = "/kaggle/working/busbra"\\n' in cell.get('source', []):
        new_source = []
        for line in cell['source']:
            if 'busbra_path =' in line:
                new_source.append('busbra_path = "/kaggle/input/bus-bra-a-breast-ultrasound-dataset" # Upload manually!\\n')
            else:
                new_source.append(line)
        cell['source'] = new_source

with open('Phase8_Swin_MoE_Kaggle.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print('Notebook updated successfully!')
