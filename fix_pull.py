import json

with open('Phase8_Swin_MoE_Kaggle.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    source_str = ''.join(cell.get('source', []))
    if '!cd vision_transformer_research && git pull' in source_str:
        new_source = source_str.replace(
            '!cd vision_transformer_research && git pull', 
            '!cd vision_transformer_research && git fetch origin main && git reset --hard origin/main'
        )
        cell['source'] = [line + '\n' for line in new_source.split('\n') if line]

with open('Phase8_Swin_MoE_Kaggle.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)
