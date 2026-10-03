import json

with open('Phase8_Swin_MoE_Kaggle.ipynb', 'r') as f:
    nb = json.load(f)

new_setup_source = [
    "import os\n",
    "import sys\n",
    "\n",
    "# Ensure we are in the root directory before cloning (safe for multiple runs)\n",
    "os.chdir('/kaggle/working')\n",
    "\n",
    "if not os.path.exists('vision_transformer_research'):\n",
    "    !git clone https://github.com/toqeer-ahmed/vision_tranformer_moe.git vision_transformer_research\n",
    "else:\n",
    "    !cd vision_transformer_research && git pull\n",
    "\n",
    "os.chdir('vision_transformer_research')\n",
    "sys.path.append(os.path.abspath('.'))\n"
]

# Find and replace the setup cell
for cell in nb['cells']:
    source_str = ''.join(cell.get('source', []))
    if '!git clone https://github.com/toqeer-ahmed/vision_tranformer_moe.git vision_transformer_research' in source_str:
        cell['source'] = new_setup_source

with open('Phase8_Swin_MoE_Kaggle.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print('Updated Kaggle notebook setup cell to be idempotent!')
