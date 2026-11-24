import json
import os

nb_path = "Phase9_AuxLoss_Corrected_Reproducibility.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 1: Clean and robust clone
cell_1_source = [
    "# 1. SETUP ENVIRONMENT & REPO (FRESH CLONE FROM MAIN)\n",
    "import os, sys, shutil, glob, torch, json\n",
    "\n",
    "# Ensure clean root in Kaggle\n",
    "if os.path.exists('/kaggle/working'):\n",
    "    os.chdir('/kaggle/working')\n",
    "    !rm -rf /kaggle/working/vision_transformer_research\n",
    "\n",
    "print('Fetching latest repository from GitHub (main branch)...')\n",
    "!git clone --depth 1 -b main https://github.com/toqeer-ahmed/vision_tranformer_moe.git vision_transformer_research\n",
    "\n",
    "repo_dir = '/kaggle/working/vision_transformer_research' if os.path.exists('/kaggle/working/vision_transformer_research') else os.path.abspath('vision_transformer_research')\n",
    "os.chdir(repo_dir)\n",
    "\n",
    "sys.path.insert(0, repo_dir)\n",
    "if 'datasets' in sys.modules:\n",
    "    del sys.modules['datasets']\n",
    "\n",
    "!pip install -r requirements.txt -q\n",
    "!pip install albumentations -q\n",
    "\n",
    "# Run diagnostic verification\n",
    "!python tests/test_moe_aux_loss.py\n"
]

# Cell 4: Python zipfile and copy to /kaggle/working
cell_4_code = nb["cells"][4]["source"]
new_cell_4 = []
for line in cell_4_code:
    if line.startswith("!zip"):
        new_cell_4.append("import zipfile\n")
        new_cell_4.append("zip_filename = 'results_phase9_auxloss_corrected.zip'\n")
        new_cell_4.append("print(f'Creating {zip_filename} archive...')\n")
        new_cell_4.append("with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:\n")
        new_cell_4.append("    for folder in ['results', 'configs/phase9_auxloss_corrected']:\n")
        new_cell_4.append("        if os.path.exists(folder):\n")
        new_cell_4.append("            for root, _, files in os.walk(folder):\n")
        new_cell_4.append("                for file in files:\n")
        new_cell_4.append("                    full_p = os.path.join(root, file)\n")
        new_cell_4.append("                    zipf.write(full_p, full_p)\n")
        new_cell_4.append("    if os.path.exists('outputs'):\n")
        new_cell_4.append("        for root, _, files in os.walk('outputs'):\n")
        new_cell_4.append("            for file in files:\n")
        new_cell_4.append("                if file == 'best_model.pth' or file.endswith(('.json', '.csv', '.png', '.svg', '.txt')) or 'events.out.tfevents' in file:\n")
        new_cell_4.append("                    full_p = os.path.join(root, file)\n")
        new_cell_4.append("                    zipf.write(full_p, full_p)\n")
        new_cell_4.append("if os.path.exists('/kaggle/working'):\n")
        new_cell_4.append("    shutil.copy(zip_filename, os.path.join('/kaggle/working', zip_filename))\n")
    elif "Done! Download results_phase9_auxloss_corrected.zip" in line:
        new_cell_4.append("print(f'\\nDone! Download {zip_filename} ({os.path.getsize(zip_filename)/(1024*1024):.2f} MB) from Kaggle output window.')\n")
    else:
        new_cell_4.append(line)

nb["cells"][1]["source"] = cell_1_source
nb["cells"][4]["source"] = new_cell_4

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Updated Phase9_AuxLoss_Corrected_Reproducibility.ipynb successfully.")
