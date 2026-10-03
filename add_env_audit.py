import json

with open('Phase8_Swin_MoE_Kaggle.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

audit_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import sys\n",
        "import torch\n",
        "import os\n",
        "\n",
        "print(\"Generating Environment Audit...\")\n",
        "os.makedirs('outputs', exist_ok=True)\n",
        "with open('outputs/environment.txt', 'w') as f:\n",
        "    f.write(\"=== Kaggle Environment Audit ===\\n\")\n",
        "    f.write(f\"Python Version: {sys.version}\\n\")\n",
        "    f.write(f\"PyTorch Version: {torch.__version__}\\n\")\n",
        "    f.write(f\"CUDA Available: {torch.cuda.is_available()}\\n\")\n",
        "    if torch.cuda.is_available():\n",
        "        f.write(f\"Device Name: {torch.cuda.get_device_name(0)}\\n\")\n",
        "        f.write(f\"CUDA Version: {torch.version.cuda}\\n\")\n",
        "    f.write(\"\\n=== Installed Packages ===\\n\")\n",
        "    import subprocess\n",
        "    result = subprocess.run(['pip', 'freeze'], capture_output=True, text=True)\n",
        "    f.write(result.stdout)\n",
        "print(\"Environment Audit saved to outputs/environment.txt\")\n"
    ]
}

nb['cells'].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 11. Environment Audit"]
})
nb['cells'].append(audit_cell)

with open('Phase8_Swin_MoE_Kaggle.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Added environment audit cell.")
