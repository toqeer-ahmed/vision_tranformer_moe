import json

with open('notebooks/train_ablation_phase4_kaggle.ipynb', 'r', encoding='utf-8') as f:
    data = json.load(f)

data['cells'][0]['source'][0] = '# \ud83d\udd2c Breast Cancer (BUSI) Ablation Phase 4\n'
data['cells'][0]['source'][2] = 'This notebook trains **Experiment F** (Wide SegFormer Control with Warm Tiling) on Kaggle.\n'
data['cells'][5]['source'][0] = '!PYTHONPATH=. python scripts/run_ablation_phase4.py'
data['cells'][7]['source'][2] = 'shutil.make_archive("/kaggle/working/ablation_phase4_results", "zip", "outputs/")\n'
data['cells'][7]['source'][3] = 'IPython.display.FileLink("/kaggle/working/ablation_phase4_results.zip")'

with open('notebooks/train_ablation_phase4_kaggle.ipynb', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)
