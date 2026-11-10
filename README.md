# PBL6: Ischemic Stroke Lesion Segmentation (ISLES 2022)

Automated ischemic stroke lesion segmentation from multi-modal brain MRI scans (DWI, ADC, FLAIR) using deep learning.

## Repository Structure

```text
PBL6/
├── configs/             # Configuration files (YAML) for training and experiments
├── data/                # Dataset directory (Ignored by Git)
│   ├── raw/             # Raw BIDS-format ISLES-2022 dataset (immutable source)
│   ├── interim/         # Intermediate preprocessed data
│   └── processed/       # Resampled, normalized volumes ready for training
├── docs/                # Research papers, medical background notes, slicer scenes
├── experiments/         # Experiment logs, checkpoints, and tracking
├── notebooks/           # Jupyter notebooks for EDA and visualization
├── results/             # Evaluation metrics, predictions, reports
├── scripts/             # Entry point CLI scripts (inspect, preprocess, train, eval)
├── src/                 # Reusable source code modules
│   ├── dataset/         # PyTorch / MONAI dataset loaders & transforms
│   ├── models/          # Neural network architectures (UNet3D, Attention UNet, etc.)
│   ├── metrics/         # Medical segmentation metrics (Dice, HD95)
│   └── utils/           # Utilities (logger, path resolvers, seed)
├── requirements.txt     # Python dependencies
└── .gitignore           # Git ignore rules
```

## Quick Start

### 1. Environment Setup
Create and activate virtual environment:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Dataset Setup
Download the ISLES-2022 dataset and place it under `data/raw/`:
```text
data/raw/ISLES-2022/
├── dataset_description.json
├── participants.tsv
├── sub-strokecase0001/
└── ...
```

### 3. Inspect Dataset
Scan the dataset and generate shape & spacing statistics:
```bash
python scripts/inspect_dataset.py
```

## Documentation & Decisions

- **[ISLES-2022 Knowledge Vault](notes/00_ISLES2022_Knowledge_Vault.md):** Medical domain context, ischemic cascade, multi-modal MRI signals, and challenge guidelines.
- **[Model Benchmark Report](Benchmark.md):** Comparative benchmark on ISLES-2022 dataset and selection of **MedNeXt** (MICCAI 2023) as core contribution.
- **[Cloud GPU Pricing & Training Guide](server_pricing.md):** GPU server rental analysis (RunPod / Vast.ai), budget estimation (~$10 USD), and remote training workflow.

