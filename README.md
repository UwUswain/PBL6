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

## Web Demo

A local web app for viewing segmentation results: a clinician workspace (3 planes + 3D, lesion volume, comparison with the expert label) and a patient explanation mode that unlocks only after the clinician approves. Vietnamese and English UI. Runs fully offline.

### Run

From the repository root, with the environment above installed:

```bash
python -m uvicorn web.app:app --port 8000
```

Open http://127.0.0.1:8000. A CUDA GPU is used when available (about 2–3 s per case on an RTX 4060); CPU works but is slower.

Requirements:
- A checkpoint at the path listed in `web/config.yaml` (default: `experiments/runs/segresnet_fold0_20260929_003648/best_metric_model.pth`).
- For the sample-case list: the ISLES-2022 dataset under `ISLES-2022/` (path set by `data_dir` in `web/config.yaml`). Without it, use the "Upload files" tab.

### Demo walkthrough

1. Pick a case from **Featured cases** (all from the fold-0 validation set, so the default model never trained on them), then **Run segmentation**.
2. Tick **Compare with expert label**: red = AI only, green = expert only (missed by AI), orange = both.
3. Write an optional note for the patient and click **Approve**: the view switches to the see-through 3D brain with a plain-language explanation.
4. **Export PDF report** opens the browser print dialog; choose "Save as PDF".
5. **About the model** shows the architecture and evaluation metrics.

### Swapping in a new model

Add an entry under `models` in `web/config.yaml` (architecture, input channels in training order, checkpoint paths — several paths make an averaged ensemble — patch size, threshold, metrics) and set `default_model`. Restart the server, then run `python -m pytest tests/test_web.py`. No code change is needed as long as the model uses the same preprocessing as `get_val_transforms` (1 mm spacing, foreground crop, per-channel z-score).

### Editing patient-facing text

All patient explanation text (Vietnamese and English) lives in `web/static/patient_text.json`. It is a draft and has not been reviewed by a medical professional.

### Brain regions

`web/regions.py` registers the MNI152 brain mask to the DWI brain mask (affine, SimpleITK) and reports lesion volume per lobe-level region, shown in the clinician panel and in patient mode. Atlases in `web/atlas/` are from FSL (MNI Structural Atlas, Harvard-Oxford subcortical) and are licensed for non-commercial use only. Accuracy is lobe level and has not been measured: ISLES has no region labels.

## Documentation & Decisions

- **[ISLES-2022 Knowledge Vault](notes/00_ISLES2022_Knowledge_Vault.md):** Medical domain context, ischemic cascade, multi-modal MRI signals, and challenge guidelines.
- **[Model Benchmark Report](Benchmark.md):** Comparative benchmark on ISLES-2022 dataset and selection of **MedNeXt** (MICCAI 2023) as core contribution.
- **[Cloud GPU Pricing & Training Guide](server_pricing.md):** GPU server rental analysis (RunPod / Vast.ai), budget estimation (~$10 USD), and remote training workflow.

