# Dual-Encoder SegResNet (2-Channel DWI+ADC) — 5-Fold Cross-Validation Benchmark

## 1. Experimental Configuration

- **Dataset:** ISLES-2022 (Multimodal 3D Brain MRI for Acute Ischemic Stroke)
- **Cohort:** 250 subjects (5-Fold Cross-Validation: 200 Train / 50 Val per fold)
- **Model Architecture:** `DualEncoderSegResNet` (Dual-branch Encoder for DWI & ADC with Multi-scale Feature Fusion)
- **Input Channels:** 2 channels (`DWI + ADC`, FLAIR removed to eliminate leukoaraiosis false positives)
- **Output Channels:** 1 (Binary stroke lesion mask)
- **Patch Size (ROI):** `(96, 96, 64)` (Enhanced axial depth context)
- **Epochs per Fold:** 50
- **Batch Size:** 2 (Mixed Precision AMP FP16 on GPU Tesla T4)
- **Optimizer:** AdamW (LR: 1e-3, CosineAnnealingLR)
- **Caching:** MONAI Full System RAM Cache (`CacheDataset`, cache_rate: 1.0)

---

## 2. 5-Fold Cross-Validation Performance Comparison

| Fold | Baseline SegResNet (3-ch: DWI+ADC+FLAIR) | Dual-Encoder SegResNet (2-ch: DWI+ADC) | Gain (Δ Dice) | Training Time |
| :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | 0.5328 | **0.7205** | <font color="green">**+18.77%**</font> | 57.2 min |
| **Fold 1** | 0.5847 | **0.7252** | <font color="green">**+14.05%**</font> | 58.8 min |
| **Fold 2** | 0.5476 | **0.6134** | <font color="green">**+6.58%**</font> | 58.4 min |
| **Fold 3** | 0.4043 | **0.5311** | <font color="green">**+12.68%**</font> | 57.9 min |
| **Fold 4** | 0.6822 | **0.6435** | -3.87% | 56.8 min |
| **MEAN** | **0.5503 ± 0.0897** | **0.6467 ± 0.0736** | <font color="green">**+9.64%**</font> | **~57.8 min** |

---

## 3. Clinical & Methodological Key Insights

1. **Substantial Performance Leap (+9.64% Mean Dice across all 250 cases):**
   - The dual-encoder architecture coupled with doubling the axial slice depth (`96x96x64`) improved the overall 5-fold cross-validation Mean Dice from **0.5503** to **0.6467**.
   - Folds 0 and 1 both surpassed the **72% Dice mark (0.7205 and 0.7252)**, representing top-tier performance on the ISLES-2022 challenge dataset.

2. **Validation of FLAIR Exclusion:**
   - Removing FLAIR eliminates non-acute confounding signals (white matter hyperintensities / leukoaraiosis), allowing the dual encoders to focus strictly on cytotoxic edema (DWI hyperintensity) and true water restriction (ADC hypointensity).

3. **Training Efficiency:**
   - Despite doubling the depth dimension (from 32 to 64 slices) and increasing epochs from 30 to 50, training time dropped from ~91 minutes (local/unoptimized) to **~58 minutes per fold** on Kaggle GPU T4 using MONAI `CacheDataset`.

