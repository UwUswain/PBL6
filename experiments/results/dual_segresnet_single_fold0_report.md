# ISLES-2022 Evaluation Report: dual_segresnet_single_fold0

**Evaluation Date:** 2026-11-03 23:37:06  
**Total Cases Evaluated:** 50  

---

## 1. Overall ISLES-2022 Benchmark Suite (Mean ± Std [Median])

| Metric | Mean ± Std | Median | Unit / Target |
| :--- | :--- | :--- | :--- |
| **Dice Similarity Coefficient (DSC)** | **0.5231 ± 0.2661** | **0.5800** | Higher is better (Max 1.0) |
| **Absolute Volume Difference (AVD)** | **18.93 ± 101.99** | **1.57** | mL (Lower is better) |
| **Lesion Count Difference (LCD)** | **19.98 ± 14.46** | **18.50** | Count (Lower is better) |
| **Lesion Detection F1 Score** | **0.2203 ± 0.1687** | **0.1984** | Higher is better (Max 1.0) |

---

## 2. Lesion Volume Subgroup Stratification

| Subgroup Category | Case Count | Mean DSC | Mean AVD | Mean LCD | Mean Lesion F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Small (< 5 mL) | 26 | 0.3561 | 28.99 mL | 23.38 | 0.1570 |
| Medium (5 - 30 mL) | 17 | 0.6641 | 2.92 mL | 21.06 | 0.2563 |
| Large (> 30 mL) | 7 | 0.8006 | 20.45 mL | 4.71 | 0.3681 |

---

## 3. Case-by-case Distribution
- Top 5 Best Cases (by DSC):
| case_id | dsc | avd_ml | lcd | lesion_f1 |
| :---: | :---: | :---: | :---: | :---: |
| single_fold0_case04 | 0.8916 | 1.8480 | 2 | 0.6667 |
| single_fold0_case49 | 0.8581 | 0.9570 | 1 | 0.3158 |
| single_fold0_case21 | 0.8551 | 0.0580 | 23 | 0.2069 |
| single_fold0_case12 | 0.8549 | 12.4130 | 3 | 0.3158 |
| single_fold0_case30 | 0.8012 | 0.2960 | 14 | 0.2000 |

- Top 5 Most Challenging Cases (by DSC):
| case_id | dsc | avd_ml | lcd | lesion_f1 |
| :---: | :---: | :---: | :---: | :---: |
| single_fold0_case25 | 0.0007 | 722.8280 | 28 | 0.0000 |
| single_fold0_case01 | 0.0259 | 1.3000 | 37 | 0.0000 |
| single_fold0_case48 | 0.0354 | 0.3150 | 14 | 0.0000 |
| single_fold0_case23 | 0.0658 | 3.2550 | 35 | 0.0541 |
| single_fold0_case17 | 0.0865 | 2.3590 | 21 | 0.1600 |
