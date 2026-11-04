# ISLES-2022 Evaluation Report: dual_segresnet_single_fold0

**Evaluation Date:** 2026-11-04 07:31:13  
**Total Cases Evaluated:** 50  

---

## 1. Overall ISLES-2022 Benchmark Suite (Mean ± Std [Median])

| Metric | Mean ± Std | Median | Unit / Target |
| :--- | :--- | :--- | :--- |
| **Dice Similarity Coefficient (DSC)** | **0.5866 ± 0.2392** | **0.6336** | Higher is better (Max 1.0) |
| **Absolute Volume Difference (AVD)** | **13.99 ± 70.67** | **1.09** | mL (Lower is better) |
| **Lesion Count Difference (LCD)** | **12.42 ± 17.30** | **8.00** | Count (Lower is better) |
| **Lesion Detection F1 Score** | **0.3147 ± 0.1949** | **0.2857** | Higher is better (Max 1.0) |

---

## 2. Lesion Volume Subgroup Stratification

| Subgroup Category | Case Count | Mean DSC | Mean AVD | Mean LCD | Mean Lesion F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Small (< 5 mL) | 26 | 0.4483 | 21.11 mL | 11.77 | 0.2784 |
| Medium (5 - 30 mL) | 17 | 0.6887 | 3.07 mL | 17.12 | 0.3368 |
| Large (> 30 mL) | 7 | 0.8524 | 14.06 mL | 3.43 | 0.3959 |

---

## 3. Case-by-case Distribution
- Top 5 Best Cases (by DSC):
| case_id | dsc | avd_ml | lcd | lesion_f1 |
| :---: | :---: | :---: | :---: | :---: |
| single_fold0_case12 | 0.9017 | 2.2140 | 1 | 0.5333 |
| single_fold0_case04 | 0.8852 | 5.1550 | 0 | 0.5625 |
| single_fold0_case21 | 0.8758 | 0.2210 | 8 | 0.4286 |
| single_fold0_case28 | 0.8691 | 8.9620 | 8 | 0.5417 |
| single_fold0_case44 | 0.8523 | 25.7750 | 6 | 0.4444 |

- Top 5 Most Challenging Cases (by DSC):
| case_id | dsc | avd_ml | lcd | lesion_f1 |
| :---: | :---: | :---: | :---: | :---: |
| single_fold0_case25 | 0.0004 | 500.7440 | 75 | 0.0000 |
| single_fold0_case01 | 0.0208 | 22.6480 | 35 | 0.0000 |
| single_fold0_case48 | 0.1077 | 0.5210 | 4 | 0.0000 |
| single_fold0_case14 | 0.1359 | 2.7110 | 8 | 0.1429 |
| single_fold0_case03 | 0.1628 | 6.1460 | 9 | 0.1818 |
