# ISLES-2022 Evaluation Report: dual_segresnet_single_fold0

**Evaluation Date:** 2026-11-06 11:01:30  
**Total Cases Evaluated:** 50  

---

## 1. Overall ISLES-2022 Benchmark Suite (Mean ± Std [Median])

| Metric | Mean ± Std | Median | Unit / Target |
| :--- | :--- | :--- | :--- |
| **Dice Similarity Coefficient (DSC)** | **0.6798 ± 0.2059** | **0.7361** | Higher is better (Max 1.0) |
| **Absolute Volume Difference (AVD)** | **2.86 ± 6.63** | **0.73** | mL (Lower is better) |
| **Lesion Count Difference (LCD)** | **7.48 ± 14.19** | **4.00** | Count (Lower is better) |
| **Lesion Detection F1 Score** | **0.4482 ± 0.2065** | **0.4410** | Higher is better (Max 1.0) |

---

## 2. Lesion Volume Subgroup Stratification

| Subgroup Category | Case Count | Mean DSC | Mean AVD | Mean LCD | Mean Lesion F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Small (< 5 mL) | 26 | 0.5843 | 0.99 mL | 4.77 | 0.4256 |
| Medium (5 - 30 mL) | 17 | 0.7449 | 2.37 mL | 12.94 | 0.4589 |
| Large (> 30 mL) | 7 | 0.8764 | 11.02 mL | 4.29 | 0.5062 |

---

## 3. Case-by-case Distribution
- Top 5 Best Cases (by DSC):
| case_id | dsc | avd_ml | lcd | lesion_f1 |
| :---: | :---: | :---: | :---: | :---: |
| single_fold0_case44 | 0.9183 | 3.8510 | 3 | 0.6667 |
| single_fold0_case12 | 0.9084 | 8.2920 | 2 | 0.6667 |
| single_fold0_case04 | 0.9022 | 3.6290 | 1 | 0.6452 |
| single_fold0_case21 | 0.8922 | 0.4670 | 4 | 0.6000 |
| single_fold0_case28 | 0.8914 | 3.3870 | 8 | 0.5000 |

- Top 5 Most Challenging Cases (by DSC):
| case_id | dsc | avd_ml | lcd | lesion_f1 |
| :---: | :---: | :---: | :---: | :---: |
| single_fold0_case25 | 0.0368 | 10.5150 | 8 | 0.2000 |
| single_fold0_case22 | 0.0983 | 1.7340 | 3 | 0.4000 |
| single_fold0_case48 | 0.1919 | 0.5190 | 3 | 0.0000 |
| single_fold0_case01 | 0.3955 | 1.6580 | 4 | 0.1429 |
| single_fold0_case38 | 0.4077 | 10.3260 | 11 | 0.2105 |
