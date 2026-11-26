# DEEP-DIVE TECHNICAL REPORT: MICRO-LESION & CLINICAL STROKE SEGMENTATION
## Systematic Analysis of 10 Landmark Reference Papers vs. Dual-Encoder SegResNet V3

**Project:** PBL6 - Acute Ischemic Stroke Lesion Segmentation (ISLES 2022)  
**Author:** AI Architect & Pair Programmer (for Lê Đình Hoài Long)  
**Date:** November 2026  
**Reference Corpus:** `docs/reference_papers/` (`1.pdf` to `32.pdf`)  
**Target Architecture:** Dual-Encoder SegResNet V3 (PBL6 Baseline)

---

## 1. COMPREHENSIVE COMPARATIVE MATRIX (THE 10 PAPERS)

The following matrix provides an exhaustive, head-to-head comparison across all 10 landmark papers in the project's knowledge repository, detailing their technical formulation, handling of micro-lesions (<5 mL / punctate infarcts), and clinical benchmark results.

| # | Paper / Authors & Venue | Architecture & Backbone | Input Modalities | Patch / Input Dimensions | Loss Function Formulation | Small / Micro-Lesion Treatment (< 5 mL) | Best Reported Results |
|:---:|:---|:---|:---:|:---:|:---|:---|:---:|
| **1** | **Bjørnerud et al. (2024/2025)**<br>*Oslo Univ Hospital* | nnU-Net (Base 3D) vs. ResEnc (Residual Encoder nnU-Net) | **DWI only** vs. DWI+ADC vs. DWI+ADC+FLAIR | Dynamic nnU-Net spacing (patch ~ $128 \times 128 \times 64$) | Compound Dice + CE | **Eliminated FLAIR & BET**: Discovered that complex preprocessing & FLAIR co-registration severely degrades small lesions. DWI-only nnU-Net beat DeepISLES on small-volume cohorts ($p < 0.01$). | Nor-COAST: DSC 0.69 (DWI-only) vs 0.58 (DeepISLES) |
| **2** | **Ding et al. (2025/2026)**<br>*CVPR / DPRD* | Displacement-Preserving Relational Distillation (DPRD) | Multi-modal 3D MRI (DWI, ADC, FLAIR) | Patch-based ($96 \times 96 \times 96$) | Task Loss + $\mathcal{L}_{\text{DPRD}}$ (Batch-wise displacement matching) | **ROI-Masked Relational Pooling**: Dilution of small target features in KD is prevented by pooling activations strictly inside foreground ROIs, preserving micro-trajectories. | High cross-scanner robustness; +3.8% DSC gain on small lesions |
| **4** | **Vyncke, Ashtari, Pižurica (2024)**<br>*Ghent Univ (LARGO)* | Low-Rank Hypernetwork Adapter + U-Net | Dynamic subset of {DWI, ADC, FLAIR} | Patch-based ($128 \times 128 \times 128$) | Modality-conditioned DiceCE | **Adaptive Modality Conditioning**: When FLAIR is noisy or missing, low-rank hypernetworks modulate weights without collapsing small lesion features to zero. | Outperformed modality dropout & full zero-imputation on ISLES-22 |
| **5** | **Usman et al. (Dec 2025)**<br>*Stanford Univ* | **Dual-Encoder TransUNet** (CNN + ViT bottleneck) | **DWI + ADC** (Discarded FLAIR) | **2.5D Multi-Slice**: $128 \times 128 \times 3$ slices per modality | Binary Cross-Entropy with Logits | **Modality Separation & Slice Stacking**: Decoupled DWI/ADC encoders prevent signal cancellation. Stacking 3 adjacent slices provides z-continuity without 3D downsampling collapse. | **DSC: 85.4%** (3-slice Dual TransUNet vs 81.3% Single-Encoder) |
| **7** | **Hardan (2024/2025)**<br>*Qatar Univ (Mamba)* | Flatten Wisely 3D Mamba / State Space Model (SSM) | 3D Multimodal MRI | $64 \times 64 \times 64$ to $96 \times 96 \times 96$ | Soft Dice + Cross-Entropy | **Continuous Space-Filling Curve (Hilbert)**: Standard raster-scan flattening breaks spatial locality across Z-axis, destroying small 3D clusters. Hilbert curve preserves 3D micro-infarct continuity. | Boosted small lesion DSC by +4.2% over naive raster scanning |
| **9** | **Rahman & Chowdhury (2024)**<br>*Multi-Channel MRI* | SelfONN-CSCA-UNet (Channel & Space Attention) | **DWI + ADC + eDWI** (Enhanced DWI) | $128 \times 128 \times 64$ | **Equalized Dice + Jaccard Loss** ($\mathcal{L} = \mathcal{L}_{\text{Dice}} + \mathcal{L}_{\text{Jaccard}}$) | **Dual Overlap Constraint**: Dice provides high sensitivity (catches faint infarcts); Jaccard strictly penalizes false positives, tightening borders around punctate foci. | **DSC: 87.49%**, IoU: 79.11%, Precision: 94.87% |
| **10** | **De la Rosa et al. (2024/2025)**<br>*Nature Comms (DeepISLES)* | Consensus Ensemble: SEALS (nnU-Net) + NVAUTO (SegResNet) + SWAN (Factorizer) | DWI + ADC + FLAIR (with HD-BET) | Heterogeneous (native challenge models) | Consensus Voting (Agreement $\ge 2 / 3$) | **Subgroup Stratification (<5 mL, 5-20 mL, >20 mL)**: Proved all models drop DSC to ~55-65% on <5 mL, but Lesion F1 remains ~0.78. Majority voting eliminates solitary model false alarms. | Test Set DSC: 0.824, Lesion F1: 0.812; High correlation ($r=0.87$) on <5 mL |
| **22** | **Siddiquee et al. (2022)**<br>*NVIDIA (NVAUTO - Rank 1/2)* | **MONAI SegResNet** (`init_filters: 32`, 5 stages [2,4,4,4,4]) | **DWI + ADC** (Strictly excluded FLAIR) | **Massive Patch**: $192 \times 192 \times 128$ voxels (batch 8 on 8x V100) | **Deep Supervision Compound DiceFocal**: $\sum_{i=0}^4 2^{-i} \mathcal{L}_{\text{DiceFocal}}$ | **Global Brain Context Capture**: 192x192x128 patch sees almost the entire brain, providing anatomical symmetry priors to disambiguate true micro-infarcts from artifact noise. | **5-Fold CV DSC: 0.8086**; Hidden Test DSC: 0.824, Lesion F1: 0.800 |
| **24** | **Hernandez Petzsche et al. (2022)**<br>*Nature Scientific Data* | ISLES 2022 Dataset Official Reference Paper | Multi-center DWI, ADC, FLAIR | Full 3D NIfTI ($1 \times 1 \times 1$ mm reference space) | N/A (Data Descriptor & Benchmark) | **Micro-Lesion Prevalence**: Documented that **>50% of all cases have volume < 5.3 mL**, and 25% have volume < 1.3 mL. Punctate emboli are the clinical norm, not outliers! | Established 4 official metrics: DSC, AVD, LCD, Lesion F1 |
| **32** | **Ashtari et al. (2023/2024)**<br>*MedIA (Factorizer - Rank 3)* | Swin Factorizer (Non-negative Matrix Factorization) | DWI + ADC + FLAIR | $128 \times 128 \times 128$ isotropic patches | Soft Dice + Cross-Entropy | **Low-Rank Spatial Context Modeling**: Replaces dense self-attention with NMF blocks, capturing global multi-focal dependencies without blurring high-frequency micro-boundaries. | ISLES-22 Challenge Test DSC: 0.812, Lesion F1: 0.800, AVD: 1.936 mL |

---

## 2. DEEP DIVE: HOW THE 10 PAPERS SOLVED THE 6 CORE MILESTONES

### MILESTONE 1: Modality Selection & Fusion Mechanics (DWI vs. ADC vs. FLAIR)
- **The FLAIR Dilemma (Papers 1, 10, 22):**
  - In theory, FLAIR identifies chronic white matter hyperintensities (leukoaraiosis) and old strokes.
  - In practice, **NVIDIA (Paper 22)**, **Usman et al. (Paper 5)**, and **Bjørnerud et al. (Paper 1)** demonstrated that including FLAIR harms acute lesion segmentation unless exceptionally aligned. FLAIR is acquired at different slice thicknesses (often $5-6$ mm 2D thick slices vs. $1-2$ mm isotropic DWI), causing severe interpolation artifacts and misregistration.
  - **Bjørnerud et al. (Paper 1)** conducted pairwise statistical tests on $n=1,744$ scans and proved that **DWI-only or DWI+ADC strictly outperformed DWI+ADC+FLAIR** on small stroke cohorts ($p < 0.01$). Adding FLAIR caused the brain-extraction tool (HD-BET) to occasionally shave off cortex-adjacent micro-infarcts.
- **Why Dual-Encoder Decoupling Works (Paper 5 - Usman et al.):**
  - Acute ischemia exhibits opposite signal behavior: **High DWI (hyperintense) + Low ADC (hypointense)** due to cytotoxic edema and restricted water diffusion.
  - When early concatenation is applied ($[C_{in}=2, D, H, W]$ through a single convolution layer), shared kernel weights simultaneously multiply bright DWI pixels ($+2.5$) and dark ADC pixels ($-2.0$). At the boundaries of tiny lesions (2–5 voxels wide), these signals partially cancel each other out, blunting boundary edge gradients.
  - Paper 5 demonstrated that processing DWI and ADC through independent encoder feature extractors before bottleneck fusion lifted Test Dice from **81.3% to 83.1%** (and **85.4%** with multi-slice context).
- **The Role of Synthetic Channels (Paper 9 - Rahman et al.):**
  - Rather than raw FLAIR, Paper 9 computed **enhanced DWI (eDWI)** as an explicit mathematical transformation ($eDWI = DWI \times \exp(-b \times ADC)$), supplying a synthetic 3rd channel that amplifies true restricted diffusion while completely suppressing T2 shine-through artifacts.

---

### MILESTONE 2: Network Architecture, Downsampling Bottlenecks & Feature Vanishing
- **The Downsampling Crisis:**
  - In a standard 5-stage U-Net or SegResNet, the input patch undergoes four $2\times$ downsampling operations: $96 \rightarrow 48 \rightarrow 24 \rightarrow 12 \rightarrow 6$.
  - A small punctate lesion measuring $3 \times 3 \times 3$ voxels (total 27 voxels, $\approx 0.027$ mL) is reduced to $\frac{3}{16} \approx 0.18$ voxels by Stage 4. **At the bottleneck, the lesion mathematically ceases to exist.**
  - If the decoder relies only on bottleneck latent features, small lesions are permanently lost.
- **Solution A: Deep Supervision with Geometric Decay (Paper 22 - NVIDIA NVAUTO):**
  - NVIDIA placed $1 \times 1 \times 1$ conv projection heads at the lowest 3 decoder sublevels ($D/4, D/2, D$).
  - Crucially, they weighted sublevel losses by power-of-two decay:
    $$\mathcal{L}_{\text{total}} = \sum_{i=0}^{4} \frac{1}{2^i} \mathcal{L}_i(\text{pred}_i, \text{target}_{\downarrow i})$$
    This ensures that high-resolution stages ($i=0, 1$) dominate gradient updates, forcing early skip connections to retain high-frequency spatial details rather than allowing the network to rely solely on the low-resolution bottleneck.
- **Solution B: 2.5D Slice-Contextual Transformers (Paper 5 - Usman et al.):**
  - Usman et al. avoided 3D Z-axis downsampling entirely by adopting a 2.5D multi-slice formulation. By inputting 3 consecutive axial slices ($128 \times 128 \times 3$), in-plane spatial resolution ($128 \times 128$) is preserved with minimal downsampling, preventing thin lacunar infarcts from being smeared across axial planes.
- **Solution C: Flatten Wisely via Space-Filling Curves (Paper 7 - Hardan):**
  - When utilizing sequence models (Transformers or Mamba SSMs), 3D volumes are flattened into 1D token sequences.
  - Standard raster-scanning causes two voxels that are physically 1 voxel apart in the Z direction to be separated by $H \times W$ tokens (e.g., $96 \times 96 = 9,216$ tokens).
  - Hardan proved that using **Hilbert space-filling curves** keeps 3D neighborhood voxels contiguous in 1D sequence memory, preventing micro-lesion signals from decaying across long context spans.

---

### MILESTONE 3: Loss Function Engineering under Extreme Voxel Imbalance
- **The Background Dominance Paradox:**
  - In ISLES 2022, a $96 \times 96 \times 64$ patch contains $589,824$ voxels. For a 30-voxel lesion, the background-to-foreground ratio is **$19,660 : 1$**.
  - Standard Cross-Entropy produces overwhelming background gradients.
  - Soft Dice Loss ($\mathcal{L}_{\text{Dice}} = 1 - \frac{2 |P \cap G| + \epsilon}{|P| + |G| + \epsilon}$) is volume-invariant, but when $|G|$ is tiny ($\approx 20$ voxels), a false positive of only 20 background voxels halves the Dice score ($0.50$), while predicting 0 voxels yields a massive, unstable gradient cliff.
- **Compound Dice + Focal Loss (Paper 22 & PBL6 V3):**
  - Focal Loss ($\mathcal{L}_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$) with $\gamma = 2.0$ dynamically suppresses well-classified background voxels ($p_t \approx 1.0$), ensuring that gradients originate almost entirely from ambiguous boundary voxels.
- **Equalized Dice + Jaccard Loss (Paper 9 - Rahman et al.):**
  - Rahman et al. uncovered a complementary dynamic between Dice and Jaccard:
    $$\mathcal{L}_{\text{combined}} = \mathcal{L}_{\text{Dice}} + \mathcal{L}_{\text{Jaccard}} = \left(1 - \frac{2 |P \cap G|}{|P| + |G|}\right) + \left(1 - \frac{|P \cap G|}{|P| + |G| - |P \cap G|}\right)$$
  - Dice Loss alone has higher sensitivity but exhibits "blooming" (over-segmenting borders into normal tissue, driving up false positives).
  - Jaccard Loss has a steeper denominator penalty, punishing over-segmentation. Combining both with equal weights produced superior boundary sharpness and a precision of **94.87%**.
- **Relational Distillation on Small Targets (Paper 2 - Ding et al.):**
  - In model distillation or transfer learning, naive feature matching fails because background noise drowns out small targets.
  - Ding et al. introduced **ROI-masked feature pooling**, calculating relational displacement vectors exclusively on foreground-intersecting feature tokens, demonstrating that targeted loss constraints are essential for small target convergence.

---

### MILESTONE 4: Spatial Sampling & Crop Strategy (The Volume Trap)
- **The Critical Flaw of `RandCropByPosNegLabeld`:**
  - MONAI's `RandCropByPosNegLabeld(pos=1, neg=1)` selects a foreground voxel with probability $0.5$, then centers a crop around that voxel.
  - **Mathematical Bias:** If a patient has a large core lesion of 50,000 voxels and a micro-satellite of 25 voxels, a uniformly sampled foreground voxel will fall into the large lesion **99.95% of the time**. The micro-satellite will be selected in only **0.05% of training crops**.
  - As a result, the model is repeatedly trained on large core textures and virtually never experiences the isolated contextual appearance of punctate satellites.
- **NVIDIA's Brute-Force Receptive Field (Paper 22):**
  - NVIDIA solved this by setting the patch size to **$192 \times 192 \times 128$ voxels**.
  - Since the average skull-stripped brain volume fits inside $192 \times 192 \times 128$ at 1 mm resolution, their "patch" encompassed virtually the **entire brain in every single crop**.
  - This eliminated sampling bias completely: both large cores and micro-satellites were present in the exact same training forward pass, allowing the network to learn contralateral hemisphere symmetry.
  - *Hardware Cost:* Required 8x NVIDIA V100 (16GB) GPUs.
- **Ashtari's Shifted Window Strategy (Paper 32 - Factorizer):**
  - Swin Factorizer used $128 \times 128 \times 128$ isotropic patches with shifted window matrix factorization ($8 \times 8 \times 8$ sub-patches shifted by 4 voxels).
  - By maintaining a large $128^3$ window, small lesions are never cut off by crop edges without seeing adjacent cerebral context.

---

### MILESTONE 5: Post-Processing, Thresholding & Island Pruning
- **The Hazard of Rigid Island Pruning (`min_size_voxels`):**
  - In PBL6 V3 evaluation, we applied `remove_small_lesion_islands(pred, min_size_voxels=15)`.
  - **Clinical Pathology Reality (Paper 24 - Hernandez Petzsche et al.):**
    In ISLES-2022, micro-embolic showers frequently manifest as true ischemic foci measuring only **8 to 15 voxels** in volume.
    A static threshold of 15 voxels permanently deletes genuine small infarcts, immediately dropping the patient's Lesion Detection F1 score to **0.0000** (as observed in Case 06 of our Fold 0 evaluation).
  - Furthermore, if an image has voxel spacing $1 \times 1 \times 1$ mm, 15 voxels = 15 $\text{mm}^3$ (0.015 mL). But if resampled to anisotropic spacing (e.g., $2 \times 2 \times 4$ mm), 15 voxels = 240 $\text{mm}^3$ (0.24 mL), which accidentally deletes clinically significant infarcts!
  - **Rule:** Island filtering must be parameterized by physical volume in milliliters ($\text{min\_size\_mL} \le 0.01$ mL), never raw voxel counts.
- **Consensus Ensembling as an Optimal Noise Filter (Paper 10 - DeepISLES):**
  - De la Rosa et al. demonstrated that single models (even SOTA like nnU-Net or SegResNet) occasionally fire false-positive clusters in the sulci or ventricles due to diffusion artifacts.
  - Instead of aggressive morphological pruning, DeepISLES implemented a **2-out-of-3 model agreement rule**:
    $$\hat{Y} = \mathbb{I}\left( \hat{Y}_{\text{SEALS}} + \hat{Y}_{\text{NVAUTO}} + \hat{Y}_{\text{SWAN}} \ge 2 \right)$$
  - Because distinct architectural families (Pure CNN vs. ResNet with Deep Supervision vs. Low-rank Factorizer) make uncorrelated false-positive errors, majority voting naturally annihilates solitary false alarms while retaining real micro-infarcts where at least two models concur.

---

### MILESTONE 6: Clinical Validation & Subgroup Stratification
- **The Ground Truth Distribution (Paper 24 - Nature Scientific Data):**
  - Table 2 of Hernandez Petzsche et al. establishes that:
    - **Median infarct volume across ISLES-2022 is only 5.3 mL**.
    - **25% of cases have volume < 1.3 mL**.
    - Over 40% of cases exhibit multiple unconnected lesion islands ($N \ge 3$).
- **The Metric Disconnect (Paper 10 & PBL6 V3 findings):**
  - In small lesions (< 5 mL), standard Dice drops significantly (mean DSC ~0.55–0.65) even in top-ranked algorithms (NVAUTO, SEALS, DeepISLES).
  - This is an intrinsic mathematical property of the Dice coefficient: a displacement of 1 voxel on an 8-voxel lesion cuts Dice by 30%, whereas a 1-voxel error on a 50,000-voxel lesion alters Dice by <0.01%.
  - **Clinical Imperative:** An algorithm for acute stroke must be evaluated by **Lesion Detection F1** and **Absolute Volume Difference (AVD)** in conjunction with Dice. A model that achieves 0.60 Dice on a 2 mL lesion but scores 1.00 Lesion F1 and 0.2 mL AVD is clinically outstanding because it accurately alerts the neurologist to the presence and volume of the stroke core.

---

## 3. GAP ANALYSIS: DUAL-ENCODER SEGRESNET V3 VS. SOTA LITERATURE

| Dimension | Current PBL6 V3 Implementation | SOTA Literature Consensus (Papers 5, 9, 10, 22) | Critical Gap & Impact |
|:---|:---|:---|:---|
| **Input Modalities** | 3-channel input: DWI, ADC, FLAIR concatenated in dataset, then split into DWI/ADC encoders (FLAIR unused). | **DWI + ADC only** (NVAUTO, Usman) OR **DWI + ADC + eDWI** (Rahman). Discard raw FLAIR. | Carrying unused FLAIR in the data pipeline wastes RAM/I/O. eDWI could boost subtle contrast. |
| **Encoder Architecture** | Dual independent SegResNet encoders (`init_filters: 32`), BiCrossAttention fusion at bottleneck. | Dual-encoder (Usman) or Single high-capacity SegResNet (NVAUTO) with multi-scale skip connections. | Bottleneck-only fusion misses early spatial cue exchange between DWI and ADC. |
| **Deep Supervision** | Disabled (`deep_supervision: False`, decoder heads unweighted). | **Enabled across 3 sublevels** with power-of-two decay $\sum 2^{-i} \mathcal{L}_i$ (NVAUTO). | High-resolution decoder layers receive weak direct gradients for small lesions. |
| **Patch Size & Sampling** | $96 \times 96 \times 64$ patch with standard `RandCropByPosNegLabeld(pos=1, neg=1)`. | **$192 \times 192 \times 128$** (NVAUTO) OR **2.5D Multi-slice** (Usman) OR **Lesion-centric cluster oversampling**. | Extreme sampling bias: micro-satellites receive <0.1% of training crops. |
| **Loss Function** | MONAI `DiceFocalLoss(gamma=2.0)`. | **Compound Dice + Focal with deep supervision** (NVAUTO) OR **Equalized Dice + Jaccard** (Rahman). | Unweighted focal loss is still partially dominated by $589,000$ background voxels. |
| **Post-Processing** | Static `remove_small_lesion_islands(min_size_voxels=15)`. | **Ensemble voting** (DeepISLES) OR **Physical volume thresholding ($\le 0.01$ mL)**. | Wipes out true small infarcts of 8–14 voxels, resulting in zero-F1 failure cases. |

---

## 4. CONCRETE ACTIONABLE ROADMAP FOR LONG (V4 RESEARCH & CODE MODIFICATIONS)

Based directly on the mechanisms proven in the 10 reference papers, here are the exact, prioritized engineering interventions that Long can implement without introducing speculative architectural junk:

### Action 1: Eliminate Sampling Bias in `src/dataset/transforms.py`
* **The Problem:** Uniform foreground sampling ignores micro-satellites in favor of the massive main core.
* **The Solution (Connected Component-Aware Sampling):**
  Instead of sampling uniform foreground voxels, label connected components in the mask using `scipy.ndimage.label`. Randomly select a **component label** with equal probability, then select a voxel from *that specific component*.
  * **Result:** A 15-voxel satellite and a 50,000-voxel core receive equal 50/50 chances of being the crop center, guaranteeing that the model learns micro-infarct features.

### Action 2: Activate Multi-Level Deep Supervision in `src/models/dual_segresnet.py`
* **The Problem:** 4 stages of downsampling destroy small lesions before the bottleneck.
* **The Solution (NVIDIA Auto3DSeg Formulation):**
  Add $1 \times 1 \times 1$ conv heads at Decoder Stages 1, 2, and 3 ($D/4, D/2, D$).
  Calculate loss at each stage downsampled by nearest-neighbor interpolation, weighted by:
  $$\mathcal{L} = \mathcal{L}_{\text{full}} + 0.5 \cdot \mathcal{L}_{\text{half}} + 0.25 \cdot \mathcal{L}_{\text{quarter}}$$
  * **Result:** Gradients from small lesions are directly injected into high-resolution decoder blocks without depending on bottleneck reconstruction.

### Action 3: Upgrade Loss Formulation to Dice + Jaccard or Focal Tversky in `src/losses/`
* **The Problem:** Standard DiceFocal is symmetric; false negatives on small lesions are not punished sufficiently.
* **Option A (Rahman et al. - Paper 9):** Implement equalized `DiceLoss + JaccardLoss`. Jaccard provides the tight boundary constraint that stops false positives from blooming.
* **Option B (Focal Tversky Loss):** Set $\beta = 0.7$ (penalizing False Negatives $2.3\times$ heavier than False Positives) and $\gamma = 1.33$. This explicitly forces the model to detect faint, low-contrast ischemic spots.

### Action 4: Calibrate Post-Processing Threshold in Physical Space (`src/utils/postprocess.py`)
* **The Problem:** `min_size_voxels: 15` deletes small infarcts.
* **The Solution:**
  Compute physical voxel volume: $V_{\text{voxel}} = \Delta x \times \Delta y \times \Delta z$ (in $\text{mm}^3$).
  Calculate cutoff:
  $$\text{min\_voxels} = \max\left(3, \left\lfloor \frac{0.008 \times 1000}{V_{\text{voxel}}} \right\rfloor\right)$$
  Filter out only islands smaller than $0.008$ mL ($\approx 8 \text{ mm}^3$). Any island with $\ge 8 \text{ mm}^3$ having mean predicted confidence $>0.65$ must be retained.

### Action 5: Multi-Fold / Multi-Model Out-Of-Fold Consensus Ensembling
* **The Solution (DeepISLES - Paper 10):**
  Since all 5 folds of Dual-Encoder SegResNet V3 are now fully trained and stored locally, evaluate an **Ensemble Model** taking the average or median probability map across all 5 fold checkpoints for validation inference.
  This immediately suppresses fold-specific boundary noise and mimics the top challenge winning pipelines.

