"""Visualizes qualitative comparison between Original DWI, CLAHE-Enhanced DWI (eDWI), ADC, and Ground Truth Mask."""

import argparse
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from src.dataset.parser import parse_isles_dataset


def apply_slice_clahe(
    volume: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8),
) -> np.ndarray:
    """Applies slice-wise CLAHE along axial dimension."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = np.zeros_like(volume, dtype=np.float32)

    # volume shape: (H, W, D) or (D, H, W). We handle axial as last or first axis.
    # Typically in NIfTI: shape is (X, Y, Z) where Z is axial slices
    num_slices = volume.shape[-1]
    for z in range(num_slices):
        sl = volume[:, :, z]
        sl_min, sl_max = float(sl.min()), float(sl.max())
        if (sl_max - sl_min) > 1e-5 and np.count_nonzero(sl > sl_min) > 20:
            sl_u8 = ((sl - sl_min) / (sl_max - sl_min) * 255.0).clip(0, 255).astype(np.uint8)
            enhanced[:, :, z] = clahe.apply(sl_u8).astype(np.float32) / 255.0
        else:
            enhanced[:, :, z] = 0.0

    return enhanced


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize DWI vs eDWI (CLAHE) comparison")
    parser.add_argument("--data-dir", type=str, default="data/raw/ISLES-2022")
    parser.add_argument("--output-file", type=str, default="experiments/results/edwi_comparison_demo.png")
    parser.add_argument("--case-idx", type=int, default=0, help="Case index to visualize")
    args = parser.parse_args()

    cases = parse_isles_dataset(args.data_dir, require_mask=True)
    if not cases:
        print(f"Error: No cases found in {args.data_dir}")
        return

    # Pick a case with substantial lesion voxels
    selected_case = None
    for c in cases[args.case_idx:]:
        mask_data = nib.load(c["mask"]).get_fdata()
        if mask_data.sum() > 300:
            selected_case = c
            break

    if selected_case is None:
        selected_case = cases[0]
        mask_data = nib.load(selected_case["mask"]).get_fdata()

    case_id = selected_case["case_id"]
    print(f"Visualizing Case: {case_id}")

    dwi_nii = nib.load(selected_case["dwi"])
    dwi_data = dwi_nii.get_fdata().astype(np.float32)

    adc_nii = nib.load(selected_case["adc"])
    adc_data = adc_nii.get_fdata().astype(np.float32)

    # Synthesize eDWI via CLAHE
    edwi_data = apply_slice_clahe(dwi_data, clip_limit=2.0, tile_grid_size=(8, 8))

    # Find the axial slice with the largest lesion area
    slice_lesion_counts = [mask_data[:, :, z].sum() for z in range(mask_data.shape[-1])]
    best_z = int(np.argmax(slice_lesion_counts))
    print(f"Optimal Axial Slice index: {best_z} (Lesion voxels on slice: {slice_lesion_counts[best_z]})")

    # Extract 2D slices
    dwi_slice = np.rot90(dwi_data[:, :, best_z])
    edwi_slice = np.rot90(edwi_data[:, :, best_z])
    adc_slice = np.rot90(adc_data[:, :, best_z])
    mask_slice = np.rot90(mask_data[:, :, best_z])

    # Plot 1x4 comparison figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="#1e1e1e")
    fig.suptitle(
        f"ISLES-2022 Feature Engineering: DWI vs eDWI (CLAHE) [{case_id} - Slice {best_z}]",
        fontsize=16,
        fontweight="bold",
        color="white",
        y=1.02,
    )

    titles = [
        "1. Original DWI (b=1000)",
        "2. eDWI (CLAHE Enhanced, Clip=2.0)",
        "3. ADC Map (Diffusion Coeff)",
        "4. eDWI + Lesion Ground Truth",
    ]

    # Normalize slices for clean display
    def norm_sl(s):
        s_min, s_max = s.min(), s.max()
        return (s - s_min) / (s_max - s_min + 1e-6)

    dwi_disp = norm_sl(dwi_slice)
    edwi_disp = norm_sl(edwi_slice)
    adc_disp = norm_sl(adc_slice)

    # 1. Original DWI
    axes[0].imshow(dwi_disp, cmap="bone")
    axes[0].set_title(titles[0], color="white", fontsize=12, pad=10)
    axes[0].axis("off")

    # 2. eDWI
    axes[1].imshow(edwi_disp, cmap="bone")
    axes[1].set_title(titles[1], color="#00ffcc", fontsize=12, pad=10, fontweight="bold")
    axes[1].axis("off")

    # 3. ADC
    axes[2].imshow(adc_disp, cmap="bone")
    axes[2].set_title(titles[2], color="white", fontsize=12, pad=10)
    axes[2].axis("off")

    # 4. eDWI + Mask overlay
    axes[3].imshow(edwi_disp, cmap="bone")
    masked_overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
    axes[3].imshow(masked_overlay, cmap="autumn", alpha=0.6)
    axes[3].set_title(titles[3], color="#ff4444", fontsize=12, pad=10, fontweight="bold")
    axes[3].axis("off")

    plt.tight_layout()
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Comparison image successfully saved at: {output_path}")


if __name__ == "__main__":
    main()
