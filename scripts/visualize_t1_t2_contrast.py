"""MRI Signal Intensity Simulation (T1W vs T2W).

Simulates a 2D cross-sectional brain phantom with distinct tissue compartments:
- Cerebrospinal Fluid (CSF)
- Gray Matter (GM)
- White Matter (WM)
- Muscle
- Subcutaneous Fat
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def generate_brain_phantom_masks(
    height: int = 300, width: int = 300
) -> Dict[str, np.ndarray]:
    """Generates geometric masks simulating brain tissues on a 2D grid.

    Args:
        height: Height of the 2D grid.
        width: Width of the 2D grid.

    Returns:
        Dictionary mapping tissue names to boolean mask arrays.
    """
    y, x = np.ogrid[:height, :width]
    cy, cx = height / 2.0, width / 2.0

    # Distance maps for concentric anatomical structures
    dist_head = ((y - cy) / 120.0) ** 2 + ((x - cx) / 100.0) ** 2
    dist_muscle = ((y - cy) / 112.0) ** 2 + ((x - cx) / 92.0) ** 2
    dist_csf_rim = ((y - cy) / 104.0) ** 2 + ((x - cx) / 84.0) ** 2
    dist_cortex = ((y - cy) / 96.0) ** 2 + ((x - cx) / 76.0) ** 2
    dist_wm = ((y - cy) / 74.0) ** 2 + ((x - cx) / 54.0) ** 2

    # Lateral ventricles (CSF located in deep brain)
    ventricle_left = ((y - (cy - 10)) / 28.0) ** 2 + (
        (x - (cx - 20)) / 10.0
    ) ** 2
    ventricle_right = ((y - (cy - 10)) / 28.0) ** 2 + (
        (x - (cx + 20)) / 10.0
    ) ** 2

    # Tissue masks (ordered from outer layer to inner core)
    fat_mask = (dist_head <= 1.0) & (dist_muscle > 1.0)
    muscle_mask = (dist_muscle <= 1.0) & (dist_csf_rim > 1.0)
    subarachnoid_csf = (dist_csf_rim <= 1.0) & (dist_cortex > 1.0)
    ventricles_csf = (ventricle_left <= 1.0) | (ventricle_right <= 1.0)
    csf_mask = subarachnoid_csf | ventricles_csf

    wm_mask = (dist_wm <= 1.0) & ~ventricles_csf
    gm_mask = (dist_cortex <= 1.0) & (dist_wm > 1.0) & ~ventricles_csf

    return {
        "fat": fat_mask,
        "muscle": muscle_mask,
        "csf": csf_mask,
        "gm": gm_mask,
        "wm": wm_mask,
    }


def synthesize_mri_sequence(
    masks: Dict[str, np.ndarray],
    intensities: Dict[str, int],
    shape: Tuple[int, int] = (300, 300),
    noise_std: float = 2.5,
) -> np.ndarray:
    """Combines tissue masks and assigned grayscale intensities into an image.

    Args:
        masks: Dictionary of tissue boolean masks.
        intensities: Grayscale values [0, 255] for each tissue.
        shape: Output image dimensions (H, W).
        noise_std: Standard deviation of additive Gaussian noise.

    Returns:
        Synthesized 2D MRI image as uint8 numpy array.
    """
    image = np.zeros(shape, dtype=np.float32)
    for tissue, mask in masks.items():
        image[mask] = intensities[tissue]

    # Add Gaussian noise simulating acquisition noise
    if noise_std > 0:
        noise = np.random.normal(0.0, noise_std, shape)
        image = np.clip(image + noise, 0, 255)

    return image.astype(np.uint8)


def plot_comparison(
    t1_image: np.ndarray,
    t2_image: np.ndarray,
    save_path: Path | None = None,
    show_plot: bool = True,
) -> None:
    """Renders and optionally saves side-by-side comparison of T1W and T2W.

    Args:
        t1_image: Synthesized T1W image.
        t2_image: Synthesized T2W image.
        save_path: Optional output path to save PNG figure.
        show_plot: Whether to display interactive matplotlib window.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor="#121212")

    # Display T1W
    axes[0].imshow(t1_image, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(
        "T1-Weighted (T1W) - Anatomical\n"
        "[CSF: Dark | WM > GM | Fat: Bright]",
        color="white",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    axes[0].axis("off")

    # Display T2W
    axes[1].imshow(t2_image, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(
        "T2-Weighted (T2W) - Pathological / Fluid\n"
        "[CSF: Bright | GM > WM | Fat: Bright]",
        color="white",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    axes[1].axis("off")

    # Annotate key anatomical landmarks for intuitive visual comparison
    for ax in axes:
        # Ventricles (CSF)
        ax.annotate(
            "Ventricles (CSF)",
            xy=(130, 140),
            xytext=(30, 180),
            color="#00E5FF",
            fontsize=10,
            fontweight="bold",
            arrowprops=dict(facecolor="#00E5FF", shrink=0.08, width=1.5, headwidth=6),
        )
        # Cortex (GM)
        ax.annotate(
            "Cortex (GM)",
            xy=(85, 100),
            xytext=(20, 70),
            color="#FFD700",
            fontsize=10,
            fontweight="bold",
            arrowprops=dict(facecolor="#FFD700", shrink=0.08, width=1.5, headwidth=6),
        )
        # Subcutaneous Fat
        ax.annotate(
            "Subcutaneous Fat",
            xy=(150, 32),
            xytext=(160, 15),
            color="#FF5252",
            fontsize=10,
            fontweight="bold",
            arrowprops=dict(facecolor="#FF5252", shrink=0.08, width=1.5, headwidth=6),
        )

    plt.suptitle(
        "MRI Contrast Comparison: T1W vs T2W Signal Intensities",
        color="white",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor())
        logger.info("Saved contrast comparison figure to: %s", save_path)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Simulate and visualize T1W vs T2W MRI contrast."
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="results/t1_t2_comparison.png",
        help="Path to save output comparison figure.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display GUI window (useful in headless environments).",
    )
    args = parser.parse_args()

    # 1. Generate anatomical masks
    masks = generate_brain_phantom_masks()

    # 2. Define signal intensities based on medical MRI physics
    # T1W: CSF (Dark), GM (Medium Gray), WM (Bright/Brighter than GM), Fat (Bright)
    t1_intensities = {
        "csf": 15,
        "gm": 120,
        "wm": 180,
        "muscle": 80,
        "fat": 240,
    }

    # T2W: CSF (Bright), GM (Medium Gray), WM (Dark/Darker than GM), Fat (Bright)
    t2_intensities = {
        "csf": 250,
        "gm": 150,
        "wm": 90,
        "muscle": 80,
        "fat": 210,
    }

    logger.info("Synthesizing T1W and T2W phantom slices...")
    t1_img = synthesize_mri_sequence(masks, t1_intensities)
    t2_img = synthesize_mri_sequence(masks, t2_intensities)

    output_file = Path(args.save_path) if args.save_path else None
    plot_comparison(
        t1_img,
        t2_img,
        save_path=output_file,
        show_plot=not args.no_show,
    )


if __name__ == "__main__":
    main()
