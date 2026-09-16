"""
Script to inspect and generate summary statistics for the ISLES-2022 dataset.
Scans NIfTI volumes, extracts shapes, spacings, modalities, and exports report CSV.
"""

from pathlib import Path
import nibabel as nib
import pandas as pd

# ============================================================
# DYNAMIC PROJECT PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "ISLES-2022"
OUTPUT_PATH = PROJECT_ROOT / "results" / "reports"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# ============================================================
# FIND ALL NIFTI FILES
# ============================================================
nii_files = list(DATASET_PATH.rglob("*.nii.gz"))

print("=" * 70)
print("ISLES 2022 DATASET INSPECTION")
print("=" * 70)
print(f"Dataset path: {DATASET_PATH}")
print(f"Total NIfTI files found: {len(nii_files)}")

# ============================================================
# READ NIFTI HEADERS
# ============================================================
data = []

for i, file_path in enumerate(nii_files, start=1):
    try:
        img = nib.load(file_path)
        shape = img.shape
        spacing = img.header.get_zooms()[:3]
        depth = shape[2] if len(shape) >= 3 else None

        case_name = None
        for parent in file_path.parents:
            if parent.name.startswith("sub-strokecase"):
                case_name = parent.name
                break

        data.append({
            "case": case_name,
            "file": file_path.name,
            "path": str(file_path.relative_to(PROJECT_ROOT)),
            "shape": str(shape),
            "dim_x": shape[0] if len(shape) > 0 else None,
            "dim_y": shape[1] if len(shape) > 1 else None,
            "depth_z": depth,
            "spacing_x": spacing[0],
            "spacing_y": spacing[1],
            "spacing_z": spacing[2],
        })

        if i % 50 == 0 or i == len(nii_files):
            print(f"[{i}/{len(nii_files)}] {case_name} | {file_path.name} | Shape={shape} | Spacing={spacing}")

    except Exception as e:
        print(f"\nERROR reading: {file_path}")
        print(e)

# ============================================================
# CREATE DATAFRAME & SUMMARY
# ============================================================
df = pd.DataFrame(data)

print("\n" + "=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)
print(f"Total NIfTI files: {len(df)}")
print(f"Total detected cases: {df['case'].nunique()}")

print("\n" + "=" * 70)
print("UNIQUE SHAPES")
print("=" * 70)
print(df["shape"].value_counts())

unique_shapes = df["shape"].nunique()
print("\n" + "=" * 70)
print("SHAPE CONSISTENCY")
print("=" * 70)
if unique_shapes == 1:
    print("ALL FILES HAVE THE SAME SHAPE")
else:
    print(f"FILES HAVE {unique_shapes} DIFFERENT SHAPES")

print("\n" + "=" * 70)
print("DEPTH STATISTICS")
print("=" * 70)
print(df["depth_z"].describe())

print("\n" + "=" * 70)
print("VOXEL SPACING STATISTICS")
print("=" * 70)
spacing_columns = ["spacing_x", "spacing_y", "spacing_z"]
print(df[spacing_columns].describe())

# ============================================================
# EXPORT CSV
# ============================================================
output_csv = OUTPUT_PATH / "dataset_statistics.csv"
df.to_csv(output_csv, index=False)

print("\n" + "=" * 70)
print(f"Inspection complete. Summary saved to: {output_csv}")
print("=" * 70)
