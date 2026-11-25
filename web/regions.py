"""Vùng não của tổn thương theo atlas (giai đoạn 6).

Đăng ký affine mặt nạ não chuẩn MNI vào mặt nạ não của DWI (ảnh ISLES đã tách sọ, nên so mặt nạ với mặt nạ,
không phụ thuộc loại ảnh), rồi kéo bản đồ thùy não theo. Chỉ đúng ở mức thùy; không có nhãn vùng não để đo độ chính xác.

Atlas (FSL, web/atlas/): MNI Structural Atlas maxprob-thr25 2mm cho thùy và nhân xám sâu; Harvard-Oxford subcortical
cho thân não; MNI152 brain mask.
"""

from functools import lru_cache
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage

ATLAS_DIR = Path(__file__).parent / "atlas"
REGIONS = ["frontal", "parietal", "temporal", "occipital", "insula", "basal_ganglia", "thalamus", "cerebellum", "brainstem"]
# Giá trị trong MNI-maxprob (index trong MNI.xml + 1) -> vùng hiển thị
_MNI_LABELS = {1: "basal_ganglia", 2: "cerebellum", 3: "frontal", 4: "insula", 5: "occipital",
               6: "parietal", 7: "basal_ganglia", 8: "temporal", 9: "thalamus"}
_HO_BRAINSTEM = 8
RIGHT = 10  # mã vùng = số thứ tự vùng (1..9) + 10 nếu ở bán cầu phải

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _mni() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(bản đồ mã vùng, mặt nạ não, affine) trên lưới MNI 2 mm. Mọi voxel não đều có vùng."""
    mask_img = nib.load(str(ATLAS_DIR / "MNI152_T1_2mm_brain_mask.nii.gz"))
    brain = np.asarray(mask_img.dataobj) > 0
    lobes = np.asarray(nib.load(str(ATLAS_DIR / "MNI-maxprob-thr25-2mm.nii.gz")).dataobj).astype(int)
    ho = np.asarray(nib.load(str(ATLAS_DIR / "HarvardOxford-sub-maxprob-thr25-2mm.nii.gz")).dataobj).astype(int)

    region = np.zeros(brain.shape, np.uint8)
    for value, name in _MNI_LABELS.items():
        region[lobes == value] = REGIONS.index(name) + 1
    region[(region == 0) & (ho == _HO_BRAINSTEM)] = REGIONS.index("brainstem") + 1
    # thr25 bỏ trống chất trắng sâu: gán cho vùng gần nhất (ở mức thùy, chất trắng dưới vỏ thuộc thùy đó)
    _, nearest = ndimage.distance_transform_edt(region == 0, return_indices=True)
    region = np.where(brain, region[tuple(nearest)], 0).astype(np.uint8)

    x_mm = (np.indices(brain.shape).reshape(3, -1).T @ mask_img.affine[0, :3] + mask_img.affine[0, 3]).reshape(brain.shape)
    sided = (region > 0) & (region != REGIONS.index("brainstem") + 1) & (x_mm > 0)  # RAS: +x là bên phải
    region[sided] += RIGHT
    return region, brain, mask_img.affine


def _to_sitk(arr: np.ndarray, affine: np.ndarray):
    """Mảng nibabel (i, j, k, RAS) -> ảnh SimpleITK (LPS), giữ đúng thứ tự voxel để đổi ngược bằng transpose."""
    import SimpleITK as sitk

    img = sitk.GetImageFromArray(np.ascontiguousarray(arr.transpose(2, 1, 0)))
    ras_to_lps = np.diag([-1.0, -1.0, 1.0])
    lin = ras_to_lps @ affine[:3, :3]
    spacing = np.linalg.norm(lin, axis=0)
    img.SetSpacing(spacing.tolist())
    img.SetDirection((lin / spacing).ravel().tolist())
    img.SetOrigin((ras_to_lps @ affine[:3, 3]).tolist())
    return img


def region_map(dwi_path: Path) -> Tuple[np.ndarray, float]:
    """Bản đồ mã vùng trên lưới DWI gốc và độ khớp (Dice mặt nạ não MNI đã đăng ký với mặt nạ não DWI)."""
    import SimpleITK as sitk

    ref = nib.load(str(dwi_path))
    data = np.asarray(ref.dataobj)
    brain = ndimage.binary_fill_holes(data.reshape(data.shape[:3]) > 0)
    region, mni_brain, mni_affine = _mni()

    fixed = sitk.SmoothingRecursiveGaussian(_to_sitk(brain.astype(np.float32), ref.affine), 2.0)
    moving = sitk.SmoothingRecursiveGaussian(_to_sitk(mni_brain.astype(np.float32), mni_affine), 2.0)
    init = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.AffineTransform(3), sitk.CenteredTransformInitializerFilter.MOMENTS
    )
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMeanSquares()
    reg.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-4, 300)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetInitialTransform(init, inPlace=False)
    tx = reg.Execute(fixed, moving)

    def warp(arr: np.ndarray) -> np.ndarray:
        out = sitk.Resample(_to_sitk(arr, mni_affine), fixed, tx, sitk.sitkNearestNeighbor, 0)
        return sitk.GetArrayFromImage(out).transpose(2, 1, 0)

    warped = warp(mni_brain.astype(np.uint8)) > 0
    fit = float(2 * (brain & warped).sum() / (brain.sum() + warped.sum()))
    labels = warp(region)
    # Affine không khớp hết mép não (hay gặp ở hố sau): voxel não của bệnh nhân rơi ngoài atlas lấy vùng gần nhất
    _, nearest = ndimage.distance_transform_edt(labels == 0, return_indices=True)
    return np.where(brain, labels[tuple(nearest)], 0).astype(np.uint8), fit


def safe_region_map(dwi_path: Path) -> Optional[Tuple[np.ndarray, float]]:
    """Như region_map nhưng không làm hỏng kết quả phân đoạn nếu thiếu SimpleITK hoặc đăng ký lỗi."""
    try:
        return region_map(dwi_path)
    except Exception:
        log.exception("Atlas registration failed; region analysis skipped.")
        return None


def region_stats(mask: np.ndarray, regions: np.ndarray, zooms) -> List[dict]:
    """Thể tích tổn thương theo vùng, sắp giảm dần. region=None là phần nằm ngoài não của atlas."""
    mask = np.asarray(mask) > 0
    total = int(mask.sum())
    if not total:
        return []
    voxel_ml = float(np.prod(zooms)) / 1000.0
    codes, counts = np.unique(np.asarray(regions)[mask], return_counts=True)
    out = []
    for code, n in zip(codes.tolist(), counts.tolist()):
        name = REGIONS[code % RIGHT - 1] if code else None
        side = None if name in (None, "brainstem") else "right" if code > RIGHT else "left"
        out.append({"region": name, "side": side, "ml": n * voxel_ml, "share": n / total})
    return sorted(out, key=lambda r: -r["share"])
