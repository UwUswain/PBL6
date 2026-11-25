"""Kiểm thử backend web demo: pipeline suy luận, đổi model, và API."""

from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import torch

from src.models.builder import get_model
from web.inference import ROOT, get_model_cfg, load_config, predict

CONFIG = load_config()
CASE_ID = "sub-strokecase0121"


def _sample_case():
    from web.app import _cases

    return _cases().get(CASE_ID)


def _default_checkpoint_exists() -> bool:
    return all((ROOT / c).is_file() for c in get_model_cfg(CONFIG)["checkpoints"])


@unittest.skipUnless(_sample_case(), "ISLES sample data not available")
class TestInference(unittest.TestCase):
    def setUp(self):
        self.case = _sample_case()
        self.paths = {m: Path(self.case[m]) for m in ("dwi", "adc", "flair")}

    @unittest.skipUnless(_default_checkpoint_exists(), "default checkpoint not available")
    def test_mask_is_in_original_dwi_space_and_overlaps_ground_truth(self):
        mask_img, stats = predict(self.paths, get_model_cfg(CONFIG))
        gt = nib.load(self.case["mask"])
        pred, truth = np.asarray(mask_img.dataobj) > 0, np.asarray(gt.dataobj) > 0

        self.assertEqual(pred.shape, truth.shape)
        self.assertTrue(np.allclose(mask_img.affine, gt.affine, atol=1e-3))
        # Mask lệch lưới sẽ cho Dice gần 0; ca này model cũ đạt ~0.90
        dice = 2 * (pred & truth).sum() / (pred.sum() + truth.sum())
        self.assertGreater(dice, 0.5)

        voxel_ml = float(np.prod(gt.header.get_zooms()[:3])) / 1000.0
        self.assertAlmostEqual(stats["volume_ml"], pred.sum() * voxel_ml, places=4)
        self.assertAlmostEqual(stats["left_ml"] + stats["right_ml"], stats["volume_ml"], places=4)
        self.assertEqual(stats["hemisphere"], "right")
        # Tâm ổ lớn nhất phải rơi vào một voxel tổn thương thật
        vox = np.rint(np.linalg.inv(gt.affine) @ [*stats["focus_mm"], 1])[:3].astype(int)
        self.assertTrue(pred[tuple(vox)])

    def test_swapping_in_a_two_channel_model_needs_only_config(self):
        kwargs = {"init_filters": 8}
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "random_dual.pth"
            net = get_model("dual_segresnet", in_channels=2, out_channels=1, **kwargs)
            torch.save({"model_state_dict": net.state_dict()}, ckpt)
            model_cfg = {
                "arch": "dual_segresnet",
                "modalities": ["dwi", "adc"],
                "checkpoints": [str(ckpt)],
                "roi_size": [96, 96, 64],
                "kwargs": kwargs,
            }
            # Không đưa FLAIR: model 2 kênh không được đòi nó
            mask_img, stats = predict({"dwi": self.paths["dwi"], "adc": self.paths["adc"]}, model_cfg)
        self.assertEqual(mask_img.shape, nib.load(str(self.paths["dwi"])).shape)
        self.assertGreaterEqual(stats["volume_ml"], 0.0)

    @unittest.skipUnless(_default_checkpoint_exists(), "default checkpoint not available")
    def test_empty_prediction_reports_no_lesion(self):
        cfg = dict(get_model_cfg(CONFIG), threshold=2.0)  # xác suất không bao giờ vượt 2
        _, stats = predict(self.paths, cfg)
        self.assertEqual((stats["volume_ml"], stats["lesion_count"]), (0.0, 0))
        self.assertEqual(stats["hemisphere"], "none")
        self.assertIsNone(stats["focus_mm"])

    def test_sample_case_without_flair_is_not_fed_dwi_as_flair(self):
        from web.app import _case_files

        fake = {"dwi": "a_dwi.nii", "adc": "a_adc.nii", "flair": "a_dwi.nii", "mask": ""}
        self.assertNotIn("flair", _case_files(fake))

    def test_missing_required_modality_is_a_user_error(self):
        with self.assertRaisesRegex(ValueError, "ADC"):
            predict({"dwi": self.paths["dwi"]}, get_model_cfg(CONFIG))


class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from web.app import app

        cls.client = TestClient(app)

    def test_models_endpoint_lists_default(self):
        body = self.client.get("/api/models").json()
        self.assertIn(body["default"], [m["name"] for m in body["models"]])

    def test_rejects_non_nifti_upload(self):
        r = self.client.post("/api/predict", files={"dwi": ("scan.png", b"x"), "adc": ("adc.nii", b"x")})
        self.assertEqual(r.status_code, 400)

    def test_rejects_corrupt_nifti(self):
        files = {m: (f"{m}.nii", b"not a nifti") for m in ("dwi", "adc", "flair")}
        self.assertEqual(self.client.post("/api/predict", files=files).status_code, 400)

    def test_unknown_model_and_result(self):
        self.assertEqual(self.client.post("/api/predict", data={"model": "nope"}).status_code, 400)
        self.assertEqual(self.client.get("/api/results/../../x/predicted_mask.nii.gz").status_code, 404)
        self.assertEqual(self.client.get(f"/api/cases/{CASE_ID}/..%2F..%2Fsecret.nii").status_code, 404)

    @unittest.skipUnless(_sample_case(), "ISLES sample data not available")
    def test_featured_cases_exist_and_have_both_languages(self):
        featured = {c["case_id"]: c["featured"] for c in self.client.get("/api/cases").json() if c["featured"]}
        self.assertEqual(set(featured), set(CONFIG["featured_cases"]))
        for note in featured.values():
            self.assertTrue(note["vi"] and note["en"])

    def test_vendored_viewer_library_is_served(self):
        r = self.client.get("/vendor/niivue-0.69.0.umd.js")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"niivue", r.content[:400])

    def test_index_page_is_served(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("gl", r.text)

    @unittest.skipUnless(_sample_case(), "ISLES sample data not available")
    def test_case_file_urls_are_downloadable(self):
        case = next(c for c in self.client.get("/api/cases").json() if c["case_id"] == CASE_ID)
        r = self.client.get(case["files"]["dwi"])
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.content), Path(_sample_case()["dwi"]).stat().st_size)

    @unittest.skipUnless(_sample_case() and _default_checkpoint_exists(), "sample data or checkpoint missing")
    def test_predict_sample_case_and_download_mask(self):
        r = self.client.post("/api/predict", data={"case_id": CASE_ID})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertGreater(body["volume_ml"], 0)
        self.assertGreater(body["ground_truth"]["dice"], 0.5)
        self.assertGreater(body["ground_truth"]["volume_ml"], 0)
        mask = self.client.get(body["mask_url"])
        self.assertEqual(mask.status_code, 200)
        self.assertEqual(mask.content[:2], b"\x1f\x8b")  # gzip

    @unittest.skipUnless(_sample_case() and _default_checkpoint_exists(), "sample data or checkpoint missing")
    def test_edited_mask_recomputes_stats(self):
        import gzip

        body = self.client.post("/api/predict", data={"case_id": CASE_ID}).json()
        original = nib.Nifti1Image.from_bytes(gzip.decompress(self.client.get(body["mask_url"]).content))
        data = np.asarray(original.dataobj).copy()
        data[:] = 0
        data[10:15, 10:15, 10:15] = 1  # bác sĩ xóa hết và vẽ một khối 5x5x5 voxel
        edited_bytes = nib.Nifti1Image(data, original.affine).to_bytes()

        url = f"/api/results/{body['job_id']}/edited"
        r = self.client.post(url, files={"mask": ("edited.nii", edited_bytes)})
        self.assertEqual(r.status_code, 200, r.text)
        out = r.json()
        voxel_ml = float(np.prod(original.header.get_zooms()[:3])) / 1000.0
        self.assertAlmostEqual(out["volume_ml"], 125 * voxel_ml, places=4)
        self.assertEqual(out["lesion_count"], 1)
        self.assertLess(out["ground_truth"]["dice"], body["ground_truth"]["dice"])
        saved = self.client.get(out["mask_url"])
        self.assertEqual(int(np.asarray(nib.Nifti1Image.from_bytes(gzip.decompress(saved.content)).dataobj).sum()), 125)
        # Bản AI gốc vẫn tải được, không bị ghi đè
        self.assertEqual(self.client.get(body["mask_url"]).content[:2], b"\x1f\x8b")

        wrong_grid = nib.Nifti1Image(data[:-1], original.affine).to_bytes()
        self.assertEqual(self.client.post(url, files={"mask": ("x.nii", wrong_grid)}).status_code, 400)
        self.assertEqual(self.client.post(url, files={"mask": ("x.nii", b"junk")}).status_code, 400)
        self.assertEqual(self.client.post("/api/results/" + "0" * 32 + "/edited", files={"mask": ("x.nii", edited_bytes)}).status_code, 404)
        self.assertEqual(self.client.get(f"/api/results/{body['job_id']}/meta.json").status_code, 404)
        self.assertEqual(self.client.get(f"/api/results/{body['job_id']}/regions.nii.gz").status_code, 404)
        # Vùng não tính lại theo mask đã sửa
        self.assertGreater(body["atlas_fit"], 0.8)
        self.assertAlmostEqual(sum(r["share"] for r in out["regions"]), 1.0, places=6)


@unittest.skipUnless(_sample_case(), "ISLES sample data not available")
class RegionAtlasTest(unittest.TestCase):
    def test_ground_truth_lesion_lands_in_right_cerebellum(self):
        from web.regions import region_map, region_stats

        case = _sample_case()
        regions, fit = region_map(Path(case["dwi"]))
        gt = nib.load(case["mask"])
        self.assertEqual(regions.shape, gt.shape[:3])
        self.assertGreater(fit, 0.85)  # mặt nạ não MNI sau đăng ký khớp mặt nạ não DWI
        stats = region_stats(gt.dataobj, regions, gt.header.get_zooms()[:3])
        # Ca 0121: ổ nhỏ ở tiểu não phải (kiểm tra bằng mắt); cũng kiểm tra luôn quy ước trái/phải
        self.assertEqual((stats[0]["region"], stats[0]["side"]), ("cerebellum", "right"))
        self.assertGreater(stats[0]["share"], 0.8)
        self.assertAlmostEqual(sum(r["share"] for r in stats), 1.0, places=6)
        self.assertEqual(region_stats(np.zeros(gt.shape[:3]), regions, (2, 2, 2)), [])


@unittest.skipUnless(_sample_case(), "ISLES sample data not available")
class ExternalModelTest(unittest.TestCase):
    """arch: external — model chạy bằng lệnh riêng (SEALS, Factorizer) ghi mask ra {out}."""

    SCRIPT = (
        "import sys, nibabel as nib, numpy as np; ref = nib.load(sys.argv[1]); m = np.zeros(ref.shape[:3], np.uint8); "
        "m[40:43, 40:43, 30:33] = 1; nib.save(nib.Nifti1Image(m, ref.affine), sys.argv[2])"
    )

    def _cfg(self, script):
        import sys

        return {"arch": "external", "modalities": ["dwi", "adc"], "checkpoints": [],
                "command": [sys.executable, "-c", script, "{dwi}", "{out}"]}

    def test_external_command_mask_and_stats(self):
        from web.app import _case_files

        paths = {m: Path(p) for m, p in _case_files(_sample_case()).items()}
        mask_img, stats = predict(paths, self._cfg(self.SCRIPT))
        self.assertEqual(int(np.asarray(mask_img.dataobj).sum()), 27)
        self.assertEqual(stats["lesion_count"], 1)
        self.assertEqual(mask_img.shape, nib.load(str(paths["dwi"])).shape[:3])
        with self.assertRaises(RuntimeError):  # lệnh lỗi phải báo rõ, không trả mask rỗng
            predict(paths, self._cfg("import sys; sys.exit('boom')"))


if __name__ == "__main__":
    unittest.main()
