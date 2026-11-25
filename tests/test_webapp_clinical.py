"""Phase-7 M2 doctor workflow: patients, cases, uploads (validation, grid check, lock), jobs, approval, isolation."""

import glob
import os
from pathlib import Path
import unittest

from tests.webapp_env import ADMIN_PASS, ADMIN_USER

ROOT = Path(__file__).resolve().parent.parent


def _real(kind: str) -> str | None:
    # ISLES files may sit inside a folder named "<case>_dwi.nii/", so match real files at any depth.
    pattern = str(ROOT / "ISLES-2022" / "**" / "sub-strokecase0121" / "**" / f"*{kind}*")
    hits = sorted(p for p in glob.glob(pattern, recursive=True) if os.path.isfile(p) and ".nii" in p.lower())
    return hits[0] if hits else None


DWI, ADC, FLAIR = _real("dwi"), _real("adc"), _real("flair")


@unittest.skipUnless(DWI and ADC and FLAIR, "ISLES sample case 0121 not available")
class ClinicalFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        from app.main import app

        cls.TestClient = TestClient
        cls.app = app
        cls.ctx = TestClient(app)
        cls.ctx.__enter__()
        admin = cls._signin(ADMIN_USER, ADMIN_PASS, "Strong-New-Pass-42")
        cls.doc_a = cls._make_doctor(admin, "doc.a")
        cls.doc_b = cls._make_doctor(admin, "doc.b")
        cls.admin = admin

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)

    @classmethod
    def _signin(cls, user, password, new_password):
        c = cls.TestClient(cls.app)
        # Try the changed password first: failed tries count toward the login rate limit.
        r = c.post("/api/v1/auth/login", json={"username": user, "password": new_password})
        if r.status_code != 200:
            r = c.post("/api/v1/auth/login", json={"username": user, "password": password})
        tok = r.json()["access_token"]
        if r.json()["user"]["must_change_password"]:
            tok = c.post("/api/v1/auth/change-password", headers={"Authorization": f"Bearer {tok}"},
                         json={"current_password": password, "new_password": new_password}).json()["access_token"]
        return {"Authorization": f"Bearer {tok}"}

    @classmethod
    def _make_doctor(cls, admin, username):
        c = cls.TestClient(cls.app)
        r = c.post("/api/v1/admin/users", headers=admin, json={"username": username, "display_name": username, "role": "doctor"})
        return cls._signin(username, r.json()["temporary_password"], "Doctor-Pass-123")

    def setUp(self):
        self.c = self.TestClient(self.app)

    def upload(self, case_id, modality, path, headers=None):
        with open(path, "rb") as f:
            return self.c.put(f"/api/v1/cases/{case_id}/volumes/{modality}", headers=headers or self.doc_a,
                              files={"file": (Path(path).name, f)})

    def new_case(self, code):
        pt = self.c.post("/api/v1/patients", headers=self.doc_a, json={"code": code, "birth_year": 1960, "sex": "F"})
        self.assertEqual(pt.status_code, 201, pt.text)
        case = self.c.post(f"/api/v1/patients/{pt.json()['id']}/cases", headers=self.doc_a, json={"study_date": "2026-11-20"})
        self.assertEqual(case.status_code, 201, case.text)
        return pt.json(), case.json()

    def add_model(self, name="m1", required=("dwi", "adc"), available=True):
        from app.db import SessionLocal
        from app.models import ModelEntry

        with SessionLocal() as db:
            db.merge(ModelEntry(name=name, label=name, modalities=list(required), required=list(required),
                                available=available, is_default=True))
            db.commit()

    def add_mask(self, case_id, version_no=1):
        """Stands in for the worker's output."""
        import uuid

        from app.db import SessionLocal
        from app.models import Case, MaskVersion
        from app.storage import case_dir

        with SessionLocal() as db:
            path = case_dir(uuid.UUID(case_id)) / "masks" / f"v{version_no}.nii.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x1f\x8b fake")
            m = MaskVersion(case_id=uuid.UUID(case_id), version_no=version_no, source="ai", model_name="m1",
                            created_by=db.get(Case, uuid.UUID(case_id)).owner_id, path=str(path), stats={"volume_ml": 1.0})
            db.add(m)
            db.get(Case, uuid.UUID(case_id)).status = "awaiting_review"
            db.commit()
            return str(m.id)

    def test_patient_isolation_and_roles(self):
        pt, case = self.new_case("PT-ISO")
        self.assertEqual(self.c.post("/api/v1/patients", headers=self.doc_a, json={"code": "PT-ISO"}).status_code, 409)
        self.assertEqual(self.c.post("/api/v1/patients", headers=self.doc_a, json={"code": "bad code!"}).status_code, 422)
        # Another doctor sees nothing and gets 404 (not 403), so existence is not revealed.
        self.assertNotIn("PT-ISO", [p["code"] for p in self.c.get("/api/v1/patients", headers=self.doc_b).json()])
        self.assertEqual(self.c.get(f"/api/v1/cases/{case['id']}", headers=self.doc_b).status_code, 404)
        self.assertEqual(self.upload(case["id"], "dwi", DWI, headers=self.doc_b).status_code, 404)
        self.assertEqual(self.c.get("/api/v1/patients", headers=self.admin).status_code, 403)  # admin: no clinical data
        mine = self.c.get("/api/v1/patients?q=ISO", headers=self.doc_a).json()
        self.assertEqual((mine[0]["code"], mine[0]["case_count"]), ("PT-ISO", 1))

    def test_uploads_validation_grid_check_and_lock(self):
        _, case = self.new_case("PT-UP")
        cid = case["id"]
        r = self.c.put(f"/api/v1/cases/{cid}/volumes/dwi", headers=self.doc_a, files={"file": ("x.txt", b"hello")})
        self.assertEqual(r.status_code, 400)
        r = self.c.put(f"/api/v1/cases/{cid}/volumes/dwi", headers=self.doc_a, files={"file": ("x.nii", b"not nifti")})
        self.assertEqual(r.status_code, 400)
        r = self.upload(cid, "dwi", DWI)
        self.assertEqual((r.status_code, r.json()["status"]), (200, "draft"), r.text)
        # FLAIR uploaded as ADC: different grid -> refused, DWI untouched.
        r = self.upload(cid, "adc", FLAIR)
        self.assertEqual(r.status_code, 400)
        self.assertIn("same grid", r.json()["detail"])
        r = self.upload(cid, "adc", ADC)
        self.assertEqual((r.status_code, r.json()["status"]), (200, "ready"))
        self.assertEqual(sorted(v["modality"] for v in r.json()["volumes"]), ["adc", "dwi"])
        self.assertEqual(self.c.get(f"/api/v1/cases/{cid}/volumes/dwi/file", headers=self.doc_a).status_code, 200)
        self.add_mask(cid)
        r = self.upload(cid, "flair", FLAIR)
        self.assertEqual(r.status_code, 409)  # locked after the AI ran

    def test_jobs_and_approval(self):
        _, case = self.new_case("PT-JOB")
        cid = case["id"]
        self.add_model("unavailable", available=False)
        self.assertEqual(self.c.post(f"/api/v1/cases/{cid}/jobs", headers=self.doc_a, json={"model_name": "m1"}).status_code, 400)
        self.add_model("m1")
        self.assertIn("Upload first", self.c.post(f"/api/v1/cases/{cid}/jobs", headers=self.doc_a,
                                                    json={"model_name": "m1"}).json()["detail"])
        self.upload(cid, "dwi", DWI)
        self.upload(cid, "adc", ADC)
        self.assertEqual(self.c.post(f"/api/v1/cases/{cid}/jobs", headers=self.doc_a, json={"model_name": "unavailable"}).status_code, 400)
        job = self.c.post(f"/api/v1/cases/{cid}/jobs", headers=self.doc_a, json={"model_name": "m1"})
        self.assertEqual((job.status_code, job.json()["state"]), (202, "queued"))
        self.assertEqual(self.c.post(f"/api/v1/cases/{cid}/jobs", headers=self.doc_a, json={"model_name": "m1"}).status_code, 409)
        self.assertEqual(self.c.get(f"/api/v1/jobs/{job.json()['id']}", headers=self.doc_b).status_code, 404)
        r = self.c.post(f"/api/v1/jobs/{job.json()['id']}/cancel", headers=self.doc_a)
        self.assertEqual(r.json()["state"], "cancelled")
        self.assertEqual(self.c.get(f"/api/v1/cases/{cid}", headers=self.doc_a).json()["status"], "ready")

        mask_id = self.add_mask(cid)
        _, other = self.new_case("PT-OTHER")
        other_mask = self.add_mask(other["id"])
        self.assertEqual(self.c.post(f"/api/v1/cases/{cid}/approve", headers=self.doc_a,
                                     json={"mask_version_id": other_mask}).status_code, 400)
        r = self.c.post(f"/api/v1/cases/{cid}/approve", headers=self.doc_a,
                        json={"mask_version_id": mask_id, "patient_note": "Tái khám sau 1 tuần", "show_functions": False})
        self.assertEqual((r.status_code, r.json()["status"]), (200, "approved"), r.text)
        self.assertEqual(r.json()["approval"]["patient_note"], "Tái khám sau 1 tuần")
        self.assertEqual(self.c.post(f"/api/v1/cases/{cid}/jobs", headers=self.doc_a, json={"model_name": "m1"}).status_code, 409)
        self.assertEqual(self.c.get(f"/api/v1/masks/{mask_id}/file", headers=self.doc_a).status_code, 200)
        self.assertEqual(self.c.get(f"/api/v1/masks/{mask_id}/file", headers=self.doc_b).status_code, 404)
        r = self.c.post(f"/api/v1/cases/{cid}/unapprove", headers=self.doc_a)
        self.assertEqual((r.json()["status"], r.json()["approval"]), ("awaiting_review", None))
        queue = self.c.get("/api/v1/cases?status=awaiting_review", headers=self.doc_a).json()
        self.assertIn(cid, [c["id"] for c in queue])


if __name__ == "__main__":
    unittest.main()
