"""Phase-7 M3 patient portal: accounts issued by the doctor, approved-only visibility, isolation."""

from pathlib import Path
import unittest
import uuid

from tests.test_webapp_clinical import ADC, DWI, FLAIR
from tests.webapp_env import ADMIN_PASS, ADMIN_USER


@unittest.skipUnless(DWI and ADC and FLAIR, "ISLES sample case 0121 not available")
class PatientPortalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        from app.main import app

        cls.TestClient, cls.app = TestClient, app
        cls.ctx = TestClient(app)
        cls.ctx.__enter__()
        admin = cls.signin(ADMIN_USER, ADMIN_PASS, "Strong-New-Pass-42")
        cls.doc = cls.make_user(admin, "portal.doc", "doctor")
        cls.other_doc = cls.make_user(admin, "portal.doc2", "doctor")

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)

    @classmethod
    def signin(cls, user, password, new_password):
        c = cls.TestClient(cls.app)
        r = c.post("/api/v1/auth/login", json={"username": user, "password": new_password})
        if r.status_code != 200:
            r = c.post("/api/v1/auth/login", json={"username": user, "password": password})
        assert r.status_code == 200, r.text
        tok = r.json()["access_token"]
        if r.json()["user"]["must_change_password"]:
            tok = c.post("/api/v1/auth/change-password", headers={"Authorization": f"Bearer {tok}"},
                         json={"current_password": password, "new_password": new_password}).json()["access_token"]
        return {"Authorization": f"Bearer {tok}"}

    @classmethod
    def make_user(cls, admin, username, role):
        r = cls.TestClient(cls.app).post("/api/v1/admin/users", headers=admin,
                                         json={"username": username, "display_name": username, "role": role})
        return cls.signin(username, r.json()["temporary_password"], "Portal-Pass-123")

    def setUp(self):
        self.c = self.TestClient(self.app)

    def study_with_ai_mask(self, code):
        """Patient + study with DWI/ADC and one AI mask version (stands in for the worker)."""
        from app.db import SessionLocal
        from app.models import Case, MaskVersion
        from app.storage import case_dir

        pt = self.c.post("/api/v1/patients", headers=self.doc, json={"code": code}).json()
        case = self.c.post(f"/api/v1/patients/{pt['id']}/cases", headers=self.doc, json={}).json()
        for m, path in (("dwi", DWI), ("adc", ADC)):
            with open(path, "rb") as f:
                self.c.put(f"/api/v1/cases/{case['id']}/volumes/{m}", headers=self.doc, files={"file": (Path(path).name, f)})
        with SessionLocal() as db:
            path = case_dir(uuid.UUID(case["id"])) / "masks" / "v1.nii.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x1f\x8b fake")
            m = MaskVersion(case_id=uuid.UUID(case["id"]), version_no=1, source="ai", model_name="m1",
                            created_by=db.get(Case, uuid.UUID(case["id"])).owner_id, path=str(path),
                            stats={"volume_ml": 4.6, "lesion_count": 2, "hemisphere": "right", "regions": [],
                                   "atlas_fit": 0.9, "midline_x": 1.0, "shape": [1, 2, 3]})
            db.add(m)
            db.get(Case, uuid.UUID(case["id"])).status = "awaiting_review"
            db.commit()
            return pt, case, str(m.id)

    def patient_login(self, pt):
        r = self.c.post(f"/api/v1/patients/{pt['id']}/account", headers=self.doc)
        self.assertEqual(r.status_code, 201, r.text)
        self.assertEqual(r.json()["username"], f"bn-{pt['code'].lower()}")
        return r.json(), self.signin(r.json()["username"], r.json()["temporary_password"], "Patient-Pass-123")

    def test_approved_only_and_isolated(self):
        pt, case, mask_id = self.study_with_ai_mask("PORTAL-A")
        account, me = self.patient_login(pt)
        self.assertEqual(self.c.post(f"/api/v1/patients/{pt['id']}/account", headers=self.doc).status_code, 409)
        self.assertEqual(self.c.post(f"/api/v1/patients/{pt['id']}/account", headers=self.other_doc).status_code, 404)

        # Not approved yet: nothing visible.
        self.assertEqual(self.c.get("/api/v1/me/results", headers=me).json(), [])
        self.assertEqual(self.c.get(f"/api/v1/me/results/{case['id']}", headers=me).status_code, 404)
        self.assertEqual(self.c.get(f"/api/v1/me/results/{case['id']}/mask", headers=me).status_code, 404)

        self.c.post(f"/api/v1/cases/{case['id']}/approve", headers=self.doc,
                    json={"mask_version_id": mask_id, "patient_note": "Uống thuốc đều", "show_functions": False})
        results = self.c.get("/api/v1/me/results", headers=me).json()
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual((r["patient_note"], r["show_functions"], r["edited_by_doctor"]), ("Uống thuốc đều", False, False))
        self.assertEqual(set(r["stats"]), {"volume_ml", "lesion_count", "hemisphere", "regions", "atlas_fit"})  # no internals
        self.assertEqual(self.c.get(f"/api/v1/me/results/{case['id']}/mask", headers=me).status_code, 200)
        self.assertEqual(self.c.get(f"/api/v1/me/results/{case['id']}/dwi", headers=me).status_code, 200)

        # Another patient sees nothing of it.
        pt_b, _, _ = self.study_with_ai_mask("PORTAL-B")
        _, me_b = self.patient_login(pt_b)
        self.assertEqual(self.c.get(f"/api/v1/me/results/{case['id']}", headers=me_b).status_code, 404)
        self.assertEqual(self.c.get(f"/api/v1/me/results/{case['id']}/mask", headers=me_b).status_code, 404)

        # Patients cannot use doctor or admin APIs.
        self.assertEqual(self.c.get("/api/v1/patients", headers=me).status_code, 403)
        self.assertEqual(self.c.get(f"/api/v1/cases/{case['id']}", headers=me).status_code, 403)
        self.assertEqual(self.c.get("/api/v1/admin/users", headers=me).status_code, 403)

        # Withdrawn approval hides it again at once.
        self.c.post(f"/api/v1/cases/{case['id']}/unapprove", headers=self.doc)
        self.assertEqual(self.c.get(f"/api/v1/me/results/{case['id']}", headers=me).status_code, 404)

    def test_reset_password_revokes_sessions(self):
        pt, _, _ = self.study_with_ai_mask("PORTAL-C")
        account, me = self.patient_login(pt)
        r = self.c.post(f"/api/v1/patients/{pt['id']}/account/reset-password", headers=self.doc)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.c.get("/api/v1/auth/me", headers=me).status_code, 401)  # old session ended
        login = self.c.post("/api/v1/auth/login", json={"username": account["username"], "password": r.json()["temporary_password"]})
        self.assertTrue(login.json()["user"]["must_change_password"])
        self.assertEqual(self.c.get(f"/api/v1/patients/{pt['id']}", headers=self.doc).json()["has_account"], True)


if __name__ == "__main__":
    unittest.main()
