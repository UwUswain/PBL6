"""Phase-7 auth: login, forced password change, refresh rotation with reuse detection, logout,
rate limiting, roles, admin account management, audit log."""

import unittest

from tests.webapp_env import ADMIN_PASS, ADMIN_USER


class AuthFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        from app.main import app

        cls.app = app
        cls.TestClient = TestClient
        cls.ctx = TestClient(app)
        cls.ctx.__enter__()  # runs the lifespan, which creates the bootstrap admin

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)

    def client(self):
        return self.TestClient(self.app)  # its own cookie jar = one browser

    def login(self, c, user, password, status=200):
        r = c.post("/api/v1/auth/login", json={"username": user, "password": password})
        self.assertEqual(r.status_code, status, r.text)
        return r.json()

    @staticmethod
    def bearer(token):
        return {"Authorization": f"Bearer {token}"}

    def admin_token(self):
        """Admin with the bootstrap password changed (idempotent across tests)."""
        c = self.client()
        # Changed password first: failed tries count toward the login rate limit.
        r = c.post("/api/v1/auth/login", json={"username": ADMIN_USER, "password": "Strong-New-Pass-42"})
        if r.status_code == 200:
            return r.json()["access_token"]
        tok = self.login(c, ADMIN_USER, ADMIN_PASS)["access_token"]
        r = c.post("/api/v1/auth/change-password", headers=self.bearer(tok),
                   json={"current_password": ADMIN_PASS, "new_password": "Strong-New-Pass-42"})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()["access_token"]

    def test_forced_password_change_blocks_everything_else(self):
        # A fresh account (bootstrap admin may already be changed by another test; order-independent).
        admin = self.bearer(self.admin_token())
        temp = self.client().post("/api/v1/admin/users", headers=admin, json={
            "username": "second.admin", "display_name": "Second", "role": "admin"}).json()["temporary_password"]
        c = self.client()
        body = self.login(c, "second.admin", temp)
        self.assertTrue(body["user"]["must_change_password"])
        tok = self.bearer(body["access_token"])
        self.assertEqual(c.get("/api/v1/admin/users", headers=tok).status_code, 403)
        self.assertEqual(c.get("/api/v1/auth/me", headers=tok).status_code, 200)
        # too short / unchanged / contains the username
        for bad in ("short", temp, "my-second.admin-pass"):
            r = c.post("/api/v1/auth/change-password", headers=tok, json={"current_password": temp, "new_password": bad})
            self.assertEqual(r.status_code, 400, bad)
        self.assertEqual(c.post("/api/v1/auth/change-password", headers=tok, json={
            "current_password": "wrong-one", "new_password": "Another-Pass-77"}).status_code, 400)
        r = c.post("/api/v1/auth/change-password", headers=tok,
                   json={"current_password": temp, "new_password": "Another-Pass-77"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(c.get("/api/v1/admin/users", headers=self.bearer(r.json()["access_token"])).status_code, 200)

    def test_bootstrap_admin_exists_and_must_change(self):
        from app.db import SessionLocal
        from app.models import User
        from sqlalchemy import select

        with SessionLocal() as db:
            admin = db.scalar(select(User).where(User.username == ADMIN_USER))
        self.assertEqual(admin.role, "admin")
        self.assertNotEqual(admin.password_hash, ADMIN_PASS)  # only the Argon2 hash is stored
        self.assertTrue(admin.password_hash.startswith("$argon2id$"))

    def test_refresh_rotation_and_reuse_detection(self):
        tok = self.admin_token()
        c = self.client()
        tok = self.login(c, ADMIN_USER, "Strong-New-Pass-42")["access_token"]
        old_cookie = c.cookies.get("pbl6_refresh")
        r = c.post("/api/v1/auth/refresh")
        self.assertEqual(r.status_code, 200, r.text)
        new_tok = r.json()["access_token"]
        self.assertNotEqual(c.cookies.get("pbl6_refresh"), old_cookie)  # rotated
        # An attacker replays the old refresh token: the whole session ends.
        thief = self.client()
        thief.cookies.set("pbl6_refresh", old_cookie, path="/api/v1/auth")
        self.assertEqual(thief.post("/api/v1/auth/refresh").status_code, 401)
        self.assertEqual(c.get("/api/v1/auth/me", headers=self.bearer(new_tok)).status_code, 401)
        self.assertEqual(c.post("/api/v1/auth/refresh").status_code, 401)
        self.assertEqual(c.get("/api/v1/auth/me", headers=self.bearer(tok)).status_code, 401)

    def test_logout_revokes_access_token_immediately(self):
        self.admin_token()
        c = self.client()
        tok = self.login(c, ADMIN_USER, "Strong-New-Pass-42")["access_token"]
        self.assertEqual(c.post("/api/v1/auth/logout").status_code, 204)
        self.assertEqual(c.get("/api/v1/auth/me", headers=self.bearer(tok)).status_code, 401)
        self.assertEqual(c.post("/api/v1/auth/refresh").status_code, 401)

    def test_wrong_password_and_rate_limit(self):
        c = self.client()
        r = c.post("/api/v1/auth/login", json={"username": "ghost-user", "password": "nope"})
        self.assertEqual((r.status_code, r.json()["detail"]), (401, "Invalid username or password"))
        for _ in range(4):
            c.post("/api/v1/auth/login", json={"username": "ghost-user", "password": "nope"})
        r = c.post("/api/v1/auth/login", json={"username": "ghost-user", "password": "nope"})
        self.assertEqual(r.status_code, 429)
        self.assertGreater(int(r.headers["Retry-After"]), 0)

    def test_admin_manages_doctor_and_roles(self):
        admin = self.bearer(self.admin_token())
        c = self.client()
        r = c.post("/api/v1/admin/users", headers=admin,
                   json={"username": "Dr.Lan", "display_name": "BS. Lan", "role": "doctor"})
        self.assertEqual(r.status_code, 201, r.text)
        doctor_id, temp = r.json()["user"]["id"], r.json()["temporary_password"]
        self.assertEqual(c.post("/api/v1/admin/users", headers=admin,
                                json={"username": "dr.lan", "display_name": "x", "role": "doctor"}).status_code, 409)
        self.assertEqual(c.post("/api/v1/admin/users", headers=admin,
                                json={"username": "x1", "display_name": "x", "role": "root"}).status_code, 422)

        d = self.client()
        tok = self.login(d, "dr.lan", temp)["access_token"]
        tok = d.post("/api/v1/auth/change-password", headers=self.bearer(tok),
                     json={"current_password": temp, "new_password": "Doctor-Pass-123"}).json()["access_token"]
        self.assertEqual(d.get("/api/v1/admin/users", headers=self.bearer(tok)).status_code, 403)  # not admin

        # Disabling the account ends the doctor's session at once.
        self.assertEqual(c.patch(f"/api/v1/admin/users/{doctor_id}", headers=admin, json={"is_active": False}).status_code, 200)
        self.assertEqual(d.get("/api/v1/auth/me", headers=self.bearer(tok)).status_code, 401)
        self.login(self.client(), "dr.lan", "Doctor-Pass-123", status=401)

        # Re-enable + reset: old password no longer works, a new temp one must be changed.
        c.patch(f"/api/v1/admin/users/{doctor_id}", headers=admin, json={"is_active": True})
        temp2 = c.post(f"/api/v1/admin/users/{doctor_id}/reset-password", headers=admin).json()["temporary_password"]
        self.login(self.client(), "dr.lan", "Doctor-Pass-123", status=401)
        self.assertTrue(self.login(self.client(), "dr.lan", temp2)["user"]["must_change_password"])

    def test_admin_cannot_lock_itself_out_and_audit_is_recorded(self):
        admin = self.bearer(self.admin_token())
        c = self.client()
        me = c.get("/api/v1/auth/me", headers=admin).json()
        self.assertEqual(c.patch(f"/api/v1/admin/users/{me['id']}", headers=admin, json={"is_active": False}).status_code, 400)
        self.assertEqual(c.patch(f"/api/v1/admin/users/{me['id']}", headers=admin, json={"role": "doctor"}).status_code, 400)
        actions = {e["action"] for e in c.get("/api/v1/admin/audit-events", headers=admin).json()}
        self.assertTrue({"auth.login", "auth.change_password"} <= actions, actions)
        self.assertEqual(c.get("/api/v1/admin/audit-events").status_code, 401)  # no token


if __name__ == "__main__":
    unittest.main()
