"""Phase-7 platform API (webapp/backend): health endpoints. Runs on in-memory SQLite; Docker runs PostgreSQL."""

import unittest

from tests import webapp_env  # noqa: F401  (environment before importing app)


class HealthTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        from app.main import app

        cls.client = TestClient(app)

    def test_live_and_ready(self):
        self.assertEqual(self.client.get("/api/v1/health/live").json(), {"status": "ok"})
        self.assertEqual(self.client.get("/api/v1/health/ready").json(), {"status": "ready"})

    def test_ready_reports_database_down(self):
        from sqlalchemy import create_engine

        import app.main as main

        good = main.engine
        main.engine = create_engine("postgresql+psycopg://x:x@127.0.0.1:1/none", connect_args={"connect_timeout": 1})
        try:
            r = self.client.get("/api/v1/health/ready")
            self.assertEqual(r.status_code, 503)
        finally:
            main.engine = good


if __name__ == "__main__":
    unittest.main()
