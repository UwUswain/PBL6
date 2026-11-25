"""Phase-7 platform API (webapp/backend). Runs against in-memory SQLite; Docker runs it on PostgreSQL."""

import os
from pathlib import Path
import sys
import unittest

BACKEND = Path(__file__).resolve().parent.parent / "webapp" / "backend"


class HealthTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["DATABASE_URL"] = "sqlite://"
        sys.path.insert(0, str(BACKEND))
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
