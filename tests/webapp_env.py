"""Shared setup for webapp/backend tests: in-memory SQLite, throw-away JWT keys, a bootstrap admin.

Must be imported before anything from `app`, because app.config reads the environment at import time.
"""

import os
from pathlib import Path
import sys
import tempfile

BACKEND = Path(__file__).resolve().parent.parent / "webapp" / "backend"
ADMIN_USER, ADMIN_PASS = "admin", "Bootstrap-Pass-1"

os.environ.setdefault("DATABASE_URL", "sqlite://")
os.environ.setdefault("JWT_KEY_DIR", tempfile.mkdtemp(prefix="pbl6_jwt_"))
os.environ.setdefault("BOOTSTRAP_ADMIN_USERNAME", ADMIN_USER)
os.environ.setdefault("BOOTSTRAP_ADMIN_PASSWORD", ADMIN_PASS)
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.db import Base, engine  # noqa: E402
import app.models  # noqa: E402,F401  (registers the tables)

Base.metadata.create_all(engine)
