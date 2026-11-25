"""Settings from environment variables (docker compose sets them; tests override them)."""

import os
from pathlib import Path

ACCESS_TTL_SECONDS = int(os.environ.get("ACCESS_TTL_SECONDS", "600"))
REFRESH_TTL_DAYS = int(os.environ.get("REFRESH_TTL_DAYS", "7"))
JWT_KEY_DIR = Path(os.environ.get("JWT_KEY_DIR", "/srv/keys"))
JWT_ISSUER = "pbl6-platform"
# Secure cookies need HTTPS; the local Docker demo serves plain http on localhost.
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "false").lower() == "true"
REFRESH_COOKIE = "pbl6_refresh"

LOGIN_WINDOW_SECONDS = int(os.environ.get("LOGIN_WINDOW_SECONDS", "900"))
LOGIN_MAX_FAILS_USER_IP = int(os.environ.get("LOGIN_MAX_FAILS_USER_IP", "5"))
LOGIN_MAX_FAILS_IP = int(os.environ.get("LOGIN_MAX_FAILS_IP", "20"))
MIN_PASSWORD_LENGTH = 10

BOOTSTRAP_ADMIN_USERNAME = os.environ.get("BOOTSTRAP_ADMIN_USERNAME", "")
BOOTSTRAP_ADMIN_PASSWORD = os.environ.get("BOOTSTRAP_ADMIN_PASSWORD", "")
