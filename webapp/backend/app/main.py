"""PBL6 platform API (phase 7). Run: uvicorn app.main:app (from webapp/backend)."""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException
from sqlalchemy import func, select, text

from app import config
from app.db import SessionLocal, engine
from app.models import User
from app.routers import admin, auth
from app.security import hash_password

log = logging.getLogger("pbl6")


def bootstrap_admin() -> None:
    """First start only: create the admin from BOOTSTRAP_ADMIN_* when the users table is empty."""
    if not (config.BOOTSTRAP_ADMIN_USERNAME and config.BOOTSTRAP_ADMIN_PASSWORD):
        return
    with SessionLocal() as db:
        if db.scalar(select(func.count()).select_from(User)):
            return
        db.add(User(username=config.BOOTSTRAP_ADMIN_USERNAME.lower(), display_name="Administrator", role="admin",
                    password_hash=hash_password(config.BOOTSTRAP_ADMIN_PASSWORD), must_change_password=True))
        db.commit()
        log.warning("Bootstrap admin '%s' created; the password must be changed at first login.",
                    config.BOOTSTRAP_ADMIN_USERNAME)


@asynccontextmanager
async def lifespan(_: FastAPI):
    bootstrap_admin()
    yield


app = FastAPI(title="PBL6 Stroke Platform API", version="0.2.0", lifespan=lifespan)
app.include_router(auth.router)
app.include_router(admin.router)


@app.get("/api/v1/health/live")
def live():
    return {"status": "ok"}


@app.get("/api/v1/health/ready")
def ready():
    """Ready only when the database answers; docker compose healthcheck uses this."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        raise HTTPException(503, f"database unavailable: {type(e).__name__}")
    return {"status": "ready"}
