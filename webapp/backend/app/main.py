"""PBL6 platform API (phase 7). Run: uvicorn app.main:app (from webapp/backend)."""

from fastapi import FastAPI, HTTPException
from sqlalchemy import text

from app.db import engine

app = FastAPI(title="PBL6 Stroke Platform API", version="0.1.0")


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
