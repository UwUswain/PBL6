"""Request dependencies: DB session, current user from the Bearer token, role checks, audit log."""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
import uuid

from fastapi import Depends, HTTPException, Request
import jwt
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models import AuditEvent, AuthSession, User
from app.security import decode_access_token


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def client_ip(request: Request) -> str:
    # nginx sets X-Forwarded-For; only trust its first hop because the API port is not published.
    forwarded = request.headers.get("x-forwarded-for")
    return (forwarded.split(",")[0].strip() if forwarded else None) or (request.client.host if request.client else "unknown")


def audit(db: Session, request: Request, action: str, resource_type: str, outcome: str = "success",
          actor_id: uuid.UUID | None = None, resource_id: str | None = None, detail: dict | None = None) -> None:
    db.add(AuditEvent(actor_id=actor_id, action=action, resource_type=resource_type, resource_id=resource_id,
                      outcome=outcome, ip_address=client_ip(request), detail=detail))


@dataclass
class Principal:
    user: User
    session_id: uuid.UUID


def _principal(request: Request, db: Session) -> Principal:
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        raise HTTPException(401, "Not authenticated", headers={"WWW-Authenticate": "Bearer"})
    try:
        claims = decode_access_token(header[7:])
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid or expired token", headers={"WWW-Authenticate": "Bearer"})
    # The session lookup makes logout, password change and account lock take effect immediately.
    session = db.get(AuthSession, uuid.UUID(claims["sid"]))
    now = datetime.now(timezone.utc)
    if session is None or session.revoked_at is not None or _aware(session.expires_at) <= now:
        raise HTTPException(401, "Session ended", headers={"WWW-Authenticate": "Bearer"})
    user = db.get(User, uuid.UUID(claims["sub"]))
    if user is None or not user.is_active:
        raise HTTPException(401, "Account disabled", headers={"WWW-Authenticate": "Bearer"})
    return Principal(user, session.id)


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)  # SQLite returns naive datetimes


def current_principal_any(request: Request, db: Session = Depends(get_db)) -> Principal:
    """Authenticated, even if the password must still be changed (for /me, change-password, logout)."""
    return _principal(request, db)


def current_principal(request: Request, db: Session = Depends(get_db)) -> Principal:
    p = _principal(request, db)
    if p.user.must_change_password:
        raise HTTPException(403, "Password change required")
    return p


def require_roles(*roles: str):
    def check(p: Principal = Depends(current_principal)) -> Principal:
        if p.user.role not in roles:
            raise HTTPException(403, "Forbidden")
        return p
    return check
