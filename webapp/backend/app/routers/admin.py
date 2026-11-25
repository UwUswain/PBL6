"""Admin: user accounts and the audit log. Accounts are only created here (no self sign-up)."""

from datetime import datetime, timezone
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.deps import Principal, audit, get_db, require_roles
from app.models import AuditEvent, AuthSession, User
from app.schemas import AuditEventOut, UserCreatedOut, UserCreateIn, UserOut, UserUpdateIn
from app.security import hash_password, temporary_password

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])
admin_only = require_roles("admin")


def _revoke_all(db: Session, user_id: uuid.UUID, reason: str) -> None:
    db.execute(update(AuthSession).where(AuthSession.user_id == user_id, AuthSession.revoked_at.is_(None))
               .values(revoked_at=datetime.now(timezone.utc), revoke_reason=reason))


def _get_user(db: Session, user_id: uuid.UUID) -> User:
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(404, "User not found")
    return user


@router.get("/users", response_model=list[UserOut])
def list_users(_: Principal = Depends(admin_only), db: Session = Depends(get_db)):
    return db.scalars(select(User).order_by(User.role, User.username)).all()


@router.post("/users", response_model=UserCreatedOut, status_code=201)
def create_user(body: UserCreateIn, request: Request, p: Principal = Depends(admin_only), db: Session = Depends(get_db)):
    username = body.username.lower()
    if db.scalar(select(User.id).where(User.username == username)):
        raise HTTPException(409, "Username already exists")
    temp = temporary_password()
    user = User(username=username, display_name=body.display_name.strip(), role=body.role,
                password_hash=hash_password(temp), must_change_password=True)
    db.add(user)
    db.flush()
    audit(db, request, "admin.user_create", "user", actor_id=p.user.id, resource_id=str(user.id),
          detail={"username": username, "role": body.role})
    db.commit()
    return UserCreatedOut(user=UserOut.model_validate(user), temporary_password=temp)


@router.patch("/users/{user_id}", response_model=UserOut)
def update_user(user_id: uuid.UUID, body: UserUpdateIn, request: Request, p: Principal = Depends(admin_only),
                db: Session = Depends(get_db)):
    user = _get_user(db, user_id)
    if user.id == p.user.id and (body.is_active is False or (body.role and body.role != "admin")):
        raise HTTPException(400, "You cannot disable or demote your own account")
    changes = body.model_dump(exclude_none=True)
    for field, value in changes.items():
        setattr(user, field, value)
    if body.is_active is False:
        _revoke_all(db, user.id, "account_disabled")
    audit(db, request, "admin.user_update", "user", actor_id=p.user.id, resource_id=str(user.id), detail=changes)
    db.commit()
    return user


@router.post("/users/{user_id}/reset-password", response_model=UserCreatedOut)
def reset_password(user_id: uuid.UUID, request: Request, p: Principal = Depends(admin_only), db: Session = Depends(get_db)):
    user = _get_user(db, user_id)
    temp = temporary_password()
    user.password_hash = hash_password(temp)
    user.must_change_password = True
    _revoke_all(db, user.id, "password_reset")
    audit(db, request, "admin.password_reset", "user", actor_id=p.user.id, resource_id=str(user.id))
    db.commit()
    return UserCreatedOut(user=UserOut.model_validate(user), temporary_password=temp)


@router.get("/audit-events", response_model=list[AuditEventOut])
def audit_events(limit: int = 100, offset: int = 0, _: Principal = Depends(admin_only), db: Session = Depends(get_db)):
    limit = max(1, min(limit, 500))
    return db.scalars(select(AuditEvent).order_by(AuditEvent.created_at.desc()).limit(limit).offset(max(0, offset))).all()
