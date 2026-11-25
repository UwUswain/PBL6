"""Login, token refresh (rotation + reuse detection), logout, current user, password change."""

from datetime import datetime, timedelta, timezone
import math

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app import config
from app.deps import Principal, _aware, audit, client_ip, current_principal_any, get_db
from app.models import AuthSession, LoginAttempt, RefreshToken, User
from app.schemas import ChangePasswordIn, LoginIn, TokenOut, UserOut
from app.security import create_access_token, hash_password, hash_refresh_token, new_refresh_token, verify_password

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])
COOKIE_PATH = "/api/v1/auth"


def _set_refresh_cookie(response: Response, raw: str) -> None:
    response.set_cookie(config.REFRESH_COOKIE, raw, max_age=config.REFRESH_TTL_DAYS * 86400, path=COOKIE_PATH,
                        httponly=True, secure=config.COOKIE_SECURE, samesite="strict")


def _clear_refresh_cookie(response: Response) -> None:
    response.delete_cookie(config.REFRESH_COOKIE, path=COOKIE_PATH, httponly=True,
                           secure=config.COOKIE_SECURE, samesite="strict")


def _token_out(user: User, session_id) -> TokenOut:
    token = create_access_token(user.id, user.role, session_id, user.must_change_password)
    return TokenOut(access_token=token, expires_in=config.ACCESS_TTL_SECONDS, user=UserOut.model_validate(user))


def _retry_after(db: Session, username: str, ip: str, now: datetime) -> int | None:
    """Seconds to wait when too many recent failures, else None."""
    since = now - timedelta(seconds=config.LOGIN_WINDOW_SECONDS)
    waits = []
    for where, limit in (((LoginAttempt.username == username, LoginAttempt.ip_address == ip), config.LOGIN_MAX_FAILS_USER_IP),
                         ((LoginAttempt.ip_address == ip,), config.LOGIN_MAX_FAILS_IP)):
        count, oldest = db.execute(
            select(func.count(), func.min(LoginAttempt.attempted_at)).where(LoginAttempt.attempted_at > since, *where)
        ).one()
        if count >= limit:
            waits.append(math.ceil((_aware(oldest) + timedelta(seconds=config.LOGIN_WINDOW_SECONDS) - now).total_seconds()))
    return max(1, max(waits)) if waits else None


@router.post("/login", response_model=TokenOut)
def login(body: LoginIn, request: Request, response: Response, db: Session = Depends(get_db)):
    username = body.username.strip().lower()
    ip = client_ip(request)
    now = datetime.now(timezone.utc)
    wait = _retry_after(db, username, ip, now)
    if wait:
        audit(db, request, "auth.login_rate_limited", "auth", "failure", detail={"username": username})
        db.commit()
        raise HTTPException(429, "Too many failed attempts. Try again later.", headers={"Retry-After": str(wait)})

    user = db.scalar(select(User).where(User.username == username))
    if not verify_password(user.password_hash if user else None, body.password) or not user.is_active:
        db.add(LoginAttempt(username=username, ip_address=ip, attempted_at=now))
        audit(db, request, "auth.login", "auth", "failure", actor_id=user.id if user else None, detail={"username": username})
        db.commit()
        raise HTTPException(401, "Invalid username or password")  # same message whether the user exists or not

    session = AuthSession(user_id=user.id, expires_at=now + timedelta(days=config.REFRESH_TTL_DAYS), ip_address=ip)
    db.add(session)
    db.flush()
    raw, digest = new_refresh_token()
    db.add(RefreshToken(session_id=session.id, token_hash=digest))
    audit(db, request, "auth.login", "auth", actor_id=user.id)
    db.commit()
    _set_refresh_cookie(response, raw)
    return _token_out(user, session.id)


def _end_session(db: Session, session: AuthSession, reason: str) -> None:
    if session.revoked_at is None:
        session.revoked_at = datetime.now(timezone.utc)
        session.revoke_reason = reason


@router.post("/refresh", response_model=TokenOut)
def refresh(request: Request, response: Response, db: Session = Depends(get_db)):
    raw = request.cookies.get(config.REFRESH_COOKIE)
    if not raw:
        raise HTTPException(401, "No refresh token")
    token = db.scalar(select(RefreshToken).where(RefreshToken.token_hash == hash_refresh_token(raw)).with_for_update())
    session = db.get(AuthSession, token.session_id) if token else None
    now = datetime.now(timezone.utc)
    if token is None or session is None or session.revoked_at is not None or _aware(session.expires_at) <= now:
        _clear_refresh_cookie(response)
        raise HTTPException(401, "Session ended")
    if token.used_at is not None:
        # A rotated token came back: someone else may hold a copy, so end the whole session.
        _end_session(db, session, "refresh_reused")
        audit(db, request, "auth.refresh_reused", "session", "failure", actor_id=session.user_id, resource_id=str(session.id))
        db.commit()
        _clear_refresh_cookie(response)
        raise HTTPException(401, "Session ended")
    user = db.get(User, session.user_id)
    if user is None or not user.is_active:
        _end_session(db, session, "account_disabled")
        db.commit()
        _clear_refresh_cookie(response)
        raise HTTPException(401, "Account disabled")

    token.used_at = now
    raw_new, digest = new_refresh_token()
    db.add(RefreshToken(session_id=session.id, token_hash=digest))
    db.commit()
    _set_refresh_cookie(response, raw_new)
    return _token_out(user, session.id)


@router.post("/logout", status_code=204)
def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    raw = request.cookies.get(config.REFRESH_COOKIE)
    token = db.scalar(select(RefreshToken).where(RefreshToken.token_hash == hash_refresh_token(raw))) if raw else None
    if token:
        session = db.get(AuthSession, token.session_id)
        if session:
            _end_session(db, session, "logout")
            audit(db, request, "auth.logout", "session", actor_id=session.user_id, resource_id=str(session.id))
            db.commit()
    _clear_refresh_cookie(response)
    response.status_code = 204
    return response


@router.get("/me", response_model=UserOut)
def me(p: Principal = Depends(current_principal_any)):
    return p.user


@router.post("/change-password", response_model=TokenOut)
def change_password(body: ChangePasswordIn, request: Request, p: Principal = Depends(current_principal_any),
                    db: Session = Depends(get_db)):
    user = db.get(User, p.user.id)
    if not verify_password(user.password_hash, body.current_password):
        audit(db, request, "auth.change_password", "user", "failure", actor_id=user.id, resource_id=str(user.id))
        db.commit()
        raise HTTPException(400, "Current password is wrong")
    new = body.new_password
    if len(new) < config.MIN_PASSWORD_LENGTH or new == body.current_password or user.username in new.lower():
        raise HTTPException(400, f"New password needs at least {config.MIN_PASSWORD_LENGTH} characters, "
                                 "must differ from the current one and must not contain the username")
    now = datetime.now(timezone.utc)
    user.password_hash = hash_password(new)
    user.must_change_password = False
    user.password_changed_at = now
    # Other devices are logged out; this session stays.
    db.execute(update(AuthSession)
               .where(AuthSession.user_id == user.id, AuthSession.id != p.session_id, AuthSession.revoked_at.is_(None))
               .values(revoked_at=now, revoke_reason="password_changed"))
    audit(db, request, "auth.change_password", "user", actor_id=user.id, resource_id=str(user.id))
    db.commit()
    return _token_out(user, p.session_id)
