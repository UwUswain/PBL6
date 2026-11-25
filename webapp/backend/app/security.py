"""Password hashing (Argon2id), RS256 JWT access tokens, opaque refresh tokens."""

from datetime import datetime, timedelta, timezone
from functools import lru_cache
import hashlib
import secrets
import uuid

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import jwt

from app import config

_hasher = PasswordHasher()  # argon2-cffi defaults are Argon2id with RFC 9106 "low memory" parameters
# Verifying against this when the username is unknown keeps response time the same (no user enumeration).
_DUMMY_HASH = _hasher.hash("dummy-password-for-timing")


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str | None, password: str) -> bool:
    try:
        return _hasher.verify(password_hash or _DUMMY_HASH, password) and password_hash is not None
    except (VerifyMismatchError, InvalidHashError):
        return False


@lru_cache(maxsize=1)
def _keys() -> tuple[bytes, bytes]:
    """RSA key pair kept in JWT_KEY_DIR (a Docker volume), created on first start."""
    key_dir = config.JWT_KEY_DIR
    private_path = key_dir / "jwt_private.pem"
    if not private_path.exists():
        key_dir.mkdir(parents=True, exist_ok=True)
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
        private_path.write_bytes(pem)
        private_path.chmod(0o600)
    private_pem = private_path.read_bytes()
    public_pem = serialization.load_pem_private_key(private_pem, password=None).public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return private_pem, public_pem


def create_access_token(user_id: uuid.UUID, role: str, session_id: uuid.UUID, must_change: bool) -> str:
    now = datetime.now(timezone.utc)
    claims = {
        "iss": config.JWT_ISSUER,
        "sub": str(user_id),
        "role": role,
        "sid": str(session_id),
        "mcp": must_change,
        "iat": now,
        "exp": now + timedelta(seconds=config.ACCESS_TTL_SECONDS),
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(claims, _keys()[0], algorithm="RS256")


def decode_access_token(token: str) -> dict:
    """Raises jwt.PyJWTError when the token is invalid or expired."""
    return jwt.decode(token, _keys()[1], algorithms=["RS256"], issuer=config.JWT_ISSUER,
                      options={"require": ["exp", "sub", "sid", "iss"]})


def new_refresh_token() -> tuple[str, str]:
    """(raw token for the cookie, SHA-256 hash for the database)."""
    raw = secrets.token_urlsafe(32)
    return raw, hash_refresh_token(raw)


def hash_refresh_token(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def temporary_password() -> str:
    return secrets.token_urlsafe(9)  # 12 chars; the user must change it at first login
