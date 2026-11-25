"""ORM models. Every schema change ships with an Alembic migration in migrations/versions."""

from datetime import datetime, timezone
import uuid

from sqlalchemy import JSON, Boolean, CheckConstraint, DateTime, ForeignKey, Index, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base

ROLES = ("admin", "doctor", "patient")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"
    __table_args__ = (CheckConstraint("role IN ('admin', 'doctor', 'patient')", name="ck_users_role"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    username: Mapped[str] = mapped_column(String(64), unique=True)  # stored lower-case
    display_name: Mapped[str] = mapped_column(String(150))
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(16))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    must_change_password: Mapped[bool] = mapped_column(Boolean, default=True)
    password_changed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class AuthSession(Base):
    """One login. Access tokens carry its id, so revoking it logs the user out at once."""

    __tablename__ = "auth_sessions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoke_reason: Mapped[str | None] = mapped_column(String(32))
    ip_address: Mapped[str | None] = mapped_column(String(64))


class RefreshToken(Base):
    """Opaque refresh token, stored only as a SHA-256 hash; each use replaces it (rotation)."""

    __tablename__ = "refresh_tokens"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("auth_sessions.id", ondelete="CASCADE"), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class LoginAttempt(Base):
    """Failed logins only; counted per (username, ip) and per ip inside a time window."""

    __tablename__ = "login_attempts"
    __table_args__ = (Index("ix_login_attempts_ip_time", "ip_address", "attempted_at"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    username: Mapped[str] = mapped_column(String(64))
    ip_address: Mapped[str] = mapped_column(String(64))
    attempted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    actor_id: Mapped[uuid.UUID | None] = mapped_column(Uuid)
    action: Mapped[str] = mapped_column(String(64))
    resource_type: Mapped[str] = mapped_column(String(32))
    resource_id: Mapped[str | None] = mapped_column(String(64))
    outcome: Mapped[str] = mapped_column(String(16))  # success | failure
    ip_address: Mapped[str | None] = mapped_column(String(64))
    detail: Mapped[dict | None] = mapped_column(JSON)  # never passwords or tokens


# ---------- M2: patients, cases, images, jobs, mask versions, approvals ----------

CASE_STATUSES = ("draft", "ready", "queued", "running", "awaiting_review", "approved", "failed")
JOB_STATES = ("queued", "running", "succeeded", "failed", "cancelled")


class Patient(Base):
    """Pseudonymous: a code, birth year and sex only (no name; decision pending, docs/web/01 section 6)."""

    __tablename__ = "patients"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)  # responsible doctor
    code: Mapped[str] = mapped_column(String(50), unique=True)
    birth_year: Mapped[int | None] = mapped_column()
    sex: Mapped[str | None] = mapped_column(String(1))  # M | F | O
    user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), unique=True)  # patient account (M3)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Case(Base):
    __tablename__ = "cases"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("patients.id"), index=True)
    owner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    study_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    notes: Mapped[str | None] = mapped_column(String(2000))
    status: Mapped[str] = mapped_column(String(20), default="draft")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Volume(Base):
    """One uploaded MRI sequence of a case; the file lives under DATA_DIR."""

    __tablename__ = "volumes"
    __table_args__ = (Index("uq_volumes_case_modality", "case_id", "modality", unique=True),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    case_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"))
    modality: Mapped[str] = mapped_column(String(8))  # dwi | adc | flair
    path: Mapped[str] = mapped_column(String(500))
    sha256: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column()
    shape: Mapped[list] = mapped_column(JSON)
    spacing_mm: Mapped[list] = mapped_column(JSON)
    uploaded_by: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ModelEntry(Base):
    """Segmentation models the GPU worker can run; the worker writes this table when it starts."""

    __tablename__ = "model_registry"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    label: Mapped[str] = mapped_column(String(200))
    modalities: Mapped[list] = mapped_column(JSON)
    required: Mapped[list] = mapped_column(JSON)
    available: Mapped[bool] = mapped_column(Boolean)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    info: Mapped[dict | None] = mapped_column(JSON)  # metrics, note
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class Job(Base):
    """Queue item. The worker claims it with SELECT ... FOR UPDATE SKIP LOCKED (PostgreSQL)."""

    __tablename__ = "jobs"
    __table_args__ = (Index("ix_jobs_state_created", "state", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    case_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(16), default="segment")
    model_name: Mapped[str] = mapped_column(String(64))
    state: Mapped[str] = mapped_column(String(16), default="queued")
    stage: Mapped[str | None] = mapped_column(String(32))
    error: Mapped[str | None] = mapped_column(String(1000))
    created_by: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    worker_id: Mapped[str | None] = mapped_column(String(64))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MaskVersion(Base):
    """Immutable: every AI run or doctor edit adds a version; nothing is overwritten."""

    __tablename__ = "mask_versions"
    __table_args__ = (Index("uq_mask_versions_case_no", "case_id", "version_no", unique=True),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    case_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"))
    version_no: Mapped[int] = mapped_column()
    source: Mapped[str] = mapped_column(String(8))  # ai | doctor
    job_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("jobs.id"))
    model_name: Mapped[str | None] = mapped_column(String(64))
    created_by: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    path: Mapped[str] = mapped_column(String(500))
    stats: Mapped[dict] = mapped_column(JSON)  # volume, lesion count, hemisphere, brain regions...
    note: Mapped[str | None] = mapped_column(String(1000))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Approval(Base):
    """A doctor approves one mask version; only then the patient may see the case (M3)."""

    __tablename__ = "approvals"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    case_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"), index=True)
    mask_version_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("mask_versions.id"))
    approved_by: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    approved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    patient_note: Mapped[str | None] = mapped_column(String(2000))
    show_functions: Mapped[bool] = mapped_column(Boolean, default=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))
