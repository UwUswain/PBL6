from datetime import datetime
import uuid

from pydantic import BaseModel, ConfigDict, Field


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    username: str
    display_name: str
    role: str
    is_active: bool
    must_change_password: bool
    created_at: datetime


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserOut


class LoginIn(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class ChangePasswordIn(BaseModel):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=1, max_length=256)


class UserCreateIn(BaseModel):
    username: str = Field(pattern=r"^[A-Za-z0-9._-]{3,64}$")
    display_name: str = Field(min_length=1, max_length=150)
    role: str = Field(pattern=r"^(admin|doctor|patient)$")


class UserUpdateIn(BaseModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=150)
    role: str | None = Field(default=None, pattern=r"^(admin|doctor|patient)$")
    is_active: bool | None = None


class UserCreatedOut(BaseModel):
    user: UserOut
    temporary_password: str  # shown once; only its hash is stored


class AuditEventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    created_at: datetime
    actor_id: uuid.UUID | None
    action: str
    resource_type: str
    resource_id: str | None
    outcome: str
    ip_address: str | None
    detail: dict | None
