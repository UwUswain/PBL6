"""Database engine and session factory. DATABASE_URL comes from the environment (docker compose sets it)."""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import StaticPool

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg://pbl6:pbl6@localhost:5432/pbl6")

# In-memory SQLite (tests) needs one shared connection, or every session would see an empty database.
_sqlite = DATABASE_URL.startswith("sqlite")
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=not _sqlite,
    **({"connect_args": {"check_same_thread": False}, "poolclass": StaticPool} if _sqlite else {}),
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass
