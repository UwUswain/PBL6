from logging.config import fileConfig

from alembic import context

from app.db import Base, engine
import app.models  # noqa: F401  # registers every table; without it autogenerate sees none and emits DROP TABLE

if context.config.config_file_name:
    fileConfig(context.config.config_file_name)

target_metadata = Base.metadata


def run_migrations_online() -> None:
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    raise SystemExit("Offline migrations are not supported; run against a live database.")
run_migrations_online()
