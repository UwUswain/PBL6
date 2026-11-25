"""Baseline: empty schema, so later migrations have a fixed starting point.

Revision ID: 0001_baseline
Revises:
Create Date: 2026-11-25
"""

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
