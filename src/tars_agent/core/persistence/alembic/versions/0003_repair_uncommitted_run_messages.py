"""Remove unsuccessful Run messages from the active model context.

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-13
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Early legacy imports marked every thread row committed/active, including
    # partial assistant/tool messages from failed or incomplete Runs.  The
    # durable-runtime invariant is stricter: only a succeeded Run may commit its
    # messages to the formal context.  This statement is intentionally broad so
    # it also repairs databases imported before migration markers carried enough
    # structured provenance to identify individual message rows.
    op.execute(
        sa.text(
            """
            UPDATE messages
               SET committed = 0,
                   active = 0
             WHERE run_id IN (
                 SELECT id
                   FROM runs
                  WHERE status != 'succeeded'
             )
            """
        )
    )


def downgrade() -> None:
    # Irreversible data repair: re-activating partial failed output would violate
    # the context invariant and Alembic cannot recover the prior intent safely.
    pass
