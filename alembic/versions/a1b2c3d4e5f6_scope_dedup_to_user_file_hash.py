"""scope_dedup_to_user_file_hash

Revision ID: a1b2c3d4e5f6
Revises: 93254c8fe403
Create Date: 2026-09-19

Changes
-------
* Drop the global unique index on ``ingesteddocument.file_hash``
  (prevented two different users from uploading byte-identical files).
* Drop the global unique index on ``ingesteddocument.document_id``
  (document_id is a random UUID4; collisions are astronomically unlikely and
  the primary-key index already enforces row-level uniqueness).
* Re-create both as plain (non-unique) indexes so per-column lookups remain fast.
* Add a composite unique constraint on ``(user_id, file_hash)`` so that
  deduplication is scoped per-user: the same file content uploaded by the
  same user is still rejected as a duplicate, but two different users can
  ingest the same bytes independently and each receive their own vectors
  tagged with their user_id.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "93254c8fe403"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Replace per-column unique indexes with a composite (user_id, file_hash) one."""
    # 1. Drop the unique index on file_hash (was created by SQLModel's
    #    Field(index=True, sa_column_kwargs={"unique": True}), named ix_...)
    op.drop_index("ix_ingesteddocument_file_hash", table_name="ingesteddocument")

    # 2. Drop the unique index on document_id for the same reason.
    op.drop_index("ix_ingesteddocument_document_id", table_name="ingesteddocument")

    # 3. Re-create both as plain (non-unique) indexes.
    op.create_index(
        "ix_ingesteddocument_file_hash",
        "ingesteddocument",
        ["file_hash"],
        unique=False,
    )
    op.create_index(
        "ix_ingesteddocument_document_id",
        "ingesteddocument",
        ["document_id"],
        unique=False,
    )

    # 4. Add the composite unique constraint scoping dedup to (user_id, file_hash).
    op.create_unique_constraint(
        "uq_ingesteddocument_user_file_hash",
        "ingesteddocument",
        ["user_id", "file_hash"],
    )


def downgrade() -> None:
    """Restore per-column unique indexes and remove the composite constraint."""
    op.drop_constraint(
        "uq_ingesteddocument_user_file_hash",
        "ingesteddocument",
        type_="unique",
    )
    op.drop_index("ix_ingesteddocument_file_hash", table_name="ingesteddocument")
    op.drop_index("ix_ingesteddocument_document_id", table_name="ingesteddocument")

    op.create_index(
        "ix_ingesteddocument_file_hash",
        "ingesteddocument",
        ["file_hash"],
        unique=True,
    )
    op.create_index(
        "ix_ingesteddocument_document_id",
        "ingesteddocument",
        ["document_id"],
        unique=True,
    )
