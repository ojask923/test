"""Add chatsession and chatmessage

Revision ID: b2c3d4e5f6g7
Revises: 0c8114ab87d6
Create Date: 2026-09-19

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6g7'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create chatsession table if it doesn't exist
    op.create_table('chatsession',
    sa.Column('id', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('title', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    
    # Create chatmessage table if it doesn't exist
    op.create_table('chatmessage',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('session_id', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('role', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('content', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('tool_name', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['session_id'], ['chatsession.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_chatmessage_session_id'), 'chatmessage', ['session_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_chatmessage_session_id'), table_name='chatmessage')
    op.drop_table('chatmessage')
    op.drop_table('chatsession')
