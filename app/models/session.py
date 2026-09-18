"""Chat session database model using SQLModel."""

from datetime import datetime, timezone
from typing import List
from sqlmodel import SQLModel, Field, Relationship


class ChatSession(SQLModel, table=True):
    """Session representation in the database."""

    id: str = Field(default=None, primary_key=True)
    title: str = Field(default="New Chat")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Relationships
    messages: List["ChatMessage"] = Relationship(
        back_populates="session",
        sa_relationship_kwargs={"cascade": "all, delete-orphan", "lazy": "selectin"},
    )
