"""Database service for session and message persistence (supports SQLite and PostgreSQL)."""

from typing import List, Optional
from datetime import datetime, timezone
from sqlmodel import Session, create_engine, select
from sqlalchemy import event
from sqlalchemy.pool import QueuePool

from app.config import settings
from app.models.session import ChatSession
from app.models.message import ChatMessage


class DatabaseService:
    """Service handling all database operations for Sessions and Messages."""

    def __init__(self):
        db_url = settings.DATABASE_URL
        connect_args = {}

        # If using SQLite, allow multithreading for FastAPI
        if db_url.startswith("sqlite"):
            connect_args = {"check_same_thread": False}
            self.engine = create_engine(
                db_url,
                connect_args=connect_args,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                echo=False,
            )
            
            @event.listens_for(self.engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.close()
        else:
            # PostgreSQL or MySQL - with safe fallback to SQLite if connection fails
            try:
                self.engine = create_engine(
                    db_url,
                    pool_pre_ping=True,
                    echo=False,
                )
                with self.engine.connect():
                    pass
            except Exception as e:
                print(f"[WARNING] Could not connect to configured DATABASE_URL ({e}). Falling back to SQLite.")
                self.engine = create_engine(
                    "sqlite:///./chatbot.db",
                    connect_args={"check_same_thread": False},
                    poolclass=QueuePool,
                    pool_size=5,
                    max_overflow=10,
                    echo=False,
                )
                
                @event.listens_for(self.engine, "connect")
                def set_fallback_sqlite_pragma(dbapi_connection, connection_record):
                    cursor = dbapi_connection.cursor()
                    cursor.execute("PRAGMA journal_mode=WAL")
                    cursor.execute("PRAGMA synchronous=NORMAL")
                    cursor.close()



    def create_session(self, session_id: str, title: str = "New Chat") -> ChatSession:
        """Create and store a new chat session."""
        with Session(self.engine) as db:
            existing = db.get(ChatSession, session_id)
            if existing:
                return existing
            chat_session = ChatSession(id=session_id, title=title)
            db.add(chat_session)
            db.commit()
            db.refresh(chat_session)
            return chat_session

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Retrieve a specific chat session."""
        with Session(self.engine) as db:
            return db.get(ChatSession, session_id)

    def get_all_sessions(self) -> List[ChatSession]:
        """Get all sessions ordered by updated_at descending."""
        with Session(self.engine) as db:
            statement = select(ChatSession).order_by(ChatSession.updated_at.desc())
            return list(db.exec(statement).all())

    def update_session_title(self, session_id: str, title: str) -> Optional[ChatSession]:
        """Update session title."""
        with Session(self.engine) as db:
            chat_session = db.get(ChatSession, session_id)
            if chat_session:
                chat_session.title = title
                chat_session.updated_at = datetime.now(timezone.utc)
                db.add(chat_session)
                db.commit()
                db.refresh(chat_session)
                return chat_session
            return None

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages."""
        with Session(self.engine) as db:
            chat_session = db.get(ChatSession, session_id)
            if not chat_session:
                return False
            db.delete(chat_session)
            db.commit()
            return True

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_name: Optional[str] = None,
    ) -> ChatMessage:
        """Add a message to a session and touch the session's updated_at."""
        with Session(self.engine) as db:
            # Ensure session exists
            chat_session = db.get(ChatSession, session_id)
            if not chat_session:
                # Default title from first message
                first_title = content[:30] + ("..." if len(content) > 30 else "")
                chat_session = ChatSession(id=session_id, title=first_title)
                db.add(chat_session)
                db.commit()
                db.refresh(chat_session)
            else:
                chat_session.updated_at = datetime.now(timezone.utc)
                # If still "New Chat" and this is a user message, update title
                if chat_session.title in ("New Chat", "") and role == "user":
                    chat_session.title = content[:30] + ("..." if len(content) > 30 else "")
                db.add(chat_session)

            msg = ChatMessage(
                session_id=session_id,
                role=role,
                content=content,
                tool_name=tool_name,
            )
            db.add(msg)
            db.commit()
            db.refresh(msg)
            return msg

    def get_session_messages(self, session_id: str) -> List[ChatMessage]:
        """Get all messages for a session ordered by creation time."""
        with Session(self.engine) as db:
            statement = (
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.asc())
            )
            return list(db.exec(statement).all())

    def clear_session_messages(self, session_id: str) -> bool:
        """Clear all messages in a session."""
        with Session(self.engine) as db:
            statement = select(ChatMessage).where(ChatMessage.session_id == session_id)
            messages = db.exec(statement).all()
            for msg in messages:
                db.delete(msg)
            db.commit()
            return True

    def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            with Session(self.engine) as db:
                db.exec(select(1)).first()
                return True
        except Exception:
            return False


db_service = DatabaseService()
