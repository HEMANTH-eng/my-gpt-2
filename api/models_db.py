import datetime
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from api.database import Base


class User(Base):
    """User account ORM database model."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), default="user", nullable=False)  # 'admin', 'user', 'guest'
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    sessions = relationship("ChatSessionDB", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("ApiKeyDB", back_populates="user", cascade="all, delete-orphan")


class ApiKeyDB(Base):
    """API Key ORM database model."""

    __tablename__ = "api_keys"

    id = Column(String(50), primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(100), nullable=False)
    key_hash = Column(String(255), nullable=False, index=True)
    prefix = Column(String(20), nullable=False)
    scopes = Column(String(255), default="read,write", nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="api_keys")


class ChatSessionDB(Base):
    """Chat session ORM database model."""

    __tablename__ = "chat_sessions"

    id = Column(String(50), primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    title = Column(String(255), default="New Chat", nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    user = relationship("User", back_populates="sessions")
    messages = relationship("ChatMessageDB", back_populates="session", cascade="all, delete-orphan")


class ChatMessageDB(Base):
    """Chat message ORM database model."""

    __tablename__ = "chat_messages"

    id = Column(String(50), primary_key=True, index=True)
    session_id = Column(String(50), ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String(20), nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    session = relationship("ChatSessionDB", back_populates="messages")


class UploadedFileDB(Base):
    """Uploaded document metadata ORM database model."""

    __tablename__ = "uploaded_files"

    id = Column(String(50), primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(20), nullable=False)
    file_size = Column(Integer, nullable=False)
    content_text = Column(Text, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.datetime.utcnow)
