# AI Portal Database Models
# Extracted from main.py v26.2.0 - COMPLETE ORIGINAL FUNCTIONALITY

import uuid
from sqlalchemy import Column, String, Integer, DateTime, Text, Float, ForeignKey, JSON, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()

class Project(Base):
    __tablename__ = 'projects'
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    user_id = Column(String(100), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    chat_history = relationship("ChatHistory", back_populates="project", cascade="all, delete-orphan")

class Persona(Base):
    __tablename__ = 'personas'
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    system_prompt = Column(Text, nullable=False)
    model_preference = Column(String(100))
    user_id = Column(String(100), nullable=False, index=True)
    __table_args__ = (UniqueConstraint('user_id', 'name', name='_user_id_name_uc'),)
    chat_history = relationship("ChatHistory", back_populates="persona")

class ChatHistory(Base):
    __tablename__ = 'chat_history'
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(PG_UUID(as_uuid=True), ForeignKey('projects.id', ondelete='CASCADE'), nullable=False)
    persona_id = Column(PG_UUID(as_uuid=True), ForeignKey('personas.id', ondelete='SET NULL'), nullable=True)
    user_id = Column(String(100), nullable=False)
    message = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    response_type = Column(String(50), default='text')
    model_used = Column(String(100), nullable=False)
    cost = Column(Float, default=0.0)
    response_time = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    project = relationship("Project", back_populates="chat_history")
    persona = relationship("Persona", back_populates="chat_history")

class ProjectSettings(Base):
    __tablename__ = 'project_settings'
    project_id = Column(PG_UUID(as_uuid=True), ForeignKey('projects.id', ondelete='CASCADE'), primary_key=True)
    active_persona_id = Column(PG_UUID(as_uuid=True), ForeignKey('personas.id', ondelete='SET NULL'), nullable=True)
    context_length = Column(Integer, default=10)
    settings = Column(JSON, default={})
