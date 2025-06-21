#!/usr/bin/env python3
"""
AI PORTAL FOUNDATION BUNDLE - PHASE 1 MODULARIZATION (FIXED)
Master script to create complete foundation bundle with all modules

BATCH CREATION: Creates all foundation files in one execution
- schemas/ folder with Pydantic models
- models/ folder with SQLAlchemy models  
- decorators/ folder with utility decorators

RISK LEVEL: 🟢 LOW RISK (Safe components only)
VERSION: AI Portal v26.2.0 Foundation Bundle (DEPENDENCY-FREE)
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

def backup_original_main():
    """Backup original main.py before modifying"""
    main_file = Path("main.py")
    if main_file.exists():
        backup_name = f"main_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
        backup_path = Path(backup_name)
        shutil.copy2(main_file, backup_path)
        print(f"✅ Original main.py backed up as: {backup_path}")
        return backup_path
    else:
        print("⚠️  main.py not found - no backup needed")
        return None

def create_schemas_module():
    """Create complete schemas module"""
    print("\n📦 Creating schemas module...")
    
    schemas_dir = Path("schemas")
    schemas_dir.mkdir(exist_ok=True)
    
    # schemas/__init__.py
    init_content = '''# AI Portal Schemas Module
# Pydantic models for request/response validation

from .schemas import (
    ProjectRequest, ProjectResponse,
    PersonaRequest, PersonaResponse,
    ChatRequest, ChatCompletionResponse, ChatResponse,
    ObjectiveRequest
)

__all__ = [
    "ProjectRequest", "ProjectResponse", 
    "PersonaRequest", "PersonaResponse",
    "ChatRequest", "ChatCompletionResponse", "ChatResponse",
    "ObjectiveRequest"
]
'''
    
    # schemas/schemas.py
    schemas_content = '''# AI Portal Pydantic Schemas
# Extracted from main.py v26.2.0 - COMPLETE ORIGINAL FUNCTIONALITY

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, field_validator

class ProjectRequest(BaseModel):
    name: str
    description: Optional[str] = None
    user_id: str = "anonymous"

class ProjectResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    user_id: str
    created_at: datetime
    updated_at: datetime

class PersonaRequest(BaseModel):
    name: str
    system_prompt: str
    model_preference: Optional[str] = None
    user_id: str = "anonymous"

class PersonaResponse(BaseModel):
    id: str
    name: str
    system_prompt: str
    model_preference: Optional[str]
    user_id: str

class ChatRequest(BaseModel):
    message: str
    user_id: str = "anonymous"
    project_id: Optional[str] = None
    persona_id: Optional[str] = None
    task_type: str = "auto"
    user_tier: str = "free"

class ChatCompletionResponse(BaseModel):
    type: str
    response: str

class ChatResponse(BaseModel):
    success: bool
    response: str
    response_type: str
    model: str
    reasoning: str
    project_id: str

class ObjectiveRequest(BaseModel):
    objective: str
    user_id: str = "anonymous"
    project_id: Optional[str] = None
'''
    
    # Write files
    (schemas_dir / "__init__.py").write_text(init_content, encoding='utf-8')
    (schemas_dir / "schemas.py").write_text(schemas_content, encoding='utf-8')
    
    print(f"   ✅ {schemas_dir}/__init__.py")
    print(f"   ✅ {schemas_dir}/schemas.py")

def create_models_module():
    """Create complete models module"""
    print("\n📦 Creating models module...")
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # models/__init__.py
    init_content = '''# AI Portal Models Module
# SQLAlchemy database models

from .models import Base, Project, Persona, ChatHistory, ProjectSettings

__all__ = ["Base", "Project", "Persona", "ChatHistory", "ProjectSettings"]
'''
    
    # models/models.py
    models_content = '''# AI Portal Database Models
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
'''
    
    # Write files
    (models_dir / "__init__.py").write_text(init_content, encoding='utf-8')
    (models_dir / "models.py").write_text(models_content, encoding='utf-8')
    
    print(f"   ✅ {models_dir}/__init__.py")
    print(f"   ✅ {models_dir}/models.py")

def create_decorators_module():
    """Create complete decorators module"""
    print("\n📦 Creating decorators module...")
    
    decorators_dir = Path("decorators")
    decorators_dir.mkdir(exist_ok=True)
    
    # decorators/__init__.py
    init_content = '''# AI Portal Decorators Module
# Utility decorators for resilience and error handling

from .decorators import async_retry_with_backoff, F

__all__ = ["async_retry_with_backoff", "F"]
'''
    
    # decorators/decorators.py
    decorators_content = '''# AI Portal Utility Decorators
# Extracted from main.py v26.2.0 - COMPLETE ORIGINAL FUNCTIONALITY

import asyncio
from functools import wraps
from typing import Callable, Any, TypeVar
import aiohttp
import structlog

logger = structlog.get_logger()
F = TypeVar('F', bound=Callable[..., Any])

def async_retry_with_backoff(retries: int = 4, initial_delay: float = 1.0, backoff: float = 2.0):
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            for i in range(retries):
                try:
                    return await func(*args, **kwargs)
                except aiohttp.ClientResponseError as e:
                    if e.status in [429, 500, 502, 503, 504] and i < retries - 1:
                        logger.warning("API call failed, retrying...", 
                                     attempt=i + 1, delay=delay, status=e.status)
                        await asyncio.sleep(delay)
                        delay *= backoff
                    else: 
                        raise
                except Exception: 
                    raise
        return wrapper
    return decorator
'''
    
    # Write files
    (decorators_dir / "__init__.py").write_text(init_content, encoding='utf-8')
    (decorators_dir / "decorators.py").write_text(decorators_content, encoding='utf-8')
    
    print(f"   ✅ {decorators_dir}/__init__.py")
    print(f"   ✅ {decorators_dir}/decorators.py")

def run_basic_tests():
    """Test created modules (NO HEAVY DEPENDENCIES)"""
    print("\n🧪 Testing created modules...")
    
    try:
        # Test schemas (basic pydantic - should work)
        print("   🔍 Testing schemas...")
        sys.path.insert(0, str(Path.cwd()))
        
        from schemas import ProjectRequest, ChatRequest
        test_project = ProjectRequest(name="Test", user_id="test")
        print("   ✅ Schemas module working")
        
        # Test models (basic SQLAlchemy - should work)
        print("   🔍 Testing models...")
        from models import Base, Project, Persona
        print(f"   ✅ Models module working ({len(Base.metadata.tables)} tables)")
        
        # Test decorators (basic typing - should work)
        print("   🔍 Testing decorators...")
        from decorators import async_retry_with_backoff
        print("   ✅ Decorators module working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        print(f"   📝 Error details: {type(e).__name__}")
        return False

def main():
    """Create complete AI Portal Foundation Bundle (DEPENDENCY-FREE)"""
    print("🚀 AI PORTAL FOUNDATION BUNDLE CREATION (FIXED)")
    print("=" * 60)
    print("PHASE 1: Foundation Bundle (Low Risk Components)")
    print("Creating: schemas, models, decorators modules")
    print("FIXED: No heavy dependencies (sentence-transformers, etc.)")
    print("=" * 60)
    
    # Backup original
    backup_path = backup_original_main()
    
    # Create all modules
    create_schemas_module()
    create_models_module() 
    create_decorators_module()
    
    # Test modules (basic tests only)
    success = run_basic_tests()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ FOUNDATION BUNDLE CREATION COMPLETE!")
        print("\n📁 Created modules:")
        print("   • schemas/ - Pydantic request/response models")
        print("   • models/ - SQLAlchemy database models")
        print("   • decorators/ - Utility decorators")
        
        if backup_path:
            print(f"\n💾 Backup created: {backup_path}")
        
        print("\n🎯 READY FOR PHASE 2: Core Services (ConfigManager, ToolService, etc.)")
        print("💡 MemoryService will be handled in Phase 2 (has heavy dependencies)")
        
    else:
        print("❌ FOUNDATION BUNDLE CREATION FAILED!")
        print("Check error messages above and retry")
    
    print("=" * 60)

if __name__ == "__main__":
    main()