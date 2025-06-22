"""
SAVE AS: ai_portal/dependencies.py
CREATE THIS NEW FILE IN THE AI_PORTAL ROOT FOLDER

FastAPI Dependencies Module - CRITICAL MISSING COMPONENT
Provides dependency injection for database sessions, orchestration engine, and memory service
This file was missing and causing import errors in main.py
"""

import structlog
from typing import Generator
from sqlalchemy.orm import Session

logger = structlog.get_logger()

# Global references to be injected by the main application
_database_manager = None
_orchestration_engine = None
_memory_service = None

def inject_database_manager(database_manager):
    """Inject database manager instance"""
    global _database_manager
    _database_manager = database_manager
    logger.debug("Database manager injected into dependencies")

def inject_orchestration_engine(orchestration_engine):
    """Inject orchestration engine instance"""
    global _orchestration_engine
    _orchestration_engine = orchestration_engine
    logger.debug("Orchestration engine injected into dependencies")

def inject_memory_service(memory_service):
    """Inject memory service instance"""
    global _memory_service
    _memory_service = memory_service
    logger.debug("Memory service injected into dependencies")

def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency to get database session
    Used by FastAPI's dependency injection system
    """
    if not _database_manager:
        raise RuntimeError("Database manager not injected. Call inject_database_manager() first.")
    
    with _database_manager.get_session() as session:
        yield session

def get_orchestration_engine():
    """
    FastAPI dependency to get orchestration engine
    Used by FastAPI's dependency injection system
    """
    if not _orchestration_engine:
        raise RuntimeError("Orchestration engine not injected. Call inject_orchestration_engine() first.")
    
    return _orchestration_engine

def get_memory_service():
    """
    FastAPI dependency to get memory service
    Used by FastAPI's dependency injection system
    """
    # Memory service is optional, can be None
    return _memory_service

def get_db_session():
    """
    Alternative database session provider for backwards compatibility
    """
    if not _database_manager:
        raise RuntimeError("Database manager not injected. Call inject_database_manager() first.")
    
    return _database_manager.get_session()

# Health check function
def check_dependencies() -> dict:
    """Check status of all injected dependencies"""
    return {
        "database_manager": _database_manager is not None,
        "orchestration_engine": _orchestration_engine is not None,
        "memory_service": _memory_service is not None
    }