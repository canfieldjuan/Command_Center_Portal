"""
SQLAlchemy base setup and database configuration - COMPLETE VERSION
Extracted from main.py - ALL original functionality preserved
"""

import os
import urllib.parse
import structlog
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager

logger = structlog.get_logger()

# Create declarative base
Base = declarative_base()

def get_database_url():
    """Get database URL from environment variables with validation"""
    db_password = os.environ.get("SUPABASE_PASSWORD")
    if not db_password:
        raise ValueError("SUPABASE_PASSWORD environment variable not found. Please set it in your .env file.")
    
    encoded_password = urllib.parse.quote(db_password, safe='')
    database_url = f"postgresql://postgres.jacjorrzxilmrfxbdyse:{encoded_password}@aws-0-us-east-2.pooler.supabase.com:6543/postgres"
    
    logger.info("Database URL configured", url_masked=database_url[:50] + "...")
    return database_url

# Create engine with comprehensive configuration
engine = create_engine(
    get_database_url(),
    pool_size=10,
    max_overflow=20,
    pool_recycle=3600,
    echo=False  # Set to True for SQL debugging
)

# Create session factory
SessionLocal = sessionmaker(bind=engine)

def create_tables():
    """Create all database tables with error handling"""
    try:
        Base.metadata.create_all(engine)
        logger.info("Database tables created/verified successfully")
    except Exception as e:
        logger.error("Database table creation failed", error=str(e))
        raise

@contextmanager
def get_db_session():
    """Get database session with automatic cleanup and error handling"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        logger.error("Database session error", error=str(e))
        raise
    except Exception as e:
        session.rollback()
        logger.error("Unexpected database error", error=str(e))
        raise
    finally:
        session.close()

def test_connection() -> bool:
    """Test database connection"""
    try:
        with get_db_session() as session:
            session.execute("SELECT 1")
        logger.info("Database connection test successful")
        return True
    except Exception as e:
        logger.error("Database connection test failed", error=str(e))
        return False

def close_engine():
    """Close database engine and all connections"""
    if engine:
        engine.dispose()
        logger.info("Database engine closed")