"""
Database connection and session management
"""

import structlog
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from ..models.base import Base

logger = structlog.get_logger()

class DatabaseManager:
    """Database connection and session management"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.SessionLocal = None
        self._initialize_engine()

    def _initialize_engine(self):
        """Initialize SQLAlchemy engine with connection pooling"""
        try:
            self.engine = create_engine(
                self.database_url,
                pool_size=10,
                max_overflow=20,
                pool_recycle=3600,
                echo=False  # Set to True for SQL debugging
            )
            
            self.SessionLocal = sessionmaker(bind=self.engine)
            logger.info("Database engine initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize database engine", error=str(e))
            raise

    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created/verified successfully")
        except Exception as e:
            logger.error("Database table creation failed", error=str(e))
            raise

    @contextmanager
    def get_session(self):
        """Get database session with automatic cleanup"""
        session = self.SessionLocal()
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

    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error("Database connection test failed", error=str(e))
            return False

    def get_session_factory(self):
        """Get the session factory for dependency injection"""
        return self.SessionLocal

    def close_engine(self):
        """Close database engine and all connections"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database engine closed")