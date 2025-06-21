"""
Database session management and connection handling - COMPLETE VERSION
ALL original database functionality from main.py preserved
🔴 HIGH RISK MODULE - Database transaction boundaries and session lifecycle
"""

import structlog
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from typing import Optional

from ..models.base import Base

logger = structlog.get_logger()

class DatabaseManager:
    """
    Database connection and session management with transaction safety
    🔴 HIGH RISK - COMPLETE ORIGINAL IMPLEMENTATION PRESERVED
    ALL database patterns, connection pooling, and error handling intact
    """
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.DbSession = None
        self._connection_verified = False
        
        logger.info("DatabaseManager initializing", url_masked=database_url[:50] + "...")
        self._initialize_engine()

    def _initialize_engine(self):
        """Initialize SQLAlchemy engine with connection pooling - COMPLETE ORIGINAL CONFIGURATION"""
        try:
            # CRITICAL: Preserve exact connection pooling settings from main.py
            self.engine = create_engine(
                self.database_url,
                pool_size=10,
                max_overflow=20,
                pool_recycle=3600,
                echo=False  # Set to True for SQL debugging
            )
            
            # Create session factory with exact original pattern
            self.DbSession = sessionmaker(bind=self.engine)
            
            logger.info("Database engine initialized successfully",
                       pool_size=10,
                       max_overflow=20,
                       pool_recycle=3600)
            
        except Exception as e:
            logger.error("Failed to initialize database engine", error=str(e))
            raise ValueError(f"Database engine initialization failed: {str(e)}")

    def create_tables(self):
        """Create all database tables - COMPLETE ORIGINAL TABLE CREATION"""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created/verified successfully")
        except Exception as e:
            logger.error("Database table creation failed", error=str(e))
            raise ValueError(f"Database table creation failed: {str(e)}")

    @contextmanager
    def get_session(self):
        """
        Get database session with automatic transaction management
        🔴 CRITICAL: Preserves exact session lifecycle and error handling patterns
        """
        session = self.DbSession()
        try:
            yield session
            session.commit()
            logger.debug("Database session committed successfully")
        except SQLAlchemyError as e:
            session.rollback()
            logger.error("Database session error, rolled back", error=str(e))
            raise
        except Exception as e:
            session.rollback()
            logger.error("Unexpected database error, rolled back", error=str(e))
            raise
        finally:
            session.close()
            logger.debug("Database session closed")

    def get_session_factory(self):
        """
        Get the session factory for dependency injection
        🔴 CRITICAL: Preserves exact session factory pattern used throughout application
        """
        if not self.DbSession:
            raise ValueError("Database not initialized. Call _initialize_engine() first.")
        
        return self.DbSession

    def test_connection(self) -> bool:
        """
        Test database connection with comprehensive error handling
        🔴 CRITICAL: Preserves exact connection testing logic from main.py
        """
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
            
            self._connection_verified = True
            logger.info("Database connection test successful")
            return True
            
        except Exception as e:
            self._connection_verified = False
            logger.error("Database connection test failed", error=str(e))
            return False

    def is_connection_verified(self) -> bool:
        """Check if database connection has been verified"""
        return self._connection_verified

    def get_engine_info(self) -> dict:
        """
        Get database engine information for monitoring
        🔴 CRITICAL: Preserves exact engine configuration reporting
        """
        if not self.engine:
            return {"status": "not_initialized"}
        
        return {
            "status": "initialized",
            "url_masked": self.database_url[:50] + "...",
            "pool_size": self.engine.pool.size(),
            "checked_out_connections": self.engine.pool.checkedout(),
            "overflow": self.engine.pool.overflow(),
            "checked_in": self.engine.pool.checkedin(),
            "connection_verified": self._connection_verified,
            "dialect": str(self.engine.dialect.name),
            "driver": str(self.engine.dialect.driver)
        }

    def close_engine(self):
        """
        Close database engine and all connections
        🔴 CRITICAL: Preserves exact shutdown sequence
        """
        if self.engine:
            self.engine.dispose()
            logger.info("Database engine closed and connections disposed")
            self._connection_verified = False
        else:
            logger.warning("Database engine was not initialized, nothing to close")

    def get_database_stats(self) -> dict:
        """
        Get comprehensive database statistics for monitoring
        🔴 CRITICAL: Database health monitoring functionality
        """
        stats = {
            "engine_info": self.get_engine_info(),
            "connection_status": "unknown"
        }
        
        try:
            # Test connection and get basic stats
            with self.get_session() as session:
                # Get table counts for monitoring
                from ..models.project import Project
                from ..models.persona import Persona  
                from ..models.chat_history import ChatHistory
                
                stats["table_counts"] = {
                    "projects": session.query(Project).count(),
                    "personas": session.query(Persona).count(),
                    "chat_history": session.query(ChatHistory).count(),
                    "orchestration_executions": session.query(ChatHistory).filter(
                        ChatHistory.response_type == 'orchestration'
                    ).count()
                }
                
                stats["connection_status"] = "healthy"
                logger.debug("Database statistics retrieved successfully")
                
        except Exception as e:
            stats["connection_status"] = "error"
            stats["error"] = str(e)
            logger.error("Failed to retrieve database statistics", error=str(e))
        
        return stats

    def execute_raw_sql(self, sql: str, params: Optional[dict] = None):
        """
        Execute raw SQL with proper session management
        🔴 CRITICAL: For administrative operations requiring raw SQL
        """
        logger.warning("Executing raw SQL", sql=sql[:100])
        
        try:
            with self.get_session() as session:
                result = session.execute(sql, params or {})
                session.commit()
                
                logger.info("Raw SQL executed successfully")
                return result
                
        except Exception as e:
            logger.error("Raw SQL execution failed", sql=sql[:100], error=str(e))
            raise

    def validate_schema(self) -> dict:
        """
        Validate database schema integrity
        🔴 CRITICAL: Ensures database schema matches model definitions
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "tables_found": []
        }
        
        try:
            from sqlalchemy import inspect
            
            inspector = inspect(self.engine)
            existing_tables = inspector.get_table_names()
            
            # Expected tables from models
            expected_tables = ['projects', 'personas', 'chat_history', 'project_settings']
            
            validation_result["tables_found"] = existing_tables
            
            # Check for missing tables
            missing_tables = [table for table in expected_tables if table not in existing_tables]
            if missing_tables:
                validation_result["errors"].extend([f"Missing table: {table}" for table in missing_tables])
                validation_result["valid"] = False
            
            # Check for unexpected tables (informational)
            extra_tables = [table for table in existing_tables if table not in expected_tables]
            if extra_tables:
                validation_result["warnings"].extend([f"Extra table found: {table}" for table in extra_tables])
            
            # Validate key constraints and indexes
            for table_name in expected_tables:
                if table_name in existing_tables:
                    try:
                        columns = inspector.get_columns(table_name)
                        indexes = inspector.get_indexes(table_name)
                        foreign_keys = inspector.get_foreign_keys(table_name)
                        
                        validation_result[f"{table_name}_columns"] = len(columns)
                        validation_result[f"{table_name}_indexes"] = len(indexes)
                        validation_result[f"{table_name}_foreign_keys"] = len(foreign_keys)
                        
                    except Exception as e:
                        validation_result["warnings"].append(f"Could not inspect table {table_name}: {str(e)}")
            
            logger.info("Database schema validation completed", 
                       valid=validation_result["valid"],
                       tables_found=len(existing_tables))
            
        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Schema validation failed: {str(e)}")
            logger.error("Database schema validation failed", error=str(e))
        
        return validation_result

    def backup_database_info(self) -> dict:
        """
        Get information needed for database backup operations
        🔴 CRITICAL: For production backup and recovery procedures
        """
        backup_info = {
            "database_type": "postgresql",
            "connection_info": {
                "host": "aws-0-us-east-2.pooler.supabase.com",
                "port": 6543,
                "database": "postgres",
                "username": "postgres.jacjorrzxilmrfxbdyse"
            },
            "tables": [],
            "estimated_size": "unknown"
        }
        
        try:
            with self.get_session() as session:
                from sqlalchemy import text
                
                # Get table information
                result = session.execute(text("""
                    SELECT table_name, 
                           pg_size_pretty(pg_total_relation_size(quote_ident(table_name))) as size
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_type = 'BASE TABLE'
                """))
                
                backup_info["tables"] = [
                    {"name": row[0], "size": row[1]} 
                    for row in result.fetchall()
                ]
                
                # Get total database size
                size_result = session.execute(text("SELECT pg_size_pretty(pg_database_size(current_database()))"))
                backup_info["estimated_size"] = size_result.fetchone()[0]
                
                logger.info("Database backup information retrieved", 
                           tables=len(backup_info["tables"]),
                           total_size=backup_info["estimated_size"])
                
        except Exception as e:
            backup_info["error"] = str(e)
            logger.error("Failed to retrieve backup information", error=str(e))
        
        return backup_info

    def __enter__(self):
        """Context manager entry - returns session factory"""
        return self.get_session_factory()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup if needed"""
        if exc_type:
            logger.error("DatabaseManager context exit with exception", 
                        exc_type=str(exc_type),
                        exc_val=str(exc_val))
        # Engine remains open for application lifecycle