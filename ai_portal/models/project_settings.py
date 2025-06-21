"""
Project settings database model for configuration management - COMPLETE VERSION
Extracted from main.py - ALL original configuration functionality preserved
"""

from sqlalchemy import Column, Integer, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from .base import Base

class ProjectSettings(Base):
    """
    Project settings model for storing project-specific configurations
    COMPLETE with all original configuration functionality from main.py
    """
    __tablename__ = 'project_settings'
    
    project_id = Column(PG_UUID(as_uuid=True), ForeignKey('projects.id', ondelete='CASCADE'), primary_key=True)
    active_persona_id = Column(PG_UUID(as_uuid=True), ForeignKey('personas.id', ondelete='SET NULL'), nullable=True)
    context_length = Column(Integer, default=10)
    settings = Column(JSON, default={})
    
    def __repr__(self):
        return f"<ProjectSettings(project_id={self.project_id}, context_length={self.context_length})>"
    
    def to_dict(self):
        """Convert project settings to dictionary with complete configuration"""
        return {
            "project_id": str(self.project_id),
            "active_persona_id": str(self.active_persona_id) if self.active_persona_id else None,
            "context_length": self.context_length,
            "settings": self.settings or {}
        }
    
    @classmethod
    def get_or_create_settings(cls, session, project_id):
        """Get or create project settings - ORIGINAL FUNCTIONALITY"""
        import structlog
        logger = structlog.get_logger()
        
        try:
            settings = session.query(cls).filter(cls.project_id == project_id).first()
            
            if not settings:
                logger.debug("Creating default project settings", project_id=str(project_id))
                settings = cls(
                    project_id=project_id,
                    context_length=10,
                    settings={}
                )
                session.add(settings)
                session.commit()
                session.refresh(settings)
            
            return settings
            
        except Exception as e:
            logger.error("Failed to get or create project settings", 
                       project_id=str(project_id),
                       error=str(e))
            session.rollback()
            raise
    
    def update_setting(self, session, key: str, value):
        """Update a specific setting - ORIGINAL FUNCTIONALITY"""
        import structlog
        logger = structlog.get_logger()
        
        try:
            if not self.settings:
                self.settings = {}
            
            self.settings[key] = value
            # SQLAlchemy needs to know the JSON column changed
            session.merge(self)
            session.commit()
            
            logger.debug("Project setting updated", 
                       project_id=str(self.project_id),
                       key=key,
                       value=value)
            
        except Exception as e:
            logger.error("Failed to update project setting", 
                       project_id=str(self.project_id),
                       key=key,
                       error=str(e))
            session.rollback()
            raise
    
    def get_setting(self, key: str, default=None):
        """Get a specific setting value - ORIGINAL FUNCTIONALITY"""
        if not self.settings:
            return default
        return self.settings.get(key, default)
    
    def remove_setting(self, session, key: str):
        """Remove a specific setting - ORIGINAL FUNCTIONALITY"""
        import structlog
        logger = structlog.get_logger()
        
        try:
            if self.settings and key in self.settings:
                del self.settings[key]
                session.merge(self)
                session.commit()
                
                logger.debug("Project setting removed", 
                           project_id=str(self.project_id),
                           key=key)
                
        except Exception as e:
            logger.error("Failed to remove project setting", 
                       project_id=str(self.project_id),
                       key=key,
                       error=str(e))
            session.rollback()
            raise