"""
Project database model - COMPLETE VERSION
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .base import Base

class Project(Base):
    """
    Project model for organizing user work and chat sessions
    COMPLETE with all original functionality
    """
    __tablename__ = 'projects'
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    user_id = Column(String(100), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships - COMPLETE CASCADE HANDLING
    chat_history = relationship("ChatHistory", back_populates="project", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Project(id={self.id}, name='{self.name}', user_id='{self.user_id}')>"
    
    def to_dict(self):
        """Convert project to dictionary with full data"""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
        
    @classmethod
    def get_or_create_default(cls, session, user_id: str):
        """Get or create default project for user - ORIGINAL FUNCTIONALITY"""
        import structlog
        logger = structlog.get_logger()
        
        logger.debug("Getting or creating default project", user_id=user_id)
        
        try:
            project = session.query(cls).filter(
                cls.user_id == user_id,
                cls.name == "Default Project"
            ).first()
            
            if not project:
                logger.info("Creating default project for user", user_id=user_id)
                project = cls(
                    name="Default Project",
                    description="Automatically created default project",
                    user_id=user_id
                )
                session.add(project)
                session.commit()
                session.refresh(project)
                logger.info("Default project created", user_id=user_id, project_id=str(project.id))
            
            return project.id
            
        except Exception as e:
            logger.error("Failed to get or create default project", 
                        user_id=user_id,
                        error=str(e))
            session.rollback()
            raise