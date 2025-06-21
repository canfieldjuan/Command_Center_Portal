"""
Chat history database model for conversation tracking - COMPLETE VERSION
"""

import uuid
from sqlalchemy import Column, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .base import Base

class ChatHistory(Base):
    """
    Chat history model for storing conversation data and analytics
    COMPLETE with all original tracking and metrics functionality
    """
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
    
    # Relationships - COMPLETE RELATIONSHIP HANDLING
    project = relationship("Project", back_populates="chat_history")
    persona = relationship("Persona", back_populates="chat_history")
    
    def __repr__(self):
        return f"<ChatHistory(id={self.id}, project_id={self.project_id}, model='{self.model_used}')>"
    
    def to_dict(self):
        """Convert chat history to dictionary with complete metrics"""
        return {
            "id": str(self.id),
            "project_id": str(self.project_id),
            "persona_id": str(self.persona_id) if self.persona_id else None,
            "user_id": self.user_id,
            "message": self.message,
            "response": self.response,
            "response_type": self.response_type,
            "model_used": self.model_used,
            "cost": self.cost,
            "response_time": self.response_time,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def create_entry(cls, session, project_id: uuid.UUID, user_id: str, 
                    message: str, response: str, response_type: str,
                    model_used: str, response_time: float, persona_id: uuid.UUID = None,
                    cost: float = 0.0):
        """Create chat history entry with full validation - ORIGINAL FUNCTIONALITY"""
        import structlog
        logger = structlog.get_logger()
        
        logger.debug("Creating chat history entry", 
                    project_id=str(project_id),
                    model=model_used,
                    response_time=response_time)
        
        try:
            # Validate persona_id format if provided
            validated_persona_id = None
            if persona_id:
                if isinstance(persona_id, str):
                    try:
                        validated_persona_id = uuid.UUID(persona_id)
                    except ValueError:
                        logger.warning("Invalid persona_id format", persona_id=persona_id)
                        validated_persona_id = None
                else:
                    validated_persona_id = persona_id
            
            # Create chat history entry
            history = cls(
                project_id=project_id,
                persona_id=validated_persona_id,
                user_id=user_id,
                message=message,
                response=response,
                response_type=response_type,
                model_used=model_used,
                cost=cost,
                response_time=response_time
            )
            
            session.add(history)
            session.commit()
            session.refresh(history)
            
            logger.debug("Chat history saved successfully", 
                       history_id=str(history.id))
            
            return history
            
        except Exception as e:
            logger.error("Failed to save chat history", 
                       project_id=str(project_id),
                       error=str(e))
            session.rollback()
            raise
    
    @classmethod
    def get_project_history(cls, session, project_id: uuid.UUID, user_id: str, limit: int = 50):
        """Get chat history for project with validation - ORIGINAL FUNCTIONALITY"""
        import structlog
        logger = structlog.get_logger()
        
        logger.debug("Getting project history", 
                    project_id=str(project_id), 
                    user_id=user_id,
                    limit=limit)
        
        try:
            history = session.query(cls).filter(
                cls.project_id == project_id,
                cls.user_id == user_id
            ).order_by(cls.created_at.desc()).limit(limit).all()
            
            logger.debug("Project history retrieved", 
                       project_id=str(project_id),
                       count=len(history))
            
            return [h.to_dict() for h in history]
            
        except Exception as e:
            logger.error("Failed to get project history", 
                       project_id=str(project_id),
                       error=str(e))
            raise
    
    @classmethod
    def get_orchestration_count(cls, session) -> int:
        """Get count of orchestration executions - ORIGINAL FUNCTIONALITY"""
        try:
            return session.query(cls).filter(
                cls.response_type == 'orchestration'
            ).count()
        except Exception as e:
            import structlog
            logger = structlog.get_logger()
            logger.error("Failed to get orchestration count", error=str(e))
            return 0