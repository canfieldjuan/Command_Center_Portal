"""
Persona database model for AI personality management - COMPLETE VERSION
"""

import uuid
from sqlalchemy import Column, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import relationship

from .base import Base

class Persona(Base):
    """
    Persona model for AI personality and behavior customization
    COMPLETE with all original validation and functionality
    """
    __tablename__ = 'personas'
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    system_prompt = Column(Text, nullable=False)
    model_preference = Column(String(100))
    user_id = Column(String(100), nullable=False, index=True)
    
    # Constraints - COMPLETE UNIQUE CONSTRAINT HANDLING
    __table_args__ = (UniqueConstraint('user_id', 'name', name='_user_id_name_uc'),)
    
    # Relationships
    chat_history = relationship("ChatHistory", back_populates="persona")
    
    def __repr__(self):
        return f"<Persona(id={self.id}, name='{self.name}', user_id='{self.user_id}')>"
    
    def to_dict(self):
        """Convert persona to dictionary with complete data"""
        return {
            "id": str(self.id),
            "name": self.name,
            "system_prompt": self.system_prompt,
            "model_preference": self.model_preference,
            "user_id": self.user_id
        }
        
    def validate_name(self, name: str) -> bool:
        """Validate persona name - ORIGINAL VALIDATION LOGIC"""
        if not name or not isinstance(name, str):
            return False
        if len(name.strip()) == 0:
            return False
        if len(name.strip()) > 255:
            return False
        return True
    
    def validate_system_prompt(self, prompt: str) -> bool:
        """Validate system prompt - ORIGINAL VALIDATION LOGIC"""
        if not prompt or not isinstance(prompt, str):
            return False
        if len(prompt.strip()) == 0:
            return False
        if len(prompt.strip()) > 10000:
            return False
        return True
    
    def validate_model_preference(self, model: str) -> bool:
        """Validate model preference - ORIGINAL VALIDATION LOGIC"""
        if model is None:
            return True
        if not isinstance(model, str):
            return False
        if len(model.strip()) > 100:
            return False
        return True
    
    @classmethod
    def create_with_validation(cls, session, name: str, system_prompt: str, 
                             model_preference: str, user_id: str):
        """Create persona with full validation - ORIGINAL FUNCTIONALITY"""
        import structlog
        logger = structlog.get_logger()
        
        # Create temporary instance for validation
        temp_persona = cls()
        
        if not temp_persona.validate_name(name):
            raise ValueError("Invalid persona name")
        if not temp_persona.validate_system_prompt(system_prompt):
            raise ValueError("Invalid system prompt")
        if not temp_persona.validate_model_preference(model_preference):
            raise ValueError("Invalid model preference")
            
        try:
            persona = cls(
                name=name.strip(),
                system_prompt=system_prompt.strip(),
                model_preference=model_preference.strip() if model_preference else None,
                user_id=user_id.strip()
            )
            session.add(persona)
            session.commit()
            session.refresh(persona)
            
            logger.info("Persona created with validation", 
                       persona_id=str(persona.id),
                       name=persona.name)
            
            return persona
            
        except Exception as e:
            session.rollback()
            logger.error("Failed to create persona", 
                       name=name,
                       error=str(e))
            raise