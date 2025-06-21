"""
Persona-related Pydantic schemas for AI personality management
"""

from typing import Optional
from pydantic import BaseModel, field_validator

class PersonaRequest(BaseModel):
    """Schema for creating/updating personas"""
    name: str
    system_prompt: str
    model_preference: Optional[str] = None
    user_id: str = "anonymous"
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError('Persona name cannot be empty')
        if len(v.strip()) > 255:
            raise ValueError('Persona name too long (max 255 characters)')
        return v.strip()
    
    @field_validator('system_prompt')
    @classmethod
    def validate_system_prompt(cls, v):
        if not v or not v.strip():
            raise ValueError('System prompt cannot be empty')
        if len(v.strip()) > 10000:
            raise ValueError('System prompt too long (max 10000 characters)')
        return v.strip()
    
    @field_validator('model_preference')
    @classmethod
    def validate_model_preference(cls, v):
        if v is not None:
            if len(v.strip()) > 100:
                raise ValueError('Model preference too long (max 100 characters)')
            return v.strip() if v.strip() else None
        return v
    
    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v):
        if not v or not v.strip():
            raise ValueError('User ID cannot be empty')
        if len(v.strip()) > 100:
            raise ValueError('User ID too long (max 100 characters)')
        return v.strip()

class PersonaResponse(BaseModel):
    """Schema for persona API responses"""
    id: str
    name: str
    system_prompt: str
    model_preference: Optional[str]
    user_id: str
    
    class Config:
        from_attributes = True

class PersonaUpdateRequest(BaseModel):
    """Schema for updating existing personas"""
    name: Optional[str] = None
    system_prompt: Optional[str] = None
    model_preference: Optional[str] = None
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if v is not None:
            if not v.strip():
                raise ValueError('Persona name cannot be empty')
            if len(v.strip()) > 255:
                raise ValueError('Persona name too long (max 255 characters)')
            return v.strip()
        return v
    
    @field_validator('system_prompt')
    @classmethod
    def validate_system_prompt(cls, v):
        if v is not None:
            if not v.strip():
                raise ValueError('System prompt cannot be empty')
            if len(v.strip()) > 10000:
                raise ValueError('System prompt too long (max 10000 characters)')
            return v.strip()
        return v