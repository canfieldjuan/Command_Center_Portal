"""
Project-related Pydantic schemas for API requests and responses
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, field_validator

class ProjectRequest(BaseModel):
    """Schema for creating/updating projects"""
    name: str
    description: Optional[str] = None
    user_id: str = "anonymous"
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError('Project name cannot be empty')
        if len(v.strip()) > 255:
            raise ValueError('Project name too long (max 255 characters)')
        return v.strip()
    
    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v):
        if not v or not v.strip():
            raise ValueError('User ID cannot be empty')
        if len(v.strip()) > 100:
            raise ValueError('User ID too long (max 100 characters)')
        return v.strip()

class ProjectResponse(BaseModel):
    """Schema for project API responses"""
    id: str
    name: str
    description: Optional[str]
    user_id: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class ProjectUpdateRequest(BaseModel):
    """Schema for updating existing projects"""
    name: Optional[str] = None
    description: Optional[str] = None
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if v is not None:
            if not v.strip():
                raise ValueError('Project name cannot be empty')
            if len(v.strip()) > 255:
                raise ValueError('Project name too long (max 255 characters)')
            return v.strip()
        return v