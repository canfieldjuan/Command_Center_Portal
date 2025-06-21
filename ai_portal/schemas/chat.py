"""
Chat-related Pydantic schemas for conversation API
"""

from typing import Optional
from pydantic import BaseModel, field_validator

class ChatRequest(BaseModel):
    """Schema for chat requests"""
    message: str
    user_id: str = "anonymous"
    project_id: Optional[str] = None
    persona_id: Optional[str] = None
    task_type: str = "auto"
    user_tier: str = "free"
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
        if len(v.strip()) > 50000:
            raise ValueError('Message too long (max 50000 characters)')
        return v.strip()
    
    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v):
        if not v or not v.strip():
            raise ValueError('User ID cannot be empty')
        if len(v.strip()) > 100:
            raise ValueError('User ID too long (max 100 characters)')
        return v.strip()
    
    @field_validator('task_type')
    @classmethod
    def validate_task_type(cls, v):
        allowed_types = ["auto", "simple_qa", "code_generation", "image_generation", "function_routing", "complex_reasoning"]
        if v not in allowed_types:
            raise ValueError(f'Task type must be one of: {", ".join(allowed_types)}')
        return v
    
    @field_validator('user_tier')
    @classmethod
    def validate_user_tier(cls, v):
        allowed_tiers = ["free", "pro", "enterprise"]
        if v not in allowed_tiers:
            raise ValueError(f'User tier must be one of: {", ".join(allowed_tiers)}')
        return v

class ChatCompletionResponse(BaseModel):
    """Schema for AI completion responses"""
    type: str
    response: str
    
    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        allowed_types = ["text", "image", "tool_response", "error"]
        if v not in allowed_types:
            raise ValueError(f'Response type must be one of: {", ".join(allowed_types)}')
        return v

class ChatResponse(BaseModel):
    """Schema for complete chat responses"""
    success: bool
    response: str
    response_type: str
    model: str
    reasoning: str
    project_id: str
    
    class Config:
        from_attributes = True

class ChatHistoryResponse(BaseModel):
    """Schema for chat history entries"""
    id: str
    message: str
    response: str
    response_type: str
    model_used: str
    response_time: float
    created_at: str
    persona_id: Optional[str]
    
    class Config:
        from_attributes = True