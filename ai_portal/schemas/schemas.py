# AI Portal Pydantic Schemas
# Extracted from main.py v26.2.0 - COMPLETE ORIGINAL FUNCTIONALITY

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, field_validator

class ProjectRequest(BaseModel):
    name: str
    description: Optional[str] = None
    user_id: str = "anonymous"

class ProjectResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    user_id: str
    created_at: datetime
    updated_at: datetime

class PersonaRequest(BaseModel):
    name: str
    system_prompt: str
    model_preference: Optional[str] = None
    user_id: str = "anonymous"

class PersonaResponse(BaseModel):
    id: str
    name: str
    system_prompt: str
    model_preference: Optional[str]
    user_id: str

class ChatRequest(BaseModel):
    message: str
    user_id: str = "anonymous"
    project_id: Optional[str] = None
    persona_id: Optional[str] = None
    task_type: str = "auto"
    user_tier: str = "free"

class ChatCompletionResponse(BaseModel):
    type: str
    response: str

class ChatResponse(BaseModel):
    success: bool
    response: str
    response_type: str
    model: str
    reasoning: str
    project_id: str

class ObjectiveRequest(BaseModel):
    objective: str
    user_id: str = "anonymous"
    project_id: Optional[str] = None
