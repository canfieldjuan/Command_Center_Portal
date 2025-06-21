"""
Objective-related Pydantic schemas for orchestration API
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, field_validator

class ObjectiveRequest(BaseModel):
    """Schema for orchestration objective requests"""
    objective: str
    user_id: str = "anonymous"
    project_id: Optional[str] = None
    
    @field_validator('objective')
    @classmethod
    def validate_objective(cls, v):
        if not v or not v.strip():
            raise ValueError('Objective cannot be empty')
        if len(v.strip()) > 5000:
            raise ValueError('Objective too long (max 5000 characters)')
        return v.strip()
    
    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v):
        if not v or not v.strip():
            raise ValueError('User ID cannot be empty')
        if len(v.strip()) > 100:
            raise ValueError('User ID too long (max 100 characters)')
        return v.strip()

class OrchestrationStep(BaseModel):
    """Schema for individual orchestration steps"""
    step: int
    description: str
    objective: str
    type: str
    estimated_difficulty: Optional[str] = "medium"
    dependencies: List[int] = []
    learned_from_memory: bool = False

class OrchestrationPlan(BaseModel):
    """Schema for complete orchestration plans"""
    plan: List[OrchestrationStep]
    total_steps: int
    estimated_duration: str
    success_criteria: str
    memory_informed: bool = False

class OrchestrationResult(BaseModel):
    """Schema for orchestration step results"""
    step: int
    task: Dict[str, Any]
    result: str
    persona: str
    status: str
    critic_verdict: Dict[str, Any]
    execution_time: float
    adaptive_correction: bool = False
    memory_informed: bool = False
    correction_time: Optional[float] = None

class OrchestrationResponse(BaseModel):
    """Schema for complete orchestration responses"""
    status: str
    objective: str
    project_id: str
    execution_plan: List[OrchestrationStep]
    orchestration_results: Dict[str, Any]
    final_synthesis: str
    performance_metrics: Dict[str, Any]
    system_info: Dict[str, Any]

class MemoryStatsResponse(BaseModel):
    """Schema for memory system statistics"""
    status: str
    agent_learning: bool
    memory_stats: Dict[str, Any]
    capabilities: List[str]

class InsightRequest(BaseModel):
    """Schema for storing insights in memory"""
    insight: str
    context: Dict[str, Any] = {}
    user_id: str = "anonymous"
    
    @field_validator('insight')
    @classmethod
    def validate_insight(cls, v):
        if not v or not v.strip():
            raise ValueError('Insight cannot be empty')
        if len(v.strip()) > 2000:
            raise ValueError('Insight too long (max 2000 characters)')
        return v.strip()

class MemoryClearRequest(BaseModel):
    """Schema for clearing memory"""
    memory_type: str = "all"
    confirm_phrase: str = ""
    
    @field_validator('memory_type')
    @classmethod
    def validate_memory_type(cls, v):
        allowed_types = ["all", "plans", "tasks", "failures", "insights"]
        if v not in allowed_types:
            raise ValueError(f'Memory type must be one of: {", ".join(allowed_types)}')
        return v