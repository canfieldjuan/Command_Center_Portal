"""
Memory system API routes for agent learning management
"""

import structlog
from fastapi import APIRouter, HTTPException, Depends

from ..schemas.objective import MemoryStatsResponse, InsightRequest, MemoryClearRequest

logger = structlog.get_logger()
router = APIRouter(prefix="/memory", tags=["memory"])

def get_memory_service():
    """Dependency to get memory service"""
    # This will be injected by the main application
    pass

@router.get("/stats", response_model=MemoryStatsResponse)
async def get_memory_stats(memory_service = Depends(get_memory_service)):
    """Get agent memory statistics"""
    if not memory_service:
        return {
            "status": "memory_not_initialized",
            "agent_learning": False,
            "memory_stats": {},
            "capabilities": [],
            "error": "Memory system not initialized",
            "suggestion": "Memory system requires proper configuration and dependencies"
        }
    
    try:
        stats = await memory_service.get_memory_stats()
        return {
            "status": "memory_active",
            "agent_learning": True,
            "memory_stats": stats,
            "capabilities": [
                "Plan Learning",
                "Task Success Memory", 
                "Failure Pattern Recognition",
                "Corrective Action Learning",
                "Insight Storage"
            ]
        }
    except Exception as e:
        logger.error("Failed to get memory stats", error=str(e))
        return {
            "status": "memory_error",
            "agent_learning": False,
            "memory_stats": {},
            "capabilities": [],
            "error": f"Memory stats unavailable: {str(e)}"
        }

@router.post("/insights")
async def store_insight(
    insight_data: InsightRequest,
    memory_service = Depends(get_memory_service)
):
    """Store a learned insight in agent memory"""
    if not memory_service:
        raise HTTPException(status_code=503, detail="Memory system not initialized")
    
    try:
        memory_id = await memory_service.store_insight(
            insight_data.insight, 
            insight_data.context, 
            insight_data.user_id
        )
        
        return {
            "status": "insight_stored",
            "memory_id": memory_id,
            "insight": insight_data.insight,
            "learning_enabled": True
        }
    except Exception as e:
        logger.error("Failed to store insight", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to store insight: {str(e)}")

@router.post("/clear")
async def clear_memory(
    clear_request: MemoryClearRequest,
    memory_service = Depends(get_memory_service)
):
    """Clear agent memory (requires confirmation)"""
    if not memory_service:
        raise HTTPException(status_code=503, detail="Memory system not initialized")
    
    try:
        result = await memory_service.clear_memory(
            clear_request.memory_type, 
            clear_request.confirm_phrase
        )
        return result
    except Exception as e:
        logger.error("Failed to clear memory", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to clear memory: {str(e)}")

@router.get("/plans/similar")
async def query_similar_plans(
    objective: str,
    limit: int = 5,
    memory_service = Depends(get_memory_service)
):
    """Query memory for similar successful plans"""
    if not memory_service:
        raise HTTPException(status_code=503, detail="Memory system not initialized")
    
    try:
        similar_plans = await memory_service.query_similar_plans(objective, limit)
        return {
            "status": "plans_found",
            "objective": objective,
            "similar_plans": similar_plans,
            "count": len(similar_plans)
        }
    except Exception as e:
        logger.error("Failed to query similar plans", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to query similar plans: {str(e)}")

@router.get("/failures/similar")
async def query_similar_failures(
    task_description: str,
    failure_reason: str,
    limit: int = 3,
    memory_service = Depends(get_memory_service)
):
    """Query memory for similar past failures and their corrections"""
    if not memory_service:
        raise HTTPException(status_code=503, detail="Memory system not initialized")
    
    try:
        task = {"description": task_description}
        similar_failures = await memory_service.query_similar_failures(
            task, failure_reason, limit
        )
        return {
            "status": "failures_found",
            "task_description": task_description,
            "failure_reason": failure_reason,
            "similar_failures": similar_failures,
            "count": len(similar_failures)
        }
    except Exception as e:
        logger.error("Failed to query similar failures", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to query similar failures: {str(e)}")