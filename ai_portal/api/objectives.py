"""
Orchestration and objective execution API routes
"""

import time
import uuid
import json
import structlog
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from sqlalchemy.orm import Session

from ..schemas.objective import ObjectiveRequest, OrchestrationResponse
from ..models.chat_history import ChatHistory

logger = structlog.get_logger()
router = APIRouter(prefix="/objectives", tags=["objectives"])

def get_db_session():
    """Dependency to get database session"""
    # This will be injected by the main application
    pass

def get_orchestration_engine():
    """Dependency to get orchestration engine"""
    # This will be injected by the main application
    pass

def get_memory_service():
    """Dependency to get memory service"""
    # This will be injected by the main application
    pass

def save_orchestration_history(
    request: ObjectiveRequest, 
    results: dict, 
    project_id: uuid.UUID, 
    execution_time: float,
    db: Session
):
    """Save orchestration execution history for analysis and improvement"""
    logger.debug("Saving orchestration history", 
                project_id=str(project_id),
                execution_time=execution_time)
    
    try:
        # Save as a special chat history entry
        history = ChatHistory(
            project_id=project_id,
            persona_id=None,
            user_id=request.user_id,
            message=f"ORCHESTRATION: {request.objective}",
            response=json.dumps(results, indent=2, default=str),
            response_type='orchestration',
            model_used='memory_enhanced_reflexive_swarm_v26.2.0',
            response_time=execution_time
        )
        db.add(history)
        db.commit()
        logger.info("Orchestration history saved", 
                   project_id=str(project_id),
                   execution_time=execution_time)
    except Exception as e:
        logger.error("Failed to save orchestration history", 
                   project_id=str(project_id),
                   error=str(e))
        db.rollback()

def get_or_create_default_project(db: Session, user_id: str):
    """Get or create default project for user"""
    from ..models.project import Project
    
    logger.debug("Getting or creating default project", user_id=user_id)
    
    try:
        project = db.query(Project).filter(
            Project.user_id == user_id,
            Project.name == "Default Project"
        ).first()
        
        if not project:
            logger.info("Creating default project for user", user_id=user_id)
            project = Project(
                name="Default Project",
                description="Automatically created default project",
                user_id=user_id
            )
            db.add(project)
            db.commit()
            db.refresh(project)
            logger.info("Default project created", user_id=user_id, project_id=str(project.id))
        
        return project.id
        
    except Exception as e:
        logger.error("Failed to get or create default project", 
                    user_id=user_id,
                    error=str(e))
        db.rollback()
        raise

@router.post("/execute")
async def execute_objective(
    request: ObjectiveRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db_session),
    orchestration_engine = Depends(get_orchestration_engine),
    memory_service = Depends(get_memory_service)
):
    """
    Full Memory-Enhanced Agentic Orchestration Engine - The Learning Reflexive Swarm
    Master Planner -> Persona Dispatcher -> Critic Agent -> Adaptive Execution Loop
    """
    start_time = time.time()
    logger.info("Memory-enhanced objective orchestration started", 
               objective=request.objective,
               user_id=request.user_id)
    
    try:
        project_id = (
            uuid.UUID(request.project_id) if request.project_id 
            else get_or_create_default_project(db, request.user_id)
        )
        
        # === MEMORY-ENHANCED MASTER PLANNER AGENT ===
        logger.info("Activating Memory-Enhanced Master Planner Agent")
        
        # STEP 1: Query memory for similar past plans
        similar_plans = []
        if memory_service:
            try:
                similar_plans = await memory_service.query_similar_plans(request.objective, limit=3)
                if similar_plans:
                    logger.info("Found similar plans in memory", count=len(similar_plans))
            except Exception as e:
                logger.error("Failed to query memory for similar plans", error=str(e))
        
        # STEP 2: Execute orchestration with memory context
        try:
            orchestration_results = await orchestration_engine.run_orchestration_plan(
                request.objective, request.user_id, project_id, similar_plans
            )
        except Exception as e:
            logger.error("Orchestration execution failed", 
                       objective=request.objective,
                       error=str(e))
            raise HTTPException(
                status_code=500,
                detail=f"Orchestration failed: {str(e)}"
            )
        
        execution_time = time.time() - start_time
        
        # === STORE SUCCESSFUL PLAN IN MEMORY ===
        if (memory_service and 
            orchestration_results.get("successful_steps", 0) > 0):
            try:
                await memory_service.store_successful_plan(
                    objective=request.objective,
                    plan=orchestration_results.get("execution_plan", []),
                    execution_results=orchestration_results,
                    user_id=request.user_id
                )
                logger.info("Plan stored in agent memory for future learning")
            except Exception as e:
                logger.error("Failed to store plan in memory", error=str(e))
        
        # Save orchestration history
        background_tasks.add_task(
            save_orchestration_history,
            request, 
            orchestration_results, 
            project_id, 
            execution_time,
            db
        )
        
        logger.info("Memory-enhanced objective orchestration completed", 
                   objective=request.objective,
                   execution_time=execution_time,
                   success_rate=orchestration_results.get("success_rate", 0))
        
        return {
            "status": "orchestration_complete",
            "objective": request.objective,
            "project_id": str(project_id),
            "execution_plan": orchestration_results.get("execution_plan", []),
            "orchestration_results": orchestration_results,
            "final_synthesis": orchestration_results.get("final_synthesis", ""),
            "performance_metrics": {
                "total_execution_time": execution_time,
                "total_steps": orchestration_results.get("total_steps", 0),
                "successful_steps": orchestration_results.get("successful_steps", 0),
                "failed_steps": orchestration_results.get("failed_steps", 0),
                "adaptive_corrections": orchestration_results.get("adaptive_corrections_used", 0),
                "success_rate": orchestration_results.get("success_rate", 0),
                "memory_informed": len(similar_plans) > 0
            },
            "system_info": {
                "orchestration_engine": "Memory-Enhanced Reflexive Swarm v26.2.0",
                "agents_used": [
                    "Memory-Enhanced Master Planner", 
                    "Persona Dispatcher", 
                    "Critic Agent", 
                    "Learning Memory System"
                ],
                "learning_active": memory_service is not None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error("Orchestration failed", 
                   objective=request.objective,
                   error=str(e),
                   execution_time=execution_time)
        return {
            "status": "orchestration_failed",
            "objective": request.objective,
            "error": str(e),
            "project_id": str(project_id) if 'project_id' in locals() else None,
            "execution_time": execution_time,
            "fallback_message": "The orchestration system encountered an error. Please try with a simpler objective or check system configuration."
        }