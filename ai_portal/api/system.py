"""
System status and health check API routes
"""

import structlog
from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

logger = structlog.get_logger()
router = APIRouter(prefix="/system", tags=["system"])

def get_db_session():
    """Dependency to get database session"""
    # This will be injected by the main application
    pass

def get_memory_service():
    """Dependency to get memory service"""
    # This will be injected by the main application
    pass

def get_config():
    """Dependency to get configuration"""
    # This will be injected by the main application
    pass

@router.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "version": "26.2.0",
        "timestamp": datetime.utcnow().isoformat(),
        "system": "AI Portal Learning Machine"
    }

@router.get("/status")
async def system_status(
    db: Session = Depends(get_db_session),
    memory_service = Depends(get_memory_service),
    config: dict = Depends(get_config)
):
    """Get comprehensive system status and configuration"""
    logger.debug("System status check requested")
    
    try:
        from ..models.project import Project
        from ..models.persona import Persona
        from ..models.chat_history import ChatHistory
        
        # Get database statistics
        project_count = db.query(Project).count()
        persona_count = db.query(Persona).count()
        chat_count = db.query(ChatHistory).count()
        
        # Get orchestration statistics
        orchestration_count = db.query(ChatHistory).filter(
            ChatHistory.response_type == 'orchestration'
        ).count()
        
        # Check service availability
        services_status = {
            "openrouter": bool(config.get("openrouter_api_key")),
            "google": bool(config.get("google_application_credentials")),
            "serper": bool(config.get("serper_api_key")),
            "copyshark": bool(config.get("copyshark_api_token"))
        }
        
        # Memory system status
        memory_status = {
            "initialized": memory_service is not None,
            "learning_active": memory_service is not None
        }
        
        if memory_service:
            try:
                memory_stats = await memory_service.get_memory_stats()
                memory_status.update(memory_stats)
            except Exception as e:
                logger.error("Failed to get memory stats for status", error=str(e))
                memory_status["error"] = str(e)
        
        status_response = {
            "status": "operational",
            "version": "26.2.0",
            "system": "AI Portal Learning Machine",
            "timestamp": datetime.utcnow().isoformat(),
            "database": {
                "connected": True,
                "projects": project_count,
                "personas": persona_count,
                "chat_history_entries": chat_count,
                "orchestration_executions": orchestration_count
            },
            "services": services_status,
            "memory_system": memory_status,
            "orchestration_agents": [
                "Memory-Enhanced Master Planner",
                "Persona Dispatcher", 
                "Critic Agent",
                "Adaptive Execution Loop",
                "Learning Memory System"
            ],
            "available_tools": [tool["name"] for tool in config.get("available_tools", [])],
            "model_tiers": list(config.get("model_tiers", {}).keys()),
            "features": [
                "Multi-agent orchestration",
                "Persona-based AI routing",
                "Tool integration",
                "Persistent learning memory",
                "Adaptive error correction",
                "Real-time web search",
                "Website content extraction",
                "File workspace management",
                "Ad copy generation"
            ],
            "performance": {
                "total_projects": project_count,
                "total_personas": persona_count,
                "total_interactions": chat_count,
                "learning_enabled": memory_service is not None
            }
        }
        
        logger.debug("System status check completed successfully")
        return status_response
        
    except Exception as e:
        logger.error("System status check failed", error=str(e))
        return {
            "status": "degraded",
            "error": str(e),
            "version": "26.2.0",
            "timestamp": datetime.utcnow().isoformat(),
            "system": "AI Portal Learning Machine"
        }