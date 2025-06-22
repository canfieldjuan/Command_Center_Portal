"""
SAVE AS: ai_portal/orchestration/orchestration_engine.py
UPDATE THIS EXISTING FILE IN THE ORCHESTRATION FOLDER

Main Orchestration Engine coordinating all agents - UPDATED FOR PHASE 3B
Uses extracted TaskExecutor for core execution logic
PRESERVES ALL ORIGINAL ORCHESTRATION FUNCTIONALITY
"""

import time
import json
import asyncio
import structlog
from typing import Dict, List, Any, Optional
from datetime import datetime

from .master_planner import MasterPlannerAgent
from .persona_dispatcher import PersonaDispatcherAgent
from .critic_agent import CriticAgent
from .task_executor import TaskExecutor

logger = structlog.get_logger()

class OrchestrationEngine:
    """
    Main orchestration engine coordinating Master Planner, Persona Dispatcher, and Critic Agent
    UPDATED TO USE EXTRACTED TASKEXECUTOR - PRESERVES ALL ORIGINAL FUNCTIONALITY
    """
    
    def __init__(self, services: Dict, router, config: Dict, memory_service=None):
        self.services = services
        self.router = router
        self.config = config
        self.memory_service = memory_service
        
        # Initialize agents
        self.master_planner = MasterPlannerAgent(services, router, config)
        self.persona_dispatcher = PersonaDispatcherAgent(services, router, config)
        self.critic_agent = CriticAgent(services, router, config)
        
        # Initialize TaskExecutor - THE NEW EXTRACTED COMPONENT
        self.task_executor = TaskExecutor(services, router, config, memory_service)
        
        logger.info("Orchestration Engine initialized with all agents and TaskExecutor")

    async def execute_objective(self, objective: str, user_id: str, project_id, session) -> Dict[str, Any]:
        """
        Execute full orchestration for an objective - PRESERVED ORIGINAL FLOW
        """
        start_time = time.time()
        logger.info("Memory-enhanced objective orchestration started", 
                   objective=objective,
                   user_id=user_id)
        
        try:
            # === MEMORY-ENHANCED MASTER PLANNER AGENT ===
            logger.info("Activating Memory-Enhanced Master Planner Agent")
            
            # STEP 1: Query memory for similar past plans
            similar_plans = []
            if self.memory_service:
                try:
                    similar_plans = await self.memory_service.query_similar_plans(objective, limit=3)
                    if similar_plans:
                        logger.info("Found similar plans in memory", count=len(similar_plans))
                except Exception as e:
                    logger.error("Failed to query memory for similar plans", error=str(e))
            
            # STEP 2: Create execution plan
            execution_plan = await self.master_planner.create_execution_plan(objective, similar_plans)
            
            # === EXECUTE THE MEMORY-ENHANCED ORCHESTRATION PLAN ===
            logger.info("Starting Memory-Enhanced Reflexive Swarm execution")
            
            # USE THE EXTRACTED TASKEXECUTOR - PRESERVES ALL ORIGINAL LOGIC
            orchestration_results = await self.task_executor.run_orchestration_plan(
                execution_plan, user_id, project_id, session
            )
            
            execution_time = time.time() - start_time
            
            # === STORE SUCCESSFUL PLAN IN MEMORY ===
            if (self.memory_service and 
                orchestration_results.get("successful_steps", 0) > 0):
                try:
                    await self.memory_service.store_successful_plan(
                        objective=objective,
                        plan=execution_plan,
                        execution_results=orchestration_results,
                        user_id=user_id
                    )
                    logger.info("Plan stored in agent memory for future learning")
                except Exception as e:
                    logger.error("Failed to store plan in memory", error=str(e))
            
            # === FINAL SYNTHESIS ===
            final_synthesis = await self.master_planner.synthesize_results(objective, orchestration_results)
            
            logger.info("Memory-enhanced objective orchestration completed", 
                       objective=objective,
                       execution_time=execution_time,
                       success_rate=orchestration_results.get("success_rate", 0))
            
            return {
                "status": "orchestration_complete",
                "objective": objective,
                "project_id": str(project_id),
                "execution_plan": execution_plan,
                "orchestration_results": orchestration_results,
                "final_synthesis": final_synthesis,
                "performance_metrics": {
                    "total_execution_time": execution_time,
                    "total_steps": orchestration_results["total_steps"],
                    "successful_steps": orchestration_results["successful_steps"],
                    "failed_steps": orchestration_results["failed_steps"],
                    "adaptive_corrections": orchestration_results["adaptive_corrections_used"],
                    "success_rate": orchestration_results.get("success_rate", 0),
                    "memory_informed": len(similar_plans) > 0
                },
                "system_info": {
                    "orchestration_engine": "Memory-Enhanced Reflexive Swarm v26.2.0",
                    "agents_used": [
                        "Memory-Enhanced Master Planner", 
                        "Persona Dispatcher", 
                        "Critic Agent", 
                        "TaskExecutor",
                        "Learning Memory System"
                    ],
                    "learning_active": self.memory_service is not None
                }
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error("Orchestration failed", 
                       objective=objective,
                       error=str(e),
                       execution_time=execution_time)
            return {
                "status": "orchestration_failed",
                "objective": objective,
                "error": str(e),
                "project_id": str(project_id),
                "execution_time": execution_time,
                "fallback_message": "The orchestration system encountered an error. Please try with a simpler objective or check system configuration."
            }

    async def run_orchestration_plan(self, objective: str, user_id: str, project_id, similar_plans: List[Dict] = None) -> Dict:
        """
        Simplified interface for external callers - DELEGATES TO TASKEXECUTOR
        """
        logger.info("Running orchestration plan", 
                   objective=objective[:100],
                   user_id=user_id)
        
        try:
            # Create execution plan
            execution_plan = await self.master_planner.create_execution_plan(objective, similar_plans or [])
            
            # Get database session for execution
            from ..core.database_manager import DatabaseManager
            from ..core.config import ConfigManager
            
            config_manager = ConfigManager()
            db_manager = DatabaseManager(config_manager.get('database_url'))
            
            with db_manager.get_session() as session:
                # Execute plan using TaskExecutor
                return await self.task_executor.run_orchestration_plan(
                    execution_plan, user_id, project_id, session
                )
                
        except Exception as e:
            logger.error("Orchestration plan execution failed", 
                       objective=objective[:100],
                       error=str(e))
            raise

    async def shutdown(self):
        """Graceful shutdown of orchestration engine"""
        logger.info("Orchestration engine shutting down")
        
        # Clean up any resources if needed
        if self.memory_service:
            logger.info("Memory service cleanup")
        
        logger.info("Orchestration engine shutdown complete")