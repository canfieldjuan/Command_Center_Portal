"""
Master Planner Agent for objective breakdown and execution planning - COMPLETE VERSION
ALL original Master Planner functionality from main.py preserved
"""

import json
import structlog
from typing import Dict, List, Any, Optional

logger = structlog.get_logger()

class MasterPlannerAgent:
    """
    Master Planner Agent for breaking down complex objectives into executable steps
    COMPLETE ORIGINAL IMPLEMENTATION - THE ORCHESTRATION BRAIN
    """
    
    def __init__(self, services: Dict, router, config: Dict):
        self.services = services
        self.router = router
        self.config = config
        logger.info("Master Planner Agent initialized")

    async def create_execution_plan(self, objective: str, similar_plans: List[Dict] = None) -> List[Dict]:
        """
        Create detailed execution plan for objective using memory-informed planning
        COMPLETE ORIGINAL IMPLEMENTATION WITH MEMORY ENHANCEMENT
        """
        logger.info("Master Planner creating execution plan", objective=objective[:100])
        
        # Get planner service
        planner_route = self.router.route('code_generation', 'free')
        planner_service = self.services.get(planner_route['service'])
        
        if not planner_service:
            raise ValueError("Master Planner service not available")
        
        # Create memory context from similar plans
        memory_context = ""
        if similar_plans:
            memory_context = "\n\nRELEVANT PAST EXPERIENCE:\n"
            for i, plan in enumerate(similar_plans[:2]):  # Use top 2 similar plans
                memory_context += f"Past Objective: {plan.get('objective', '')}\n"
                memory_context += f"Success Rate: {plan.get('success_rate', 0):.2f}\n"
                memory_context += f"Key Steps: {json.dumps(plan.get('plan', [])[:3])}\n\n"  # First 3 steps
            memory_context += "Use this experience to create a better plan.\n"
        
        # Create comprehensive planning prompt
        planning_prompt = f"""
        You are the Memory-Enhanced Master Planner Agent with access to past successful operations.
        Your job is to break down complex objectives into executable steps using learned experience.
        
        OBJECTIVE: {objective}
        {memory_context}
        
        Create a detailed execution plan that improves on past experience. Each step should be:
        1. Specific and actionable
        2. Have clear success criteria
        3. Be executable by AI agents with available tools
        4. Learn from past successes and failures
        
        Available tools: web_search, browse_website, save_to_file, generateAdCopy
        
        Respond with EXACTLY this JSON format:
        {{
            "plan": [
                {{
                    "step": 1,
                    "description": "Detailed task description",
                    "objective": "Specific success criteria", 
                    "type": "simple_qa|code_generation|image_generation",
                    "estimated_difficulty": "low|medium|high",
                    "dependencies": [],
                    "learned_from_memory": true/false
                }}
            ],
            "total_steps": 5,
            "estimated_duration": "15 minutes",
            "success_criteria": "Overall objective completion criteria",
            "memory_informed": {len(similar_plans) > 0}
        }}
        
        Make the plan comprehensive, achievable, and informed by past experience.
        """
        
        try:
            planner_messages = [{"role": "user", "content": planning_prompt}]
            planner_completion = await planner_service.chat_completion(planner_messages, planner_route['model'])
            
            # Parse the plan
            try:
                plan_data = json.loads(planner_completion.response)
                execution_plan = plan_data.get("plan", [])
                
                logger.info("Master Planner created execution plan", 
                          total_steps=len(execution_plan),
                          memory_informed=plan_data.get("memory_informed", False))
                
                return execution_plan
                
            except json.JSONDecodeError:
                logger.error("Failed to parse Master Planner response", 
                           response=planner_completion.response[:500])
                
                # Fallback plan
                execution_plan = [{
                    "step": 1,
                    "description": objective,
                    "objective": "Complete the requested objective",
                    "type": "simple_qa",
                    "estimated_difficulty": "medium",
                    "dependencies": []
                }]
                
                logger.warning("Using fallback execution plan", steps=len(execution_plan))
                return execution_plan
                
        except Exception as e:
            logger.error("Master Planner execution failed", error=str(e))
            raise ValueError(f"Master Planner failed: {str(e)}")

    async def synthesize_results(self, objective: str, orchestration_results: Dict) -> str:
        """
        Synthesize final results from orchestration execution
        COMPLETE ORIGINAL FUNCTIONALITY
        """
        logger.info("Master Planner synthesizing results", objective=objective[:50])
        
        try:
            # Get synthesis service
            synthesis_route = self.router.route('simple_qa', 'free')
            synthesis_service = self.services.get(synthesis_route['service'])
            
            synthesis_prompt = f"""
            You are the Memory-Enhanced Master Planner Agent completing an orchestration cycle.
            
            ORIGINAL OBJECTIVE: {objective}
            
            EXECUTION RESULTS:
            {json.dumps(orchestration_results, indent=2, default=str)}
            
            Provide a comprehensive summary of:
            1. What was accomplished
            2. Key deliverables created
            3. Any remaining tasks or recommendations
            4. Overall success assessment
            5. Learning insights for future similar objectives
            
            Be detailed and actionable in your summary.
            """
            
            synthesis_messages = [{"role": "user", "content": synthesis_prompt}]
            synthesis_completion = await synthesis_service.chat_completion(
                synthesis_messages, synthesis_route['model']
            )
            
            logger.info("Master Planner synthesis completed", 
                       objective=objective[:50])
            
            return synthesis_completion.response
            
        except Exception as e:
            logger.error("Master Planner synthesis failed", error=str(e))
            return f"Synthesis failed: {str(e)}. Raw results: {orchestration_results}"

    async def validate_plan_feasibility(self, plan: List[Dict]) -> Dict[str, Any]:
        """
        Validate plan feasibility and provide improvement suggestions
        COMPLETE ORIGINAL FUNCTIONALITY
        """
        logger.debug("Master Planner validating plan feasibility", steps=len(plan))
        
        try:
            validation_route = self.router.route('simple_qa', 'free')
            validation_service = self.services.get(validation_route['service'])
            
            validation_prompt = f"""
            You are the Master Planner Agent performing plan validation.
            
            EXECUTION PLAN:
            {json.dumps(plan, indent=2)}
            
            Analyze this plan for:
            1. Logical step ordering and dependencies
            2. Resource requirements and tool availability
            3. Estimated time and complexity
            4. Potential failure points
            5. Improvement recommendations
            
            Respond with JSON:
            {{
                "feasible": true/false,
                "confidence": 0.0-1.0,
                "issues": ["list of potential issues"],
                "recommendations": ["list of improvements"],
                "estimated_success_rate": 0.0-1.0
            }}
            """
            
            validation_messages = [{"role": "user", "content": validation_prompt}]
            validation_completion = await validation_service.chat_completion(
                validation_messages, validation_route['model']
            )
            
            try:
                validation_result = json.loads(validation_completion.response)
                logger.debug("Plan validation completed", 
                           feasible=validation_result.get("feasible", False),
                           confidence=validation_result.get("confidence", 0))
                return validation_result
            except json.JSONDecodeError:
                logger.warning("Failed to parse validation response")
                return {
                    "feasible": True,
                    "confidence": 0.5,
                    "issues": ["Validation parsing failed"],
                    "recommendations": ["Manual review recommended"],
                    "estimated_success_rate": 0.5
                }
                
        except Exception as e:
            logger.error("Plan validation failed", error=str(e))
            return {
                "feasible": True,
                "confidence": 0.3,
                "issues": [f"Validation error: {str(e)}"],
                "recommendations": ["Proceed with caution"],
                "estimated_success_rate": 0.3
            }