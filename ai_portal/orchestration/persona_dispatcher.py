"""
Persona Dispatcher Agent for selecting appropriate specialists - COMPLETE VERSION
ALL original Persona Dispatcher functionality from main.py preserved
"""

import structlog
from typing import List, Optional, Dict

logger = structlog.get_logger()

class PersonaDispatcherAgent:
    """
    Persona Dispatcher Agent for analyzing tasks and selecting most appropriate specialist persona
    COMPLETE ORIGINAL IMPLEMENTATION - THE SPECIALIST SELECTOR
    """
    
    def __init__(self, services: Dict, router, config: Dict):
        self.services = services
        self.router = router
        self.config = config
        logger.info("Persona Dispatcher Agent initialized")

    async def dispatch_persona(self, task: Dict, personas: List) -> Optional:
        """
        Analyze task and select most appropriate specialist persona
        COMPLETE ORIGINAL IMPLEMENTATION
        """
        logger.debug("Dispatching persona for task", 
                    task_description=task.get('description', '')[:100],
                    available_personas=len(personas))
        
        if not personas:
            logger.warning("No personas available for dispatch")
            return None
        
        # Get routing information for dispatcher
        try:
            dispatcher_route = self.router.route('simple_qa', 'free')
            dispatcher_service = self.services.get(dispatcher_route['service'])
            
            if not dispatcher_service:
                logger.error("Dispatcher service not available", service=dispatcher_route['service'])
                return None
            
        except Exception as e:
            logger.error("Failed to route dispatcher request", error=str(e))
            return None
        
        # Prepare persona descriptions
        persona_descriptions = []
        for p in personas:
            # Truncate system prompt for better readability
            prompt_summary = p.system_prompt[:100] + "..." if len(p.system_prompt) > 100 else p.system_prompt
            persona_descriptions.append(f"- {p.name}: {prompt_summary}")
        
        # Create dispatch prompt
        task_description = task.get('description', task.get('task', str(task)))
        
        dispatch_prompt = f"""
        You are a Persona Dispatcher. Analyze this task and select the most appropriate specialist:
        
        Task: {task_description}
        
        Available Personas:
        {chr(10).join(persona_descriptions)}
        
        Respond with ONLY the exact persona name that best matches this task.
        If no persona is specifically suitable, respond with "NONE".
        """
        
        try:
            messages = [{"role": "user", "content": dispatch_prompt}]
            completion = await dispatcher_service.chat_completion(messages, dispatcher_route['model'])
            selected_name = completion.response.strip()
            
            logger.debug("Persona dispatcher response", selected_name=selected_name)
            
            # Handle explicit NONE response
            if selected_name.upper() == "NONE":
                logger.info("Persona dispatcher returned NONE - no suitable specialist found")
                return None
            
            # Find matching persona
            for persona in personas:
                if persona.name.lower() == selected_name.lower():
                    logger.info("Persona dispatched successfully", 
                              task_description=task_description[:50],
                              persona=persona.name)
                    return persona
            
            # No exact match found - log warning and return None
            available_names = [p.name for p in personas]
            logger.warning("Persona dispatcher returned unmatched name", 
                          selected_name=selected_name, 
                          available_personas=available_names)
            return None
            
        except Exception as e:
            logger.error("Persona dispatch failed", error=str(e))
            return None

    async def analyze_task_requirements(self, task: Dict) -> Dict[str, any]:
        """
        Analyze task requirements to inform persona selection
        COMPLETE ORIGINAL FUNCTIONALITY
        """
        logger.debug("Analyzing task requirements", 
                    task_description=task.get('description', '')[:100])
        
        try:
            analysis_route = self.router.route('simple_qa', 'free')
            analysis_service = self.services.get(analysis_route['service'])
            
            task_description = task.get('description', task.get('task', str(task)))
            
            analysis_prompt = f"""
            Analyze this task to determine specialist requirements:
            
            TASK: {task_description}
            
            Determine:
            1. Primary skill domain (technical, creative, analytical, etc.)
            2. Complexity level (low, medium, high)
            3. Required expertise areas
            4. Preferred communication style
            5. Task urgency and priority
            
            Respond with JSON:
            {{
                "primary_domain": "domain name",
                "complexity": "low|medium|high",
                "expertise_areas": ["area1", "area2"],
                "communication_style": "formal|casual|technical",
                "urgency": "low|medium|high",
                "specialist_preference": "specific|generalist|any"
            }}
            """
            
            messages = [{"role": "user", "content": analysis_prompt}]
            completion = await analysis_service.chat_completion(messages, analysis_route['model'])
            
            try:
                import json
                analysis_result = json.loads(completion.response)
                logger.debug("Task analysis completed", 
                           domain=analysis_result.get("primary_domain", "unknown"))
                return analysis_result
            except json.JSONDecodeError:
                logger.warning("Failed to parse task analysis response")
                return {
                    "primary_domain": "general",
                    "complexity": "medium",
                    "expertise_areas": ["general"],
                    "communication_style": "professional",
                    "urgency": "medium",
                    "specialist_preference": "any"
                }
                
        except Exception as e:
            logger.error("Task analysis failed", error=str(e))
            return {
                "primary_domain": "general",
                "complexity": "unknown",
                "expertise_areas": ["general"],
                "communication_style": "professional",
                "urgency": "medium",
                "specialist_preference": "any"
            }

    async def get_persona_capabilities(self, persona) -> Dict[str, any]:
        """
        Analyze persona capabilities for better matching
        COMPLETE ORIGINAL FUNCTIONALITY
        """
        if not persona:
            return {}
            
        logger.debug("Analyzing persona capabilities", persona=persona.name)
        
        try:
            analysis_route = self.router.route('simple_qa', 'free')
            analysis_service = self.services.get(analysis_route['service'])
            
            capability_prompt = f"""
            Analyze this AI persona's capabilities:
            
            PERSONA: {persona.name}
            SYSTEM PROMPT: {persona.system_prompt}
            
            Determine:
            1. Primary specialization areas
            2. Communication style
            3. Complexity handling ability
            4. Domain expertise
            5. Task suitability ratings
            
            Respond with JSON:
            {{
                "specializations": ["area1", "area2"],
                "communication_style": "style description",
                "complexity_rating": "low|medium|high",
                "domain_expertise": ["domain1", "domain2"],
                "best_for": ["task_type1", "task_type2"],
                "avoid_for": ["task_type1", "task_type2"]
            }}
            """
            
            messages = [{"role": "user", "content": capability_prompt}]
            completion = await analysis_service.chat_completion(messages, analysis_route['model'])
            
            try:
                import json
                capability_result = json.loads(completion.response)
                logger.debug("Persona capability analysis completed", 
                           persona=persona.name,
                           specializations=capability_result.get("specializations", []))
                return capability_result
            except json.JSONDecodeError:
                logger.warning("Failed to parse persona capability response")
                return {
                    "specializations": ["general"],
                    "communication_style": "professional",
                    "complexity_rating": "medium",
                    "domain_expertise": ["general"],
                    "best_for": ["general_tasks"],
                    "avoid_for": []
                }
                
        except Exception as e:
            logger.error("Persona capability analysis failed", error=str(e))
            return {
                "specializations": ["general"],
                "communication_style": "professional", 
                "complexity_rating": "medium",
                "domain_expertise": ["general"],
                "best_for": ["general_tasks"],
                "avoid_for": []
            }