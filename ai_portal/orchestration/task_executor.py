"""
SAVE AS: ai_portal/orchestration/task_executor.py
CREATE THIS NEW FILE IN THE ORCHESTRATION FOLDER

Task Execution Engine for AI Portal - Phase 3B Critical Extraction
Handles multi-step async task execution, tool coordination, and error recovery
PRESERVES ALL CRITICAL ASYNC PATTERNS AND MEMORY INTEGRATION

File: ai_portal/orchestration/task_executor.py
"""

import time
import json
import asyncio
import structlog
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = structlog.get_logger()

class TaskExecutor:
    """
    Core task execution engine handling multi-step async orchestration
    CRITICAL: Preserves all async patterns, memory integration, and error recovery
    """
    
    def __init__(self, services: Dict, router, config: Dict, memory_service=None):
        self.services = services
        self.router = router
        self.config = config
        self.memory_service = memory_service
        
        logger.info("TaskExecutor initialized with all services")

    async def execute_tool_call(self, tool_name: str, arguments: Dict) -> Any:
        """
        Execute a tool call with comprehensive error handling - PRESERVED FROM MAIN
        """
        logger.info("Executing tool call", tool=tool_name, arguments=arguments)
        
        # Validate tool name
        tool_service = self.services.get('tools')
        if not tool_service:
            raise ValueError("Tool service not available")
        
        if not hasattr(tool_service, tool_name):
            available_tools = [method for method in dir(tool_service) 
                             if not method.startswith('_') and callable(getattr(tool_service, method))]
            raise ValueError(f"Tool '{tool_name}' not found. Available tools: {available_tools}")
        
        # Validate arguments
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a dictionary")
        
        try:
            # Execute the tool
            tool_method = getattr(tool_service, tool_name)
            result = await tool_method(**arguments)
            
            logger.info("Tool call executed successfully", 
                       tool=tool_name,
                       result_type=type(result).__name__)
            
            return result
            
        except TypeError as e:
            # Handle incorrect arguments
            logger.error("Tool call failed due to incorrect arguments", 
                        tool=tool_name,
                        arguments=arguments,
                        error=str(e))
            raise ValueError(f"Tool '{tool_name}' called with incorrect arguments: {str(e)}")
        except Exception as e:
            logger.error("Tool call execution failed", 
                        tool=tool_name,
                        error=str(e))
            raise ValueError(f"Tool '{tool_name}' execution failed: {str(e)}")

    async def execute_task_with_persona(self, task: Dict, persona, user_id: str) -> str:
        """
        Execute a task using the selected persona or default execution - PRESERVED FROM MAIN
        """
        task_description = task.get('description', task.get('task', str(task)))
        task_type = task.get('type', 'simple_qa')
        
        logger.debug("Executing task with persona", 
                    task_description=task_description[:100],
                    task_type=task_type,
                    persona=persona.name if persona else "default")
        
        try:
            # Check if task requires tools
            function_router_service = self.services['openrouter']
            fc_route = self.router.route('function_routing', 'free')
            
            tool_calls = await function_router_service.determine_function_calls(
                task_description,
                self.config.get('available_tools', []),
                fc_route['model']
            )
            
            if tool_calls:
                logger.info("Task requires tool execution", 
                          task_description=task_description[:50],
                          tool_calls_count=len(tool_calls))
                
                # Execute tool calls with proper error handling
                tool_results = await asyncio.gather(*[
                    self.execute_tool_call(call['name'], call['arguments'])
                    for call in tool_calls
                ], return_exceptions=True)
                
                # Handle any tool execution errors
                successful_results = []
                for i, result in enumerate(tool_results):
                    if isinstance(result, Exception):
                        logger.error("Tool call failed", 
                                   tool_call=tool_calls[i],
                                   error=str(result))
                        successful_results.append({"error": str(result), "tool": tool_calls[i]['name']})
                    else:
                        successful_results.append(result)
                
                # Use persona to interpret tool results if available
                if persona:
                    interpretation_prompt = f"""
                    Task: {task_description}
                    
                    Tool Results: {json.dumps(successful_results, indent=2, default=str)}
                    
                    Based on your expertise, provide a comprehensive response to the task using these results.
                    If any tools failed, work around the failures and provide the best possible response.
                    """
                    
                    route_decision = self.router.route(task_type, 'free', persona)
                    service = self.services.get(route_decision['service'])
                    
                    messages = [
                        {"role": "system", "content": persona.system_prompt},
                        {"role": "user", "content": interpretation_prompt}
                    ]
                    
                    completion = await service.chat_completion(messages, route_decision['model'])
                    return completion.response
                else:
                    # No persona available - provide direct tool results with basic interpretation
                    logger.info("No persona available for task interpretation - using default execution")
                    route_decision = self.router.route(task_type, 'free')
                    service = self.services.get(route_decision['service'])
                    
                    interpretation_prompt = f"""
                    Task: {task_description}
                    
                    Tool Results: {json.dumps(successful_results, indent=2, default=str)}
                    
                    Provide a comprehensive response to the task using these tool results.
                    If any tools failed, work around the failures and provide the best possible response.
                    """
                    
                    messages = [{"role": "user", "content": interpretation_prompt}]
                    completion = await service.chat_completion(messages, route_decision['model'])
                    return completion.response
            else:
                # Standard AI response with or without persona
                logger.debug("Task does not require tools, using direct AI completion")
                
                if persona:
                    route_decision = self.router.route(task_type, 'free', persona)
                    service = self.services.get(route_decision['service'])
                    
                    messages = [
                        {"role": "system", "content": persona.system_prompt},
                        {"role": "user", "content": task_description}
                    ]
                else:
                    # No persona available - use default execution
                    logger.debug("No persona available for task - using default execution")
                    route_decision = self.router.route(task_type, 'free')
                    service = self.services.get(route_decision['service'])
                    
                    messages = [{"role": "user", "content": task_description}]
                
                completion = await service.chat_completion(messages, route_decision['model'])
                return completion.response
                
        except Exception as e:
            logger.error("Task execution failed", 
                        task_description=task_description[:50],
                        error=str(e))
            raise ValueError(f"Task execution failed: {str(e)}")

    async def generate_memory_informed_corrective_task(self, original_task: Dict, failed_result: str, 
                                                      failure_reason: str, similar_failures: List[Dict]) -> Dict:
        """
        Generate a corrective task informed by past similar failures - PRESERVED FROM MAIN
        """
        logger.debug("Generating memory-informed corrective task", 
                    original_task=original_task.get('description', '')[:50],
                    similar_failures_count=len(similar_failures))
        
        try:
            corrective_route = self.router.route('simple_qa', 'free')
            corrective_service = self.services.get(corrective_route['service'])
            
            original_description = original_task.get('description', original_task.get('task', str(original_task)))
            
            # Build memory context from similar failures
            memory_context = ""
            if similar_failures:
                memory_context = "\n\nLEARNED FROM PAST FAILURES:\n"
                for i, failure in enumerate(similar_failures[:2]):  # Use top 2 similar failures
                    memory_context += f"Past Failure {i+1}:\n"
                    memory_context += f"- Task: {failure.get('original_task', {}).get('description', '')}\n"
                    memory_context += f"- Failure Reason: {failure.get('failure_reason', '')}\n"
                    memory_context += f"- Successful Correction: {failure.get('corrective_action', {}).get('description', '')}\n\n"
                memory_context += "Apply these lessons to create a better corrective task.\n"
            
            corrective_prompt = f"""
            You are a Memory-Enhanced Adaptive Task Generator with access to past failure corrections.
            A task has failed validation and needs correction using learned experience.
            
            ORIGINAL TASK: {original_description}
            FAILED RESULT: {failed_result}
            FAILURE REASON: {failure_reason}
            {memory_context}
            
            Generate a corrective task that:
            1. Addresses the specific failure reason
            2. Learns from past similar failures
            3. Is more specific and targeted than the original
            4. Applies proven correction strategies
            5. Has clear, measurable success criteria
            
            Respond with EXACTLY this format:
            CORRECTIVE_TASK: [Description of the corrective task]
            OBJECTIVE: [Clear success criteria]
            TYPE: [task type: simple_qa, code_generation, etc.]
            MEMORY_APPLIED: [true/false - whether past failures informed this correction]
            """
            
            messages = [{"role": "user", "content": corrective_prompt}]
            completion = await corrective_service.chat_completion(messages, corrective_route['model'])
            response = completion.response
            
            # Parse the response
            corrective_task = original_description  # Fallback
            objective = "Complete the corrective task"
            task_type = "simple_qa"
            memory_applied = len(similar_failures) > 0
            
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('CORRECTIVE_TASK:'):
                    corrective_task = line.split('CORRECTIVE_TASK:')[1].strip()
                elif line.startswith('OBJECTIVE:'):
                    objective = line.split('OBJECTIVE:')[1].strip()
                elif line.startswith('TYPE:'):
                    task_type = line.split('TYPE:')[1].strip()
                elif line.startswith('MEMORY_APPLIED:'):
                    memory_applied = line.split('MEMORY_APPLIED:')[1].strip().lower() == 'true'
            
            result_dict = {
                "description": corrective_task,
                "objective": objective,
                "type": task_type,
                "is_corrective": True,
                "memory_informed": memory_applied,
                "original_task": original_task,
                "similar_failures_count": len(similar_failures)
            }
            
            logger.debug("Memory-informed corrective task generated", 
                        corrective_task=corrective_task[:100],
                        memory_applied=memory_applied)
            
            return result_dict
            
        except Exception as e:
            logger.error("Memory-informed corrective task generation failed", error=str(e))
            # Fallback to standard corrective task generation
            return await self.generate_corrective_task(original_task, failed_result, failure_reason)

    async def generate_corrective_task(self, original_task: Dict, failed_result: str, failure_reason: str) -> Dict:
        """
        Generate a corrective task when the Critic Agent reports failure - PRESERVED FROM MAIN
        """
        logger.debug("Generating standard corrective task", 
                    original_task=original_task.get('description', '')[:50])
        
        try:
            corrective_route = self.router.route('simple_qa', 'free')
            corrective_service = self.services.get(corrective_route['service'])
            
            original_description = original_task.get('description', original_task.get('task', str(original_task)))
            
            corrective_prompt = f"""
            You are an Adaptive Task Generator. A task has failed validation and needs correction.
            
            ORIGINAL TASK: {original_description}
            FAILED RESULT: {failed_result}
            FAILURE REASON: {failure_reason}
            
            Generate a corrective task that addresses the specific failure reason.
            The corrective task should be more specific and targeted than the original.
            Include clear success criteria to avoid the same failure.
            
            Respond with EXACTLY this format:
            CORRECTIVE_TASK: [Description of the corrective task]
            OBJECTIVE: [Clear success criteria]
            TYPE: [task type: simple_qa, code_generation, etc.]
            """
            
            messages = [{"role": "user", "content": corrective_prompt}]
            completion = await corrective_service.chat_completion(messages, corrective_route['model'])
            response = completion.response
            
            # Parse the response
            corrective_task = original_description  # Fallback
            objective = "Complete the corrective task"
            task_type = "simple_qa"
            
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('CORRECTIVE_TASK:'):
                    corrective_task = line.split('CORRECTIVE_TASK:')[1].strip()
                elif line.startswith('OBJECTIVE:'):
                    objective = line.split('OBJECTIVE:')[1].strip()
                elif line.startswith('TYPE:'):
                    task_type = line.split('TYPE:')[1].strip()
            
            result_dict = {
                "description": corrective_task,
                "objective": objective,
                "type": task_type,
                "is_corrective": True,
                "original_task": original_task
            }
            
            logger.debug("Standard corrective task generated", 
                        corrective_task=corrective_task[:100])
            
            return result_dict
            
        except Exception as e:
            logger.error("Corrective task generation failed", error=str(e))
            return {
                "description": f"Retry: {original_description}",
                "objective": "Complete the task successfully", 
                "type": "simple_qa",
                "is_corrective": True,
                "original_task": original_task
            }

    async def run_orchestration_plan(self, plan: List[Dict], user_id: str, project_id, session) -> Dict:
        """
        Execute the full Reflexive Swarm orchestration with adaptive execution - PRESERVED FROM MAIN
        CRITICAL: Preserves all async patterns, memory integration, and error recovery
        """
        logger.info("Starting orchestration plan execution", 
                   plan_steps=len(plan),
                   user_id=user_id,
                   project_id=str(project_id))
        
        results = []
        adaptive_corrections = 0
        max_corrections = 3
        start_time = time.time()
        
        try:
            # Import here to avoid circular imports
            from .persona_dispatcher import PersonaDispatcherAgent
            from .critic_agent import CriticAgent
            
            # Initialize agents
            persona_dispatcher = PersonaDispatcherAgent(self.services, self.router, self.config)
            critic_agent = CriticAgent(self.services, self.router, self.config)
            
            for step_index, step in enumerate(plan):
                step_start_time = time.time()
                logger.info(f"Executing step {step_index + 1}/{len(plan)}", step=step)
                
                # Persona Dispatcher - Select best persona for this task
                from ..models.persona import Persona
                personas = session.query(Persona).filter(Persona.user_id == user_id).all()
                selected_persona = await persona_dispatcher.dispatch_persona(step, personas)
                
                if selected_persona:
                    logger.info("Persona selected for step", 
                              step=step_index + 1,
                              persona=selected_persona.name)
                else:
                    logger.info("No specific persona selected, using default execution", 
                              step=step_index + 1)
                
                # Execute the task with selected persona
                task_result = await self.execute_task_with_persona(step, selected_persona, user_id)
                
                # Critic Agent - Validate the result
                critic_verdict = await critic_agent.validate_task_result(step, task_result)
                
                step_execution_time = time.time() - step_start_time
                
                if critic_verdict["status"] == "PASS":
                    logger.info(f"Step {step_index + 1} PASSED validation", 
                              verdict=critic_verdict["reasoning"],
                              execution_time=step_execution_time)
                    
                    # Store successful task in memory
                    if self.memory_service:
                        try:
                            await self.memory_service.store_task_success(
                                task=step,
                                result=task_result,
                                persona_used=selected_persona.name if selected_persona else "default"
                            )
                            logger.debug("Task success stored in memory", step=step_index + 1)
                        except Exception as e:
                            logger.error("Failed to store task success in memory", 
                                       step=step_index + 1,
                                       error=str(e))
                    
                    results.append({
                        "step": step_index + 1,
                        "task": step,
                        "result": task_result,
                        "persona": selected_persona.name if selected_persona else "Default",
                        "status": "SUCCESS",
                        "critic_verdict": critic_verdict,
                        "execution_time": step_execution_time
                    })
                    
                else:
                    logger.warning(f"Step {step_index + 1} FAILED validation", 
                                 verdict=critic_verdict["reasoning"],
                                 execution_time=step_execution_time)
                    
                    # Adaptive Execution Loop with Memory-Enhanced Correction
                    if adaptive_corrections < max_corrections:
                        logger.info("Attempting adaptive correction", 
                                  step=step_index + 1,
                                  correction_attempt=adaptive_corrections + 1)
                        
                        # Query memory for similar past failures
                        similar_failures = []
                        if self.memory_service:
                            try:
                                similar_failures = await self.memory_service.query_similar_failures(
                                    task=step,
                                    failure_reason=critic_verdict["reasoning"],
                                    limit=3
                                )
                                if similar_failures:
                                    logger.info("Found similar failures in memory", 
                                              count=len(similar_failures),
                                              step=step_index + 1)
                            except Exception as e:
                                logger.error("Failed to query similar failures", 
                                           step=step_index + 1,
                                           error=str(e))
                        
                        # Generate memory-informed corrective task
                        corrective_task = await self.generate_memory_informed_corrective_task(
                            step, task_result, critic_verdict["reasoning"], similar_failures
                        )
                        
                        logger.info("Generated corrective task", 
                                  step=step_index + 1,
                                  corrective_task=corrective_task.get("description", "")[:100])
                        
                        # Execute corrective task
                        corrective_start_time = time.time()
                        corrective_result = await self.execute_task_with_persona(
                            corrective_task, selected_persona, user_id
                        )
                        
                        # Re-validate
                        corrective_verdict = await critic_agent.validate_task_result(
                            corrective_task, corrective_result
                        )
                        
                        corrective_execution_time = time.time() - corrective_start_time
                        total_step_time = time.time() - step_start_time
                        
                        # Store failure and correction in memory
                        if self.memory_service:
                            try:
                                await self.memory_service.store_task_failure(
                                    task=step,
                                    failed_result=task_result,
                                    failure_reason=critic_verdict["reasoning"],
                                    corrective_action=corrective_task
                                )
                                logger.debug("Task failure and correction stored in memory", 
                                           step=step_index + 1)
                            except Exception as e:
                                logger.error("Failed to store task failure in memory", 
                                           step=step_index + 1,
                                           error=str(e))
                        
                        correction_status = "CORRECTED" if corrective_verdict["status"] == "PASS" else "FAILED"
                        
                        results.append({
                            "step": step_index + 1,
                            "original_task": step,
                            "corrective_task": corrective_task,
                            "result": corrective_result,
                            "persona": selected_persona.name if selected_persona else "Default",
                            "status": correction_status,
                            "critic_verdict": corrective_verdict,
                            "adaptive_correction": True,
                            "memory_informed": len(similar_failures) > 0,
                            "execution_time": total_step_time,
                            "correction_time": corrective_execution_time
                        })
                        
                        adaptive_corrections += 1
                        
                        if corrective_verdict["status"] == "PASS":
                            logger.info(f"Step {step_index + 1} CORRECTED successfully", 
                                      correction_attempt=adaptive_corrections)
                        else:
                            logger.warning(f"Step {step_index + 1} correction FAILED", 
                                         correction_attempt=adaptive_corrections)
                    else:
                        # Max corrections reached, log failure and continue
                        logger.error(f"Step {step_index + 1} FAILED after max corrections", 
                                   max_corrections=max_corrections)
                        
                        results.append({
                            "step": step_index + 1,
                            "task": step,
                            "result": task_result,
                            "persona": selected_persona.name if selected_persona else "Default",
                            "status": "FAILED_MAX_CORRECTIONS",
                            "critic_verdict": critic_verdict,
                            "execution_time": step_execution_time
                        })
            
            # Calculate final statistics
            total_execution_time = time.time() - start_time
            successful_steps = len([r for r in results if r["status"] in ["SUCCESS", "CORRECTED"]])
            failed_steps = len([r for r in results if "FAILED" in r["status"]])
            
            orchestration_result = {
                "status": "completed",
                "total_steps": len(plan),
                "successful_steps": successful_steps,
                "failed_steps": failed_steps,
                "adaptive_corrections_used": adaptive_corrections,
                "total_execution_time": total_execution_time,
                "success_rate": successful_steps / len(plan) if len(plan) > 0 else 0,
                "results": results
            }
            
            logger.info("Orchestration plan execution completed", 
                       total_steps=len(plan),
                       successful_steps=successful_steps,
                       failed_steps=failed_steps,
                       total_time=total_execution_time,
                       success_rate=orchestration_result["success_rate"])
            
            return orchestration_result
            
        except Exception as e:
            logger.error("Orchestration plan execution failed", 
                       error=str(e),
                       completed_steps=len(results))
            raise