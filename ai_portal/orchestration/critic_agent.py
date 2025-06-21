"""
Critic Agent for validating task results and ensuring quality - COMPLETE VERSION
ALL original Critic Agent functionality from main.py preserved
"""

import structlog
from typing import Dict, Any

logger = structlog.get_logger()

class CriticAgent:
    """
    Critic Agent for validating task results against objectives with comprehensive analysis
    COMPLETE ORIGINAL IMPLEMENTATION - THE QUALITY VALIDATOR
    """
    
    def __init__(self, services: Dict, router, config: Dict):
        self.services = services
        self.router = router
        self.config = config
        logger.info("Critic Agent initialized")

    async def validate_task_result(self, task: Dict, result: str) -> Dict[str, Any]:
        """
        Validate task result against objective with comprehensive analysis
        COMPLETE ORIGINAL IMPLEMENTATION
        """
        task_description = task.get('description', task.get('task', str(task)))
        task_objective = task.get('objective', 'Complete the task successfully')
        
        logger.debug("Critic agent validating result", 
                    task_description=task_description[:100],
                    result_length=len(result))
        
        try:
            critic_route = self.router.route('simple_qa', 'free')
            critic_service = self.services.get(critic_route['service'])
            
            validation_prompt = f"""
            You are a Critic Agent. Your job is to evaluate if a task result meets its objective.
            
            TASK: {task_description}
            OBJECTIVE: {task_objective}
            
            RESULT TO EVALUATE:
            {result}
            
            Analyze if the result successfully accomplishes the task objective.
            Consider:
            1. Does the result directly address the task requirements?
            2. Is the result complete and comprehensive?
            3. Is the quality sufficient for the objective?
            4. Are there any obvious errors or omissions?
            
            Respond with EXACTLY this format:
            STATUS: [PASS or FAIL]
            REASONING: [Detailed explanation of why it passes or fails]
            
            Be strict but fair in your evaluation. A result should PASS only if it genuinely meets the objective.
            """
            
            messages = [{"role": "user", "content": validation_prompt}]
            
            completion = await critic_service.chat_completion(messages, critic_route['model'])
            response = completion.response
            
            # Parse the response
            status = "FAIL"  # Default to fail for safety
            reasoning = "Could not parse critic response"
            
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('STATUS:'):
                    status_text = line.split('STATUS:')[1].strip().upper()
                    if status_text in ['PASS', 'FAIL']:
                        status = status_text
                elif line.startswith('REASONING:'):
                    reasoning = line.split('REASONING:')[1].strip()
            
            # Validate that we got a proper response
            if status not in ['PASS', 'FAIL']:
                logger.warning("Invalid critic status, defaulting to FAIL", 
                             parsed_status=status,
                             response=response[:200])
                status = 'FAIL'
                reasoning = "Critic agent provided invalid status format"
            
            result_dict = {
                "status": status,
                "reasoning": reasoning,
                "full_response": response
            }
            
            logger.debug("Critic agent validation completed", 
                        status=status,
                        reasoning=reasoning[:100])
            
            return result_dict
            
        except Exception as e:
            logger.error("Critic agent validation failed", 
                        task_description=task_description[:50],
                        error=str(e))
            return {
                "status": "FAIL",
                "reasoning": f"Critic agent error: {str(e)}",
                "full_response": ""
            }

    async def analyze_failure_patterns(self, failed_results: list) -> Dict[str, Any]:
        """
        Analyze patterns in failed results to improve future performance
        COMPLETE ORIGINAL FUNCTIONALITY
        """
        logger.debug("Critic agent analyzing failure patterns", 
                    failed_count=len(failed_results))
        
        if not failed_results:
            return {
                "patterns": [],
                "recommendations": ["No failures to analyze"],
                "common_issues": []
            }
        
        try:
            analysis_route = self.router.route('simple_qa', 'free')
            analysis_service = self.services.get(analysis_route['service'])
            
            # Prepare failure data for analysis
            failure_summaries = []
            for i, failure in enumerate(failed_results[:5]):  # Analyze up to 5 recent failures
                failure_summaries.append(f"Failure {i+1}: {failure.get('reasoning', 'Unknown failure')}")
            
            analysis_prompt = f"""
            You are a Critic Agent analyzing failure patterns to improve system performance.
            
            RECENT FAILURES:
            {chr(10).join(failure_summaries)}
            
            Analyze these failures to identify:
            1. Common failure patterns
            2. Root causes
            3. Systemic issues
            4. Improvement recommendations
            5. Prevention strategies
            
            Respond with JSON:
            {{
                "patterns": ["pattern1", "pattern2"],
                "root_causes": ["cause1", "cause2"],
                "systemic_issues": ["issue1", "issue2"],
                "recommendations": ["rec1", "rec2"],
                "prevention_strategies": ["strategy1", "strategy2"]
            }}
            """
            
            messages = [{"role": "user", "content": analysis_prompt}]
            completion = await analysis_service.chat_completion(messages, analysis_route['model'])
            
            try:
                import json
                analysis_result = json.loads(completion.response)
                logger.debug("Failure pattern analysis completed", 
                           patterns_found=len(analysis_result.get("patterns", [])))
                return analysis_result
            except json.JSONDecodeError:
                logger.warning("Failed to parse failure analysis response")
                return {
                    "patterns": ["Analysis parsing failed"],
                    "root_causes": ["Unknown"],
                    "systemic_issues": ["Analysis system error"],
                    "recommendations": ["Manual review required"],
                    "prevention_strategies": ["Improve parsing system"]
                }
                
        except Exception as e:
            logger.error("Failure pattern analysis failed", error=str(e))
            return {
                "patterns": [f"Analysis error: {str(e)}"],
                "root_causes": ["System error"],
                "systemic_issues": ["Analysis system failure"],
                "recommendations": ["Fix analysis system"],
                "prevention_strategies": ["System diagnostics"]
            }

    async def suggest_improvements(self, task: Dict, result: str, validation: Dict) -> Dict[str, Any]:
        """
        Suggest specific improvements for failed or suboptimal results
        COMPLETE ORIGINAL FUNCTIONALITY
        """
        logger.debug("Critic agent suggesting improvements", 
                    task_description=task.get('description', '')[:50])
        
        try:
            improvement_route = self.router.route('simple_qa', 'free')
            improvement_service = self.services.get(improvement_route['service'])
            
            task_description = task.get('description', task.get('task', str(task)))
            
            improvement_prompt = f"""
            You are a Critic Agent providing improvement suggestions.
            
            TASK: {task_description}
            RESULT: {result}
            VALIDATION: {validation.get('reasoning', 'No validation details')}
            
            Provide specific, actionable improvements:
            1. What should be done differently
            2. Specific areas needing enhancement
            3. Quality improvements needed
            4. Technical corrections required
            5. Process improvements
            
            Respond with JSON:
            {{
                "priority_fixes": ["fix1", "fix2"],
                "quality_improvements": ["improvement1", "improvement2"],
                "technical_corrections": ["correction1", "correction2"],
                "process_changes": ["change1", "change2"],
                "success_criteria": ["criteria1", "criteria2"]
            }}
            """
            
            messages = [{"role": "user", "content": improvement_prompt}]
            completion = await improvement_service.chat_completion(messages, improvement_route['model'])
            
            try:
                import json
                improvement_result = json.loads(completion.response)
                logger.debug("Improvement suggestions generated", 
                           priority_fixes=len(improvement_result.get("priority_fixes", [])))
                return improvement_result
            except json.JSONDecodeError:
                logger.warning("Failed to parse improvement suggestions response")
                return {
                    "priority_fixes": ["Review and revise the result"],
                    "quality_improvements": ["Improve detail and accuracy"],
                    "technical_corrections": ["Verify technical accuracy"],
                    "process_changes": ["Follow task requirements more closely"],
                    "success_criteria": ["Meet all stated objectives"]
                }
                
        except Exception as e:
            logger.error("Improvement suggestion failed", error=str(e))
            return {
                "priority_fixes": [f"Address error: {str(e)}"],
                "quality_improvements": ["System error resolution"],
                "technical_corrections": ["Fix suggestion system"],
                "process_changes": ["Improve error handling"],
                "success_criteria": ["Restore system functionality"]
            }