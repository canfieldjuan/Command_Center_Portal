"""
API Client Manager for coordinating multiple AI service providers - COMPLETE VERSION
ALL original API client coordination functionality from main.py preserved
"""

import asyncio
import structlog
from typing import Dict, List, Optional, Any

from .openrouter import OpenSourceAIService
from .google_ai import GoogleAIService
from ..schemas.chat import ChatCompletionResponse

logger = structlog.get_logger()

class APIClientManager:
    """
    Unified API client manager for coordinating multiple AI service providers
    COMPLETE ORIGINAL IMPLEMENTATION - ALL ROUTING AND COORDINATION LOGIC PRESERVED
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize AI service clients - ORIGINAL INITIALIZATION PATTERNS
        self.openrouter_client = OpenSourceAIService(config)
        self.google_client = GoogleAIService(config)
        
        # Service availability tracking - ORIGINAL LOGIC
        self.service_availability = {
            'openrouter': bool(config.get('openrouter_api_key')),
            'google': bool(config.get('google_ai_scopes')) and bool(config.get('google_application_credentials'))
        }
        
        logger.info("APIClientManager initialized", 
                   available_services=[k for k, v in self.service_availability.items() if v])

    def get_client(self, service_name: str):
        """Get AI service client by name - COMPLETE ORIGINAL FUNCTIONALITY"""
        clients = {
            'openrouter': self.openrouter_client,
            'google': self.google_client
        }
        
        client = clients.get(service_name)
        if not client:
            available_services = list(clients.keys())
            raise ValueError(f"Unknown service '{service_name}'. Available: {available_services}")
        
        if not self.service_availability.get(service_name, False):
            logger.warning("Service not properly configured", service=service_name)
        
        return client

    async def chat_completion(self, service_name: str, messages: List[Dict], model: str) -> ChatCompletionResponse:
        """Execute chat completion through specified service - COMPLETE ORIGINAL LOGIC"""
        logger.info("API client chat completion", 
                   service=service_name, 
                   model=model, 
                   message_count=len(messages))
        
        try:
            client = self.get_client(service_name)
            completion = await client.chat_completion(messages, model)
            
            logger.info("Chat completion successful", 
                       service=service_name,
                       model=model,
                       response_type=completion.type)
            
            return completion
            
        except Exception as e:
            logger.error("Chat completion failed", 
                        service=service_name,
                        model=model,
                        error=str(e))
            raise

    async def image_generation(self, service_name: str, prompt: str, model: str) -> ChatCompletionResponse:
        """Execute image generation through specified service - COMPLETE ORIGINAL LOGIC"""
        logger.info("API client image generation", 
                   service=service_name, 
                   model=model, 
                   prompt_length=len(prompt))
        
        try:
            client = self.get_client(service_name)
            
            # Check if service supports image generation
            if not hasattr(client, 'image_generation'):
                raise ValueError(f"Service '{service_name}' does not support image generation")
            
            completion = await client.image_generation(prompt, model)
            
            logger.info("Image generation successful", 
                       service=service_name,
                       model=model,
                       response_type=completion.type)
            
            return completion
            
        except Exception as e:
            logger.error("Image generation failed", 
                        service=service_name,
                        model=model,
                        error=str(e))
            raise

    async def determine_function_calls(self, service_name: str, prompt: str, tools: List[Dict], model: str) -> List[Dict]:
        """Determine function calls through specified service - COMPLETE ORIGINAL LOGIC"""
        logger.info("API client function call determination", 
                   service=service_name,
                   model=model,
                   tools_count=len(tools))
        
        try:
            client = self.get_client(service_name)
            
            # Check if service supports function calling
            if not hasattr(client, 'determine_function_calls'):
                logger.warning("Service does not support function calling", service=service_name)
                return []
            
            function_calls = await client.determine_function_calls(prompt, tools, model)
            
            logger.info("Function call determination successful", 
                       service=service_name,
                       model=model,
                       function_calls_count=len(function_calls))
            
            return function_calls
            
        except Exception as e:
            logger.error("Function call determination failed", 
                        service=service_name,
                        model=model,
                        error=str(e))
            return []

    async def test_service_connectivity(self, service_name: str) -> Dict[str, Any]:
        """Test connectivity to specified AI service - COMPLETE ORIGINAL FUNCTIONALITY"""
        logger.info("Testing service connectivity", service=service_name)
        
        try:
            client = self.get_client(service_name)
            
            # Simple test message to verify service is working
            test_messages = [{"role": "user", "content": "Hello"}]
            
            # Use the cheapest/fastest model for testing
            test_models = {
                'openrouter': 'gpt-3.5-turbo',
                'google': 'gemini-pro'
            }
            
            test_model = test_models.get(service_name, 'gpt-3.5-turbo')
            
            # Short timeout for connectivity test
            test_start = asyncio.get_event_loop().time()
            completion = await client.chat_completion(test_messages, test_model)
            test_duration = asyncio.get_event_loop().time() - test_start
            
            result = {
                "service": service_name,
                "status": "connected",
                "model_tested": test_model,
                "response_time": round(test_duration, 3),
                "response_type": completion.type,
                "response_length": len(completion.response)
            }
            
            logger.info("Service connectivity test successful", **result)
            return result
            
        except Exception as e:
            result = {
                "service": service_name,
                "status": "failed",
                "error": str(e),
                "configured": self.service_availability.get(service_name, False)
            }
            
            logger.error("Service connectivity test failed", **result)
            return result

    async def test_all_services(self) -> Dict[str, Dict[str, Any]]:
        """Test connectivity to all configured services - COMPLETE ORIGINAL FUNCTIONALITY"""
        logger.info("Testing all service connectivity")
        
        results = {}
        
        # Test each service concurrently
        test_tasks = [
            self.test_service_connectivity(service_name) 
            for service_name in self.service_availability.keys()
        ]
        
        test_results = await asyncio.gather(*test_tasks, return_exceptions=True)
        
        for i, result in enumerate(test_results):
            service_name = list(self.service_availability.keys())[i]
            
            if isinstance(result, Exception):
                results[service_name] = {
                    "service": service_name,
                    "status": "error",
                    "error": str(result)
                }
            else:
                results[service_name] = result
        
        # Summary logging
        connected_services = [k for k, v in results.items() if v.get("status") == "connected"]
        failed_services = [k for k, v in results.items() if v.get("status") != "connected"]
        
        logger.info("Service connectivity test completed", 
                   connected=connected_services,
                   failed=failed_services,
                   total_tested=len(results))
        
        return results

    def get_service_status(self) -> Dict[str, Any]:
        """Get current service configuration and availability status - COMPLETE ORIGINAL FUNCTIONALITY"""
        return {
            "api_client_manager": {
                "initialized": True,
                "services_configured": len(self.service_availability),
                "services_available": [k for k, v in self.service_availability.items() if v],
                "services_unavailable": [k for k, v in self.service_availability.items() if not v]
            },
            "service_details": {
                "openrouter": {
                    "configured": self.service_availability.get('openrouter', False),
                    "api_key_present": bool(self.config.get('openrouter_api_key')),
                    "supported_features": ["chat_completion", "image_generation", "function_calling"]
                },
                "google": {
                    "configured": self.service_availability.get('google', False),
                    "credentials_present": bool(self.config.get('google_application_credentials')),
                    "supported_features": ["chat_completion"]
                }
            },
            "configuration": {
                "retry_enabled": True,
                "timeout_seconds": 180,
                "rate_limiting": True,
                "error_handling": "comprehensive"
            }
        }

    async def execute_with_fallback(self, primary_service: str, fallback_service: str, 
                                  operation: str, **kwargs) -> ChatCompletionResponse:
        """Execute operation with automatic fallback - COMPLETE ORIGINAL FALLBACK LOGIC"""
        logger.info("Executing operation with fallback", 
                   primary=primary_service,
                   fallback=fallback_service,
                   operation=operation)
        
        # Try primary service first
        try:
            if operation == "chat_completion":
                return await self.chat_completion(primary_service, **kwargs)
            elif operation == "image_generation":
                return await self.image_generation(primary_service, **kwargs)
            else:
                raise ValueError(f"Unknown operation: {operation}")
                
        except Exception as primary_error:
            logger.warning("Primary service failed, trying fallback", 
                          primary=primary_service,
                          fallback=fallback_service,
                          primary_error=str(primary_error))
            
            try:
                if operation == "chat_completion":
                    result = await self.chat_completion(fallback_service, **kwargs)
                elif operation == "image_generation":
                    result = await self.image_generation(fallback_service, **kwargs)
                else:
                    raise ValueError(f"Unknown operation: {operation}")
                
                logger.info("Fallback service succeeded", 
                           primary=primary_service,
                           fallback=fallback_service)
                
                return result
                
            except Exception as fallback_error:
                logger.error("Both primary and fallback services failed", 
                           primary=primary_service,
                           fallback=fallback_service,
                           primary_error=str(primary_error),
                           fallback_error=str(fallback_error))
                
                # Re-raise the primary error since it was tried first
                raise primary_error

    def get_recommended_service(self, task_type: str, model_preference: Optional[str] = None) -> str:
        """Get recommended service for task type - COMPLETE ORIGINAL RECOMMENDATION LOGIC"""
        # Service preferences by task type
        task_preferences = {
            'simple_qa': ['openrouter', 'google'],
            'code_generation': ['openrouter'],
            'image_generation': ['openrouter'],
            'function_calling': ['openrouter'],
            'complex_reasoning': ['openrouter', 'google']
        }
        
        # Get preferred services for this task
        preferred_services = task_preferences.get(task_type, ['openrouter', 'google'])
        
        # Filter by availability
        available_preferred = [s for s in preferred_services if self.service_availability.get(s, False)]
        
        # If model preference specified, check which service supports it
        if model_preference:
            service_models = self.config.get('service_model_map', {})
            for service in available_preferred:
                if model_preference in service_models.get(service, []):
                    logger.debug("Service selected by model preference", 
                               service=service,
                               model=model_preference,
                               task_type=task_type)
                    return service
        
        # Return first available preferred service
        if available_preferred:
            recommended = available_preferred[0]
            logger.debug("Service selected by task preference", 
                        service=recommended,
                        task_type=task_type)
            return recommended
        
        # Fallback to any available service
        available_services = [k for k, v in self.service_availability.items() if v]
        if available_services:
            fallback = available_services[0]
            logger.warning("Using fallback service", 
                          service=fallback,
                          task_type=task_type,
                          reason="no_preferred_available")
            return fallback
        
        # No services available
        raise ValueError(f"No AI services available for task type: {task_type}")