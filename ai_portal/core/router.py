"""
Intelligent routing system for AI models and services
"""

import structlog
from typing import Dict, Optional, Any

logger = structlog.get_logger()

class SimpleIntelligentRouter:
    """Intelligent router for AI services and models based on task requirements"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_tiers = config.get('model_tiers', {})
        self.task_tier_map = config.get('task_tier_map', {})
        self.task_service_map = config.get('task_service_map', {})
        self.service_model_map = config.get('service_model_map', {})
        
        logger.info("SimpleIntelligentRouter initialized", 
                   model_tiers=len(self.model_tiers),
                   task_mappings=len(self.task_tier_map),
                   service_mappings=len(self.service_model_map))

    def route(self, task_type: str, user_tier: str = "free", persona=None) -> Dict[str, Any]:
        """Route task to appropriate service and model"""
        logger.debug("Routing request", 
                    task_type=task_type,
                    user_tier=user_tier,
                    persona_name=persona.name if persona else None)
        
        # Check persona preference first
        if persona and persona.model_preference:
            model = persona.model_preference
            logger.debug("Checking persona model preference", 
                        persona=persona.name,
                        preferred_model=model)
            
            # Find which service supports this model
            for service, models in self.service_model_map.items():
                if model in models:
                    reasoning = f"Persona preference: {persona.name} prefers {model}"
                    logger.info("Routed via persona preference", 
                              service=service,
                              model=model,
                              persona=persona.name)
                    return {
                        'service': service,
                        'model': model,
                        'reasoning': reasoning
                    }
            
            logger.warning("Persona preferred model not available", 
                         persona=persona.name,
                         preferred_model=model,
                         available_services=list(self.service_model_map.keys()))

        # Default routing logic based on task type and user tier
        service = self.task_service_map.get(task_type, 'openrouter')
        tier_name = self.task_tier_map.get(task_type, 'economy')
        
        # Upgrade tier for pro users
        if user_tier == "pro" and tier_name == "economy":
            tier_name = "standard"
            logger.debug("Upgraded tier for pro user", 
                        original_tier="economy",
                        upgraded_tier=tier_name)
        
        # Get models for this tier
        models_in_tier = self.model_tiers.get(tier_name, [])
        if not models_in_tier:
            logger.warning("No models found for tier", tier=tier_name)
            # Fallback to economy tier
            models_in_tier = self.model_tiers.get('economy', [])
            tier_name = 'economy'
        
        # Get models supported by the target service
        service_models = self.service_model_map.get(service, [])
        if not service_models:
            logger.warning("No models found for service", service=service)
            # Fallback to openrouter
            service = 'openrouter'
            service_models = self.service_model_map.get(service, [])
        
        # Find intersection of tier models and service models
        available_models = [model for model in models_in_tier if model in service_models]
        
        if not available_models:
            logger.error("No compatible models found", 
                        task_type=task_type,
                        tier=tier_name,
                        service=service,
                        tier_models=models_in_tier,
                        service_models=service_models)
            raise ValueError(f"No model found for task '{task_type}' on tier '{tier_name}' with service '{service}'")
        
        # Select the first available model (could be enhanced with load balancing)
        selected_model = available_models[0]
        reasoning = f"Task '{task_type}' on tier '{tier_name}' routed to '{service}' using model '{selected_model}'"
        
        if user_tier == "pro":
            reasoning += " (pro tier)"
        
        logger.info("Routing completed", 
                   service=service,
                   model=selected_model,
                   task_type=task_type,
                   tier=tier_name,
                   user_tier=user_tier)
        
        return {
            'service': service,
            'model': selected_model,
            'reasoning': reasoning
        }

    def get_available_models(self, service: str = None) -> Dict[str, list]:
        """Get available models, optionally filtered by service"""
        if service:
            return {service: self.service_model_map.get(service, [])}
        return self.service_model_map.copy()

    def get_model_tiers(self) -> Dict[str, list]:
        """Get model tier configuration"""
        return self.model_tiers.copy()

    def validate_model_availability(self, model: str, service: str = None) -> bool:
        """Validate if a model is available for a service"""
        if service:
            return model in self.service_model_map.get(service, [])
        
        # Check all services
        for service_models in self.service_model_map.values():
            if model in service_models:
                return True
        return False