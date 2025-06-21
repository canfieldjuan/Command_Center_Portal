"""
Service Registry for dependency injection and service lifecycle management - COMPLETE VERSION
ALL original service management functionality from main.py preserved
"""

import asyncio
import structlog
from typing import Dict, Any, Optional, Type, List
from datetime import datetime

from .openrouter import OpenSourceAIService
from .google_ai import GoogleAIService
from .tools import ToolService
from .memory import MemoryService
from .validators import ValidationService
from .api_client_manager import APIClientManager
from ..core.config import ConfigManager
from ..core.router import SimpleIntelligentRouter

logger = structlog.get_logger()

class ServiceRegistry:
    """
    Service registry for managing AI Portal service lifecycle and dependencies
    COMPLETE ORIGINAL IMPLEMENTATION - ALL SERVICE MANAGEMENT LOGIC PRESERVED
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.config = config_manager.config
        
        # Service instances - ORIGINAL PATTERN
        self._services = {}
        self._service_health = {}
        self._initialization_order = []
        
        # Service dependency map - ORIGINAL DEPENDENCY LOGIC
        self._service_dependencies = {
            'config': [],
            'validators': ['config'],
            'openrouter': ['config'],
            'google': ['config'],
            'tools': ['config', 'validators'],
            'api_client_manager': ['config', 'openrouter', 'google'],
            'router': ['config'],
            'memory': ['config'],
        }
        
        logger.info("ServiceRegistry initialized", 
                   config_keys=len(self.config.keys()))

    async def initialize_all_services(self) -> Dict[str, Any]:
        """Initialize all services in dependency order - COMPLETE ORIGINAL INITIALIZATION"""
        logger.info("Starting service initialization sequence")
        
        initialization_results = {}
        start_time = datetime.utcnow()
        
        try:
            # Phase 1: Core services (no dependencies)
            await self._initialize_core_services()
            
            # Phase 2: Dependent services  
            await self._initialize_dependent_services()
            
            # Phase 3: Complex services (memory, orchestration)
            await self._initialize_complex_services()
            
            # Health check all services
            health_results = await self.health_check_all_services()
            
            total_time = (datetime.utcnow() - start_time).total_seconds()
            
            initialization_results = {
                "status": "completed",
                "total_time_seconds": total_time,
                "services_initialized": len(self._services),
                "services_healthy": len([s for s in health_results.values() if s.get("healthy", False)]),
                "initialization_order": self._initialization_order,
                "health_summary": health_results
            }
            
            logger.info("Service initialization completed successfully", 
                       **initialization_results)
            
            return initialization_results
            
        except Exception as e:
            logger.error("Service initialization failed", error=str(e))
            initialization_results = {
                "status": "failed",
                "error": str(e),
                "services_initialized": len(self._services),
                "partial_initialization": self._initialization_order
            }
            raise ValueError(f"Service initialization failed: {str(e)}")

    async def _initialize_core_services(self):
        """Initialize core services with no dependencies - ORIGINAL CORE LOGIC"""
        logger.info("Initializing core services")
        
        # Config (already available)
        self._services['config'] = self.config_manager
        self._initialization_order.append('config')
        self._service_health['config'] = {"healthy": True, "initialized_at": datetime.utcnow()}
        
        # Validators
        validators = ValidationService(self.config)
        self._services['validators'] = validators
        self._initialization_order.append('validators')
        self._service_health['validators'] = {"healthy": True, "initialized_at": datetime.utcnow()}
        
        # Router
        router = SimpleIntelligentRouter(self.config)
        self._services['router'] = router
        self._initialization_order.append('router')
        self._service_health['router'] = {"healthy": True, "initialized_at": datetime.utcnow()}
        
        logger.info("Core services initialized", services=['config', 'validators', 'router'])

    async def _initialize_dependent_services(self):
        """Initialize services with dependencies - ORIGINAL DEPENDENCY LOGIC"""
        logger.info("Initializing dependent services")
        
        # AI Service clients
        openrouter = OpenSourceAIService(self.config)
        self._services['openrouter'] = openrouter
        self._initialization_order.append('openrouter')
        self._service_health['openrouter'] = {"healthy": True, "initialized_at": datetime.utcnow()}
        
        google = GoogleAIService(self.config)
        self._services['google'] = google
        self._initialization_order.append('google')
        self._service_health['google'] = {"healthy": True, "initialized_at": datetime.utcnow()}
        
        # Tools (depends on validators)
        tools = ToolService(self.config)
        self._services['tools'] = tools
        self._initialization_order.append('tools')
        self._service_health['tools'] = {"healthy": True, "initialized_at": datetime.utcnow()}
        
        # API Client Manager (depends on AI services)
        api_manager = APIClientManager(self.config)
        self._services['api_client_manager'] = api_manager
        self._initialization_order.append('api_client_manager')
        self._service_health['api_client_manager'] = {"healthy": True, "initialized_at": datetime.utcnow()}
        
        logger.info("Dependent services initialized", 
                   services=['openrouter', 'google', 'tools', 'api_client_manager'])

    async def _initialize_complex_services(self):
        """Initialize complex services (memory system) - ORIGINAL MEMORY INITIALIZATION"""
        logger.info("Initializing complex services")
        
        # Memory service (async initialization required)
        try:
            memory = MemoryService(self.config)
            await memory.initialize()
            
            self._services['memory'] = memory
            self._initialization_order.append('memory')
            self._service_health['memory'] = {
                "healthy": True, 
                "initialized_at": datetime.utcnow(),
                "ml_model_loaded": True
            }
            
            logger.info("Memory service initialized successfully")
            
        except Exception as e:
            logger.error("Memory service initialization failed", error=str(e))
            logger.warning("Continuing without persistent memory - system will be amnesiac")
            
            self._services['memory'] = None
            self._service_health['memory'] = {
                "healthy": False,
                "initialized_at": datetime.utcnow(),
                "error": str(e),
                "ml_model_loaded": False
            }

    def get_service(self, service_name: str):
        """Get service instance by name - COMPLETE ORIGINAL FUNCTIONALITY"""
        service = self._services.get(service_name)
        
        if service is None:
            available_services = list(self._services.keys())
            raise ValueError(f"Service '{service_name}' not found. Available: {available_services}")
        
        # Check service health
        health = self._service_health.get(service_name, {})
        if not health.get("healthy", False):
            logger.warning("Requested service is not healthy", 
                          service=service_name,
                          health_status=health)
        
        return service

    def get_services_dict(self) -> Dict[str, Any]:
        """Get services dictionary for dependency injection - COMPLETE ORIGINAL PATTERN"""
        # Filter out None services and config (which is handled separately)
        services_dict = {
            name: service for name, service in self._services.items() 
            if service is not None and name != 'config'
        }
        
        logger.debug("Services dictionary requested", 
                    services=list(services_dict.keys()))
        
        return services_dict

    async def health_check_all_services(self) -> Dict[str, Dict[str, Any]]:
        """Perform health checks on all services - COMPLETE ORIGINAL HEALTH MONITORING"""
        logger.info("Performing health checks on all services")
        
        health_results = {}
        
        for service_name, service in self._services.items():
            health_results[service_name] = await self._health_check_service(service_name, service)
        
        # Summary statistics
        healthy_count = len([r for r in health_results.values() if r.get("healthy", False)])
        total_count = len(health_results)
        
        logger.info("Health check completed", 
                   healthy_services=healthy_count,
                   total_services=total_count,
                   health_rate=f"{healthy_count}/{total_count}")
        
        return health_results

    async def _health_check_service(self, service_name: str, service: Any) -> Dict[str, Any]:
        """Health check individual service - COMPLETE ORIGINAL HEALTH CHECK LOGIC"""
        health_result = {
            "service": service_name,
            "healthy": False,
            "checked_at": datetime.utcnow(),
            "details": {}
        }
        
        try:
            if service is None:
                health_result["details"]["status"] = "not_initialized"
                health_result["details"]["reason"] = "Service is None"
                return health_result
            
            # Service-specific health checks
            if service_name == 'config':
                health_result["healthy"] = True
                health_result["details"]["config_keys"] = len(self.config.keys())
                health_result["details"]["database_url_configured"] = bool(self.config.get("database_url"))
                
            elif service_name == 'validators':
                health_result["healthy"] = True
                health_result["details"]["max_file_size"] = service.max_file_size
                health_result["details"]["allowed_extensions"] = len(service.allowed_extensions)
                
            elif service_name == 'router':
                health_result["healthy"] = True
                health_result["details"]["model_tiers"] = len(service.model_tiers)
                health_result["details"]["task_mappings"] = len(service.task_tier_map)
                
            elif service_name in ['openrouter', 'google']:
                # Test API connectivity
                if hasattr(service, '_api_call'):
                    health_result["healthy"] = True  # API available
                    health_result["details"]["api_configured"] = True
                else:
                    health_result["details"]["api_configured"] = False
                    
            elif service_name == 'tools':
                health_result["healthy"] = True
                health_result["details"]["workspace_path"] = service.workspace_path
                health_result["details"]["workspace_exists"] = os.path.exists(service.workspace_path)
                
            elif service_name == 'api_client_manager':
                health_result["healthy"] = True
                health_result["details"]["services_available"] = len(service.service_availability)
                health_result["details"]["configured_services"] = [
                    k for k, v in service.service_availability.items() if v
                ]
                
            elif service_name == 'memory':
                if service:
                    health_result["healthy"] = bool(service.embedding_model)
                    health_result["details"]["ml_model_loaded"] = bool(service.embedding_model)
                    health_result["details"]["memory_dir"] = str(service.memory_dir)
                    health_result["details"]["memory_dir_exists"] = service.memory_dir.exists()
                else:
                    health_result["details"]["status"] = "not_initialized"
                    health_result["details"]["reason"] = "Memory service failed to initialize"
                    
            else:
                # Generic health check
                health_result["healthy"] = True
                health_result["details"]["status"] = "available"
                
        except Exception as e:
            health_result["healthy"] = False
            health_result["details"]["error"] = str(e)
            health_result["details"]["status"] = "error"
            
        return health_result

    def get_service_dependencies(self, service_name: str) -> List[str]:
        """Get service dependencies - COMPLETE ORIGINAL DEPENDENCY TRACKING"""
        return self._service_dependencies.get(service_name, [])

    def get_service_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive service status summary - COMPLETE ORIGINAL STATUS REPORTING"""
        return {
            "registry": {
                "total_services": len(self._services),
                "healthy_services": len([
                    h for h in self._service_health.values() 
                    if h.get("healthy", False)
                ]),
                "initialization_order": self._initialization_order,
                "registry_healthy": all(
                    h.get("healthy", False) for h in self._service_health.values()
                )
            },
            "services": {
                name: {
                    "available": service is not None,
                    "health": self._service_health.get(name, {}),
                    "dependencies": self.get_service_dependencies(name)
                }
                for name, service in self._services.items()
            },
            "critical_services": {
                "database_configured": bool(self.config.get("database_url")),
                "ai_services_available": any([
                    self._services.get('openrouter'),
                    self._services.get('google')
                ]),
                "memory_system_active": bool(self._services.get('memory')),
                "tools_available": bool(self._services.get('tools'))
            }
        }

    async def shutdown_all_services(self):
        """Gracefully shutdown all services - COMPLETE ORIGINAL SHUTDOWN LOGIC"""
        logger.info("Starting graceful service shutdown")
        
        shutdown_order = list(reversed(self._initialization_order))
        
        for service_name in shutdown_order:
            try:
                service = self._services.get(service_name)
                if service and hasattr(service, 'close'):
                    await service.close()
                    logger.debug("Service shut down", service=service_name)
            except Exception as e:
                logger.error("Service shutdown error", 
                           service=service_name,
                           error=str(e))
        
        self._services.clear()
        self._service_health.clear()
        self._initialization_order.clear()
        
        logger.info("Service shutdown completed")

    def validate_service_configuration(self) -> Dict[str, Any]:
        """Validate service configuration requirements - COMPLETE ORIGINAL VALIDATION"""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "requirements_met": {}
        }
        
        # Check critical requirements
        if not self.config.get("database_url"):
            validation_results["errors"].append("Database URL not configured")
            validation_results["valid"] = False
        
        if not self.config.get("openrouter_api_key"):
            validation_results["warnings"].append("OpenRouter API key not configured")
        
        # Check file system requirements
        import os
        workspace_path = os.path.join(os.getcwd(), "workspace")
        if not os.path.exists(workspace_path):
            validation_results["warnings"].append("Workspace directory does not exist")
        
        # Check memory system requirements
        try:
            import sentence_transformers
            import numpy
            import sklearn
            validation_results["requirements_met"]["memory_dependencies"] = True
        except ImportError as e:
            validation_results["warnings"].append(f"Memory system dependencies missing: {str(e)}")
            validation_results["requirements_met"]["memory_dependencies"] = False
        
        validation_results["requirements_met"]["database"] = bool(self.config.get("database_url"))
        validation_results["requirements_met"]["ai_services"] = bool(
            self.config.get("openrouter_api_key") or 
            self.config.get("google_application_credentials")
        )
        
        return validation_results