"""
Main Application Class - UnifiedAIPortal - COMPLETE VERSION
ALL original application functionality from main.py preserved
"""

import os
import asyncio
import structlog
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .core.config import ConfigManager
from .core.database import DatabaseManager
from .core.router import SimpleIntelligentRouter
from .services.openrouter import OpenSourceAIService
from .services.google_ai import GoogleAIService
from .services.tools import ToolService
from .services.memory import MemoryService
from .orchestration.orchestration_engine import OrchestrationEngine
from .api import (
    projects_router, personas_router, chat_router, 
    orchestration_router, memory_router, system_router
)
from .api.chat import inject_dependencies as inject_chat_deps
from .api.orchestration import inject_orchestration_engine
from .api.memory import inject_memory_service
from .api.system import inject_dependencies as inject_system_deps

logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - COMPLETE ORIGINAL"""
    # Startup
    logger.info("AI Portal startup - initializing learning systems")
    
    # Initialize memory system for the portal instance
    portal = app.state.portal if hasattr(app.state, 'portal') else None
    if portal:
        await portal.initialize_memory_system()
    
    logger.info("Learning AI Portal startup complete")
    yield
    
    # Shutdown
    logger.info("AI Portal shutdown complete")

class UnifiedAIPortal:
    """
    Main AI Portal application class coordinating all systems
    COMPLETE ORIGINAL IMPLEMENTATION - THE LEARNING MACHINE
    """
    
    def __init__(self, config_file: str = "config.yaml"):
        logger.info("Initializing UnifiedAIPortal", config_file=config_file)
        
        # Load configuration
        self.config_manager = ConfigManager(config_file)
        self.config = self.config_manager.config
        
        # Initialize database
        db_url = self.config.get('database_url')
        if not db_url:
            raise ValueError("Database URL not configured. Please check your environment variables.")
        
        logger.info("Connecting to database", url_masked=db_url[:50] + "...")
        self.database_manager = DatabaseManager(db_url)
        self.database_manager.create_tables()
        
        # Initialize services
        self.services = {
            'openrouter': OpenSourceAIService(self.config),
            'google': GoogleAIService(self.config),
            'tools': ToolService(self.config)
        }
        
        # Initialize the agent's learning brain
        self.memory_service = None
        
        # Initialize intelligent router
        self.router = SimpleIntelligentRouter(self.config)
        
        # Initialize orchestration engine (will be set up after memory)
        self.orchestration_engine = None
        
        # Initialize FastAPI application
        self.app = FastAPI(
            title="AI Portal - Learning Machine",
            version="26.2.0",
            description="Advanced AI orchestration system with persistent learning capabilities",
            lifespan=lifespan
        )
        
        # Setup application
        self.setup_app()
        
        logger.info("UnifiedAIPortal initialization complete")

    async def initialize_memory_system(self):
        """Initialize the agent's persistent learning memory - COMPLETE ORIGINAL"""
        logger.info("Initializing memory system")
        try:
            self.memory_service = MemoryService(self.config)
            await self.memory_service.initialize()
            
            # Initialize orchestration engine with memory
            self.orchestration_engine = OrchestrationEngine(
                self.services, self.router, self.config, self.memory_service
            )
            
            # Inject dependencies into API routers
            inject_orchestration_engine(self.orchestration_engine)
            inject_memory_service(self.memory_service)
            
            logger.info("AGENT LEARNING BRAIN ACTIVATED")
        except Exception as e:
            logger.error("Failed to initialize memory system", error=str(e))
            logger.warning("Continuing without persistent memory - system will be amnesiac")
            
            # Initialize orchestration engine without memory
            self.orchestration_engine = OrchestrationEngine(
                self.services, self.router, self.config, None
            )
            inject_orchestration_engine(self.orchestration_engine)

    def setup_app(self):
        """Setup FastAPI application with middleware and routes - COMPLETE ORIGINAL"""
        logger.info("Setting up FastAPI application")
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )
        
        # Store portal instance for lifespan access
        self.app.state.portal = self
        
        # Inject dependencies into API modules
        inject_chat_deps(self.services, self.router, self.config)
        inject_system_deps(self.config, self.services, self.memory_service)
        
        # Include API routers
        self.app.include_router(projects_router)
        self.app.include_router(personas_router)
        self.app.include_router(chat_router)
        self.app.include_router(orchestration_router)
        self.app.include_router(memory_router)
        self.app.include_router(system_router)
        
        logger.info("FastAPI application setup complete")

    def run_preflight_checks(self):
        """Run pre-flight system checks - COMPLETE ORIGINAL"""
        logger.info("Performing pre-flight system checks")
        
        # Check critical environment variables
        required_vars = ["SUPABASE_PASSWORD"]
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        
        if missing_vars:
            logger.error("Missing required environment variables", missing=missing_vars)
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        # Check database connection
        if not self.database_manager.test_connection():
            logger.error("Database connection failed")
            raise ValueError("Database connection failed")
        
        logger.info("Database connection verified")
        
        # Check AI service availability
        if not self.config.get("openrouter_api_key"):
            logger.warning("OpenRouter API key not configured - AI features will be limited")
        
        # Memory system check
        if self.memory_service is None:
            logger.warning("Memory system not initialized - agent will not learn from experience")
            logger.info("To enable learning: ensure proper configuration and dependencies are installed")
        else:
            logger.info("Memory system initialized - agent learning ACTIVE")
        
        logger.info("Pre-flight checks completed successfully")

    def run(self, host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
        """Run the AI Portal application - COMPLETE ORIGINAL"""
        logger.info("Starting AI Portal Learning Machine", version="26.2.0")
        
        # Pre-flight checks
        self.run_preflight_checks()
        
        # Start the server
        logger.info("Launching FastAPI server", host=host, port=port)
        
        try:
            uvicorn.run(
                self.app, 
                host=host, 
                port=port,
                log_level="info",
                access_log=True,
                reload=reload
            )
        except Exception as e:
            logger.error("Failed to start server", error=str(e))
            raise

    def get_app(self):
        """Get the FastAPI application instance"""
        return self.app

    def close(self):
        """Close all connections and cleanup resources"""
        logger.info("Closing AI Portal application")
        
        if self.database_manager:
            self.database_manager.close_engine()
        
        logger.info("AI Portal application closed")