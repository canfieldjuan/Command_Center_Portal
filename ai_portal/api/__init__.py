"""
API package for FastAPI routes and endpoints - COMPLETE VERSION
ALL original endpoint functionality preserved
"""

from .projects import router as projects_router
from .personas import router as personas_router
from .chat import router as chat_router
from .orchestration import router as orchestration_router
from .memory import router as memory_router
from .system import router as system_router

__all__ = [
    "projects_router",
    "personas_router", 
    "chat_router",
    "orchestration_router",
    "memory_router",
    "system_router"
]