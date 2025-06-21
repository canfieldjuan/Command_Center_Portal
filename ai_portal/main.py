"""
Main entry point for AI Portal application - COMPLETE VERSION
ALL original startup functionality from main.py preserved
"""

import os
import sys
import structlog
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from .app import UnifiedAIPortal

# Setup logging
structlog.configure(processors=[structlog.processors.JSONRenderer()])
logger = structlog.get_logger()

def main():
    """Main entry point for the AI Portal application"""
    logger.info("AI Portal - Learning Machine v26.2.0 Starting Up")
    
    try:
        # Initialize and run the portal
        portal = UnifiedAIPortal()
        portal.run()
        
    except KeyboardInterrupt:
        logger.info("AI Portal shutdown requested by user")
    except Exception as e:
        logger.error("AI Portal startup failed", error=str(e))
        print(f"FATAL ERROR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()