#!/usr/bin/env python3
"""
Direct run script for AI Portal - Development convenience script
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ai_portal.main import main

if __name__ == "__main__":
    main()