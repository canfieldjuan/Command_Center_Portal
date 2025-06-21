"""
Tool validation service for security-critical input validation - COMPLETE VERSION
ALL original validation functionality from main.py and tools.py preserved
"""

import os
import re
import structlog
from typing import Dict, List, Optional

logger = structlog.get_logger()

class ValidationService:
    """
    Security-critical validation service for all tool inputs and system parameters
    COMPLETE ORIGINAL IMPLEMENTATION - ALL SECURITY PATTERNS PRESERVED
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Security configuration - COMPLETE ORIGINAL CONFIGURATION
        self.max_file_size = config.get('max_file_size', 10 * 1024 * 1024)  # 10MB default
        self.allowed_extensions = config.get('allowed_file_extensions', [
            '.txt', '.md', '.json', '.yaml', '.yml', '.py', '.js', '.html', '.css', '.xml'
        ])
        self.max_content_length = config.get('max_content_length', 1000000)  # 1MB text content

        logger.info("ValidationService initialized", 
                   max_file_size=self.max_file_size,
                   allowed_extensions=len(self.allowed_extensions))

    def validate_filename(self, filename: str) -> bool:
        """Enhanced filename validation for security - COMPLETE ORIGINAL VALIDATION"""
        if not filename or not isinstance(filename, str):
            logger.warning("Invalid filename type", filename=filename, type=type(filename))
            return False
        
        # Check for path traversal attempts - ORIGINAL SECURITY
        if ".." in filename or filename.startswith("/") or filename.startswith("\\"):
            logger.warning("Path traversal attempt detected", filename=filename)
            return False
        
        # Check for valid extension - ORIGINAL VALIDATION
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in self.allowed_extensions:
            logger.warning("File extension not allowed", extension=file_ext, filename=filename)
            return False
        
        # Check for suspicious characters - ORIGINAL SECURITY
        suspicious_chars = ['<', '>', '|', '&', ';', '`', '$', '(', ')', '*', '?', '[', ']', '{', '}']
        if any(char in filename for char in suspicious_chars):
            logger.warning("Suspicious characters in filename", filename=filename)
            return False
        
        # Check filename length - ORIGINAL VALIDATION
        if len(filename) > 255:
            logger.warning("Filename too long", filename=filename, length=len(filename))
            return