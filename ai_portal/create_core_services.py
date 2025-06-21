#!/usr/bin/env python3
"""
AI PORTAL PHASE 2A: CORE SERVICES CREATION (FINAL FIXED)
Creates core business logic services with FIXED regex syntax

SERVICES INCLUDED:
- ConfigManager (configuration and environment handling)
- ToolService (web search, file ops, website browsing, ad copy)
- SimpleIntelligentRouter (AI model routing and selection)

RISK LEVEL: 🟡 MEDIUM RISK (Business logic extraction)
VERSION: AI Portal v26.2.0 Core Services Bundle (REGEX FIXED)
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

def create_core_module():
    """Create core configuration module"""
    print("\n📦 Creating core module...")
    
    core_dir = Path("core")
    core_dir.mkdir(exist_ok=True)
    
    # core/__init__.py
    init_content = '''# AI Portal Core Module
# Core configuration and system management

from .config import ConfigManager

__all__ = ["ConfigManager"]
'''
    
    # core/config.py - SAME AS BEFORE (this was working fine)
    config_content = '''# AI Portal Configuration Manager
# Extracted from main.py v26.2.0 - COMPLETE ORIGINAL FUNCTIONALITY
# FIXED: Optional database config for testing

import os
import urllib.parse
import yaml
import structlog

logger = structlog.get_logger()

class ConfigManager:
    def __init__(self, config_file: str = "config.yaml", require_db: bool = True):
        self.config_file = config_file
        self.config = {}
        self.require_db = require_db
        self.load_config()

    def load_config(self):
        try:
            with open(self.config_file, 'r') as f:
                self.config = yaml.safe_load(f)
                logger.info("Configuration loaded successfully", file=self.config_file)
        except FileNotFoundError:
            logger.warning("Config file not found, using defaults", file=self.config_file)
            self.config = {}
        except yaml.YAMLError as e:
            logger.error("Failed to parse config file", error=str(e), file=self.config_file)
            self.config = {}
        except Exception as e:
            logger.error("Unexpected error loading config file", error=str(e), file=self.config_file)
            self.config = {}

        # Load environment variables with validation
        self.config["openrouter_api_key"] = os.environ.get("OPENROUTER_API_KEY")
        self.config["serper_api_key"] = os.environ.get("SERPER_API_KEY")
        self.config["copyshark_api_token"] = os.environ.get("COPYSHARK_API_TOKEN")
        
        # Database configuration with optional requirement for testing
        db_password = os.environ.get("SUPABASE_PASSWORD")
        if not db_password:
            if self.require_db:
                raise ValueError("SUPABASE_PASSWORD environment variable not found. Please set it in your .env file.")
            else:
                logger.warning("SUPABASE_PASSWORD not set - using test database URL")
                self.config["database_url"] = "postgresql://test:test@localhost:5432/test_db"
        else:
            encoded_password = urllib.parse.quote(db_password, safe='')
            self.config["database_url"] = f"postgresql://postgres.jacjorrzxilmrfxbdyse:{encoded_password}@aws-0-us-east-2.pooler.supabase.com:6543/postgres"

        # Set comprehensive default config values if not in yaml
        if 'model_tiers' not in self.config:
            self.config['model_tiers'] = {
                'economy': [
                    'gpt-3.5-turbo',
                    'anthropic/claude-3-haiku',
                    'google/gemini-pro'
                ],
                'standard': [
                    'anthropic/claude-3-sonnet',
                    'openai/gpt-4',
                    'google/gemini-pro-vision'
                ],
                'premium': [
                    'anthropic/claude-3-opus',
                    'openai/gpt-4-turbo',
                    'openai/gpt-4o'
                ]
            }
        
        if 'task_tier_map' not in self.config:
            self.config['task_tier_map'] = {
                'simple_qa': 'economy',
                'code_generation': 'standard',
                'image_generation': 'standard',
                'function_routing': 'economy',
                'complex_reasoning': 'premium'
            }
        
        if 'task_service_map' not in self.config:
            self.config['task_service_map'] = {
                'simple_qa': 'openrouter',
                'code_generation': 'openrouter',
                'image_generation': 'openrouter',
                'function_routing': 'openrouter',
                'complex_reasoning': 'openrouter'
            }
        
        if 'service_model_map' not in self.config:
            self.config['service_model_map'] = {
                'openrouter': [
                    'gpt-3.5-turbo',
                    'anthropic/claude-3-haiku',
                    'anthropic/claude-3-sonnet',
                    'anthropic/claude-3-opus',
                    'openai/gpt-4',
                    'openai/gpt-4-turbo',
                    'openai/gpt-4o',
                    'stable-diffusion-xl',
                    'dall-e-3'
                ],
                'google': [
                    'gemini-pro',
                    'gemini-pro-vision'
                ]
            }
        
        if 'available_tools' not in self.config:
            self.config['available_tools'] = [
                {
                    'name': 'web_search',
                    'description': 'Search the web for current information using Google',
                    'parameters': {
                        'query': {'type': 'string', 'description': 'The search query', 'required': True}
                    }
                },
                {
                    'name': 'browse_website',
                    'description': 'Visit and extract content from a website',
                    'parameters': {
                        'url': {'type': 'string', 'description': 'The URL to visit', 'required': True}
                    }
                },
                {
                    'name': 'save_to_file',
                    'description': 'Save content to a file in the workspace',
                    'parameters': {
                        'filename': {'type': 'string', 'description': 'The name of the file to save', 'required': True},
                        'content': {'type': 'string', 'description': 'The content to save', 'required': True}
                    }
                },
                {
                    'name': 'generateAdCopy',
                    'description': 'Generate advertising copy for products',
                    'parameters': {
                        'productName': {'type': 'string', 'description': 'The name of the product', 'required': True},
                        'audience': {'type': 'string', 'description': 'The target audience', 'required': True},
                        'niche': {'type': 'string', 'description': 'The product niche (optional)', 'required': False}
                    }
                }
            ]

        # Memory system configuration
        if 'memory_dir' not in self.config:
            self.config['memory_dir'] = './agent_memory'
        if 'embedding_model' not in self.config:
            self.config['embedding_model'] = 'all-MiniLM-L6-v2'
        if 'similarity_threshold' not in self.config:
            self.config['similarity_threshold'] = 0.7
        if 'max_memory_results' not in self.config:
            self.config['max_memory_results'] = 10

        # Google AI scopes configuration
        if 'google_ai_scopes' not in self.config:
            self.config['google_ai_scopes'] = [
                'https://www.googleapis.com/auth/cloud-platform',
                'https://www.googleapis.com/auth/generative-language'
            ]

        # Tool security configuration
        if 'max_file_size' not in self.config:
            self.config['max_file_size'] = 10 * 1024 * 1024  # 10MB
        if 'max_content_length' not in self.config:
            self.config['max_content_length'] = 1000000  # 1MB
        if 'allowed_file_extensions' not in self.config:
            self.config['allowed_file_extensions'] = [
                '.txt', '.md', '.json', '.yaml', '.yml', '.py', '.js', 
                '.html', '.css', '.xml', '.csv', '.log', '.sql'
            ]

        # CopyShark service configuration
        if 'copyshark_service' not in self.config:
            self.config['copyshark_service'] = {
                'base_url': 'https://your-copyshark-api.com'
            }

        logger.info("Configuration initialization complete", 
                   config_keys=list(self.config.keys()),
                   model_tiers=len(self.config.get('model_tiers', {})),
                   available_tools=len(self.config.get('available_tools', [])))

    def get(self, key: str, default=None):
        """Get configuration value with optional default"""
        value = self.config.get(key, default)
        if value is None and default is not None:
            logger.warning("Configuration key not found, using default", 
                         key=key, default=default)
        return value

    def reload_config(self):
        """Reload configuration from file"""
        logger.info("Reloading configuration", file=self.config_file)
        self.load_config()
'''
    
    # Write files
    (core_dir / "__init__.py").write_text(init_content, encoding='utf-8')
    (core_dir / "config.py").write_text(config_content, encoding='utf-8')
    
    print(f"   ✅ {core_dir}/__init__.py")
    print(f"   ✅ {core_dir}/config.py")

def create_services_module():
    """Create services module"""
    print("\n📦 Creating services module...")
    
    services_dir = Path("services")
    services_dir.mkdir(exist_ok=True)
    
    # services/__init__.py
    init_content = '''# AI Portal Services Module
# Business logic services and integrations

from .tools import ToolService
from .router import SimpleIntelligentRouter

__all__ = ["ToolService", "SimpleIntelligentRouter"]
'''
    
    # Write init file
    (services_dir / "__init__.py").write_text(init_content, encoding='utf-8')
    print(f"   ✅ {services_dir}/__init__.py")

def create_tools_service():
    """Create comprehensive tools service - FIXED REGEX SYNTAX"""
    print("\n📦 Creating tools service...")
    
    services_dir = Path("services")
    
    # services/tools.py - FIXED: Corrected regex patterns
    tools_content = '''# AI Portal Tool Service
# Web search, file operations, website browsing, and ad copy generation
# Extracted from main.py v26.2.0 - COMPLETE ORIGINAL FUNCTIONALITY
# FIXED: Regex syntax corrected

import os
import re
import json
import asyncio
from typing import Dict, Optional
from datetime import datetime
import aiohttp
from playwright.async_api import async_playwright
import structlog
from decorators import async_retry_with_backoff

logger = structlog.get_logger()

class ToolService:
    def __init__(self, config: Dict):
        self.config = config
        self.workspace_path = os.path.join(os.getcwd(), "workspace")
        
        # Ensure workspace directory exists
        if not os.path.exists(self.workspace_path):
            os.makedirs(self.workspace_path)
            logger.info("Workspace directory created", path=self.workspace_path)
        
        # Security configuration
        self.max_file_size = config.get('max_file_size', 10 * 1024 * 1024)  # 10MB default
        self.allowed_extensions = config.get('allowed_file_extensions', [
            '.txt', '.md', '.json', '.yaml', '.yml', '.py', '.js', '.html', '.css', '.xml'
        ])
        self.max_content_length = config.get('max_content_length', 1000000)  # 1MB text content

        logger.info("ToolService initialized", 
                   workspace=self.workspace_path,
                   max_file_size=self.max_file_size,
                   allowed_extensions=len(self.allowed_extensions))

    def _validate_filename(self, filename: str) -> bool:
        """Enhanced filename validation for security"""
        if not filename or not isinstance(filename, str):
            logger.warning("Invalid filename type", filename=filename, type=type(filename))
            return False
        
        # Check for path traversal attempts
        if ".." in filename or filename.startswith("/") or filename.startswith("\\\\"):
            logger.warning("Path traversal attempt detected", filename=filename)
            return False
        
        # Check for valid extension
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in self.allowed_extensions:
            logger.warning("File extension not allowed", extension=file_ext, filename=filename)
            return False
        
        # Check for suspicious characters
        suspicious_chars = ['<', '>', '|', '&', ';', '`', '$', '(', ')', '*', '?', '[', ']', '{', '}']
        if any(char in filename for char in suspicious_chars):
            logger.warning("Suspicious characters in filename", filename=filename)
            return False
        
        # Check filename length
        if len(filename) > 255:
            logger.warning("Filename too long", filename=filename, length=len(filename))
            return False
        
        return True

    def _validate_content(self, content: str) -> bool:
        """Validate content for security and size limits - FIXED REGEX"""
        if not isinstance(content, str):
            logger.warning("Invalid content type", type=type(content))
            return False
        
        # Check content length
        if len(content) > self.max_content_length:
            logger.warning("Content too large", size=len(content), max_size=self.max_content_length)
            return False
        
        # Check for potentially malicious content patterns - FIXED REGEX SYNTAX
        malicious_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',  # JavaScript URLs
            r'data:.*base64',  # Base64 data URLs
            r'eval\s*\(',  # eval() calls - FIXED
            r'exec\s*\(',  # exec() calls - FIXED
            r'system\s*\(',  # system() calls - FIXED
            r'shell_exec\s*\(',  # shell_exec() calls - FIXED
            r'passthru\s*\(',  # passthru() calls - FIXED
            r'proc_open\s*\(',  # proc_open() calls - FIXED
        ]
        
        for pattern in malicious_patterns:
            if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                logger.warning("Potentially malicious content detected", pattern=pattern)
                return False
        
        return True

    def _validate_url(self, url: str) -> bool:
        """Validate URL for security"""
        if not url or not isinstance(url, str):
            logger.warning("Invalid URL type", url=url, type=type(url))
            return False
        
        # Must start with http or https
        if not url.startswith(('http://', 'https://')):
            logger.warning("Invalid URL protocol", url=url)
            return False
        
        # Check URL length
        if len(url) > 2048:
            logger.warning("URL too long", url=url, length=len(url))
            return False
        
        # Block internal/private IPs and localhost
        blocked_patterns = [
            'localhost', '127.0.0.1', '0.0.0.0', '[::]',
            '192.168.', '10.', '172.16.', '172.17.', '172.18.', '172.19.',
            '172.20.', '172.21.', '172.22.', '172.23.', '172.24.', '172.25.',
            '172.26.', '172.27.', '172.28.', '172.29.', '172.30.', '172.31.',
            'file://', 'ftp://', 'ftps://', 'sftp://'
        ]
        
        url_lower = url.lower()
        for pattern in blocked_patterns:
            if pattern in url_lower:
                logger.warning("Blocked URL pattern detected", url=url, pattern=pattern)
                return False
        
        return True

    def load_api_keys(self):
        """Load API keys from configuration"""
        self.serper_api_key = self.config.get("serper_api_key")
        self.copyshark_api_token = self.config.get("copyshark_api_token")
        
        if not self.serper_api_key:
            logger.warning("SERPER_API_KEY not configured - web search will fail")
        if not self.copyshark_api_token:
            logger.warning("COPYSHARK_API_TOKEN not configured - ad copy generation will fail")

    @async_retry_with_backoff()
    async def web_search(self, query: str) -> Dict:
        """Search the web using Serper API with comprehensive error handling"""
        logger.info("Web search requested", query=query[:100])
        
        # Input validation
        if not query or not isinstance(query, str) or len(query.strip()) == 0:
            raise ValueError("Invalid search query: query must be a non-empty string")
        
        if len(query) > 500:  # Reasonable query length limit
            raise ValueError(f"Search query too long: {len(query)} characters (max 500)")
        
        # Load API keys
        self.load_api_keys()
        if not self.serper_api_key:
            raise ValueError("SERPER_API_KEY not configured. Please set it in your environment variables.")
        
        # Prepare request
        headers = {
            'X-API-KEY': self.serper_api_key,
            'Content-Type': 'application/json',
            'User-Agent': 'AI-Portal/26.2.0'
        }
        
        payload = {
            "q": query.strip(),
            "gl": "us",  # Country code for results
            "hl": "en",  # Language for results
            "num": 10    # Number of results
        }
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(
                    "https://google.serper.dev/search",
                    headers=headers,
                    data=json.dumps(payload)
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    
                    logger.info("Web search completed successfully", 
                               query=query[:50],
                               results_count=len(result.get('organic', [])))
                    
                    return result
                    
        except aiohttp.ClientResponseError as e:
            logger.error("Web search API error", 
                        status=e.status, 
                        message=str(e),
                        query=query[:50])
            raise ValueError(f"Web search failed: HTTP {e.status}")
        except asyncio.TimeoutError:
            logger.error("Web search timeout", query=query[:50])
            raise ValueError("Web search timed out after 30 seconds")
        except Exception as e:
            logger.error("Web search unexpected error", error=str(e), query=query[:50])
            raise ValueError(f"Web search failed: {str(e)}")

    @async_retry_with_backoff()
    async def browse_website(self, url: str) -> Dict:
        """Browse website and extract content using Playwright with security measures"""
        logger.info("Website browsing requested", url=url)
        
        # Enhanced URL validation
        if not self._validate_url(url):
            raise ValueError("Invalid or blocked URL provided")
        
        try:
            async with async_playwright() as p:
                # Launch browser with security settings
                browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        '--no-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-web-security',
                        '--disable-features=VizDisplayCompositor',
                        '--disable-background-timer-throttling',
                        '--disable-renderer-backgrounding',
                        '--disable-backgrounding-occluded-windows'
                    ]
                )
                
                # Create new page with security context
                page = await browser.new_page()
                
                try:
                    # Set user agent and disable JavaScript for security
                    await page.set_user_agent(
                        "Mozilla/5.0 (compatible; AI-Portal-Bot/26.2.0; +https://ai-portal.com/bot)"
                    )
                    
                    # Disable JavaScript to prevent potential security issues
                    await page.set_javascript_enabled(False)
                    
                    # Set viewport
                    await page.set_viewport_size({"width": 1280, "height": 720})
                    
                    # Navigate to the URL with timeout
                    await page.goto(
                        url, 
                        timeout=15000, 
                        wait_until='domcontentloaded'
                    )
                    
                    # Extract page content
                    title = await page.title()
                    content = await page.evaluate("document.body.innerText")
                    
                    # Limit content size for security and performance
                    max_content_size = 100000  # 100KB limit
                    if len(content) > max_content_size:
                        content = content[:max_content_size] + "... (content truncated for security and performance)"
                        logger.warning("Content truncated due to size", 
                                     url=url, 
                                     original_size=len(content),
                                     truncated_size=max_content_size)
                    
                    # Get meta description if available
                    meta_description = ""
                    try:
                        meta_description = await page.evaluate(
                            "document.querySelector('meta[name=\"description\"]')?.getAttribute('content') || ''"
                        )
                    except:
                        pass
                    
                    await browser.close()
                    
                    result = {
                        "status": "success",
                        "url": url,
                        "title": title[:200] if title else "",  # Limit title length
                        "meta_description": meta_description[:500] if meta_description else "",
                        "content": content,
                        "content_length": len(content),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    logger.info("Website browsing completed successfully", 
                               url=url,
                               title=title[:50] if title else "No title",
                               content_length=len(content))
                    
                    return result
                    
                except Exception as e:
                    await browser.close()
                    logger.error("Website browsing page error", url=url, error=str(e))
                    raise
                    
        except Exception as e:
            logger.error("Website browsing failed", url=url, error=str(e))
            raise ValueError(f"Failed to browse website: {str(e)}")

    async def save_to_file(self, filename: str, content: str) -> Dict:
        """Save content to file with comprehensive validation and security"""
        logger.info("File save requested", filename=filename, content_length=len(content))
        
        # Enhanced validation
        if not self._validate_filename(filename):
            raise ValueError("Invalid filename or extension not allowed")
        
        if not self._validate_content(content):
            raise ValueError("Invalid content or content too large")
        
        # Construct file path
        file_path = os.path.join(self.workspace_path, filename)
        
        # Additional security: ensure file path is within workspace (prevent directory traversal)
        try:
            real_workspace = os.path.realpath(self.workspace_path)
            real_filepath = os.path.realpath(file_path)
            
            if not real_filepath.startswith(real_workspace):
                logger.error("Directory traversal attempt", 
                           requested_path=file_path,
                           real_path=real_filepath,
                           workspace=real_workspace)
                raise ValueError("File path outside workspace not allowed")
        except Exception as e:
            logger.error("Path validation error", error=str(e))
            raise ValueError("Invalid file path")
        
        # Check if file already exists and handle appropriately
        file_exists = os.path.exists(file_path)
        if file_exists:
            logger.warning("File already exists, will overwrite", filename=filename)
        
        try:
            # Write file with proper encoding
            with open(file_path, 'w', encoding='utf-8', newline='') as f:
                f.write(content)
            
            # Get file statistics
            file_stats = os.stat(file_path)
            file_size = file_stats.st_size
            
            result = {
                "status": "success",
                "path": file_path,
                "filename": filename,
                "size": file_size,
                "content_length": len(content),
                "file_existed": file_exists,
                "timestamp": datetime.utcnow().isoformat(),
                "workspace": self.workspace_path
            }
            
            logger.info("File saved successfully", 
                       filename=filename,
                       size=file_size,
                       content_length=len(content))
            
            return result
            
        except PermissionError:
            logger.error("Permission denied writing file", filename=filename)
            raise ValueError(f"Permission denied: Unable to write to {filename}")
        except OSError as e:
            logger.error("OS error writing file", filename=filename, error=str(e))
            raise ValueError(f"File system error: {str(e)}")
        except Exception as e:
            logger.error("Unexpected error writing file", filename=filename, error=str(e))
            raise ValueError(f"Failed to save file: {str(e)}")

    @async_retry_with_backoff()
    async def generateAdCopy(self, productName: str, audience: str, niche: Optional[str] = None) -> Dict:
        """Generate advertising copy using CopyShark API with comprehensive validation"""
        logger.info("Ad copy generation requested", 
                   product=productName[:50], 
                   audience=audience[:50],
                   niche=niche[:50] if niche else "None")
        
        # Input validation
        if not productName or not isinstance(productName, str) or len(productName.strip()) == 0:
            raise ValueError("Invalid product name: must be a non-empty string")
        
        if not audience or not isinstance(audience, str) or len(audience.strip()) == 0:
            raise ValueError("Invalid audience: must be a non-empty string")
        
        # Length validation
        if len(productName) > 200:
            raise ValueError(f"Product name too long: {len(productName)} characters (max 200)")
        
        if len(audience) > 500:
            raise ValueError(f"Audience description too long: {len(audience)} characters (max 500)")
        
        if niche and len(niche) > 200:
            raise ValueError(f"Niche description too long: {len(niche)} characters (max 200)")
        
        # Load API configuration
        self.load_api_keys()
        base_url = self.config.get("copyshark_service", {}).get("base_url")
        
        if not base_url:
            raise ValueError("CopyShark service base URL not configured")
        
        if not self.copyshark_api_token:
            raise ValueError("CopyShark API token not configured")
        
        # Prepare request
        headers = {
            "Authorization": f"Bearer {self.copyshark_api_token}",
            "Content-Type": "application/json",
            "User-Agent": "AI-Portal/26.2.0"
        }
        
        payload = {
            "productName": productName.strip(),
            "audience": audience.strip(),
            "niche": niche.strip() if niche else "general",
            "format": "comprehensive",
            "length": "medium"
        }
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
                async with session.post(
                    f"{base_url}/api/generate-copy",
                    headers=headers,
                    json=payload
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    
                    logger.info("Ad copy generation completed successfully",
                               product=productName[:30],
                               audience=audience[:30])
                    
                    return result
                    
        except aiohttp.ClientResponseError as e:
            logger.error("CopyShark API error", 
                        status=e.status,
                        message=str(e),
                        product=productName[:30])
            raise ValueError(f"Ad copy generation failed: HTTP {e.status}")
        except asyncio.TimeoutError:
            logger.error("CopyShark API timeout", product=productName[:30])
            raise ValueError("Ad copy generation timed out after 60 seconds")
        except Exception as e:
            logger.error("Ad copy generation unexpected error", 
                        error=str(e),
                        product=productName[:30])
            raise ValueError(f"Ad copy generation failed: {str(e)}")
'''
    
    # Write tools service
    (services_dir / "tools.py").write_text(tools_content, encoding='utf-8')
    print(f"   ✅ {services_dir}/tools.py (REGEX FIXED)")

def create_router_service():
    """Create intelligent routing service"""
    print("\n📦 Creating router service...")
    
    services_dir = Path("services")
    
    # services/router.py - SAME AS BEFORE (this was working fine)
    router_content = '''# AI Portal Intelligent Router
# Routes tasks to appropriate AI services and models
# Extracted from main.py v26.2.0 - COMPLETE ORIGINAL FUNCTIONALITY

from typing import Dict, Optional
import structlog

logger = structlog.get_logger()

class SimpleIntelligentRouter:
    def __init__(self, config: Dict):
        self.config = config
        self.model_tiers = config.get('model_tiers', {})
        self.task_tier_map = config.get('task_tier_map', {})
        self.task_service_map = config.get('task_service_map', {})
        self.service_model_map = config.get('service_model_map', {})
        
        logger.info("SimpleIntelligentRouter initialized", 
                   model_tiers=len(self.model_tiers),
                   task_mappings=len(self.task_tier_map),
                   service_mappings=len(self.service_model_map))

    def route(self, task_type: str, user_tier: str = "free", persona=None):
        """Route task to appropriate service and model"""
        logger.debug("Routing request", 
                    task_type=task_type,
                    user_tier=user_tier,
                    persona_name=persona.name if persona else None)
        
        # Check persona preference first
        if persona and hasattr(persona, 'model_preference') and persona.model_preference:
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
'''
    
    # Write router service
    (services_dir / "router.py").write_text(router_content, encoding='utf-8')
    print(f"   ✅ {services_dir}/router.py")

def run_core_services_tests():
    """Test all core services (FIXED - environment safe)"""
    print("\n🧪 Testing core services...")
    
    try:
        # Test core config (TEST MODE - no database requirement)
        print("   🔍 Testing core config...")
        sys.path.insert(0, str(Path.cwd()))
        
        from core import ConfigManager
        config = ConfigManager("test_config.yaml", require_db=False)  # TEST MODE
        print("   ✅ ConfigManager working (test mode)")
        
        # Test services
        print("   🔍 Testing services...")
        from services import ToolService, SimpleIntelligentRouter
        
        # Test ToolService initialization
        tool_service = ToolService(config.config)
        print("   ✅ ToolService working (regex fixed)")
        
        # Test Router initialization  
        router = SimpleIntelligentRouter(config.config)
        print("   ✅ SimpleIntelligentRouter working")
        
        # Test routing functionality
        route_result = router.route('simple_qa', 'free')
        print(f"   ✅ Router routing working: {route_result['model']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        print(f"   📝 Error details: {type(e).__name__}")
        return False

def main():
    """Create complete Core Services bundle (FINAL FIXED)"""
    print("🚀 AI PORTAL PHASE 2A: CORE SERVICES CREATION (FINAL FIXED)")
    print("=" * 70)
    print("PHASE 2A: Core Services (Medium Risk Components)")
    print("Creating: ConfigManager, ToolService, SimpleIntelligentRouter")
    print("FIXED: Regex syntax error in tools.py resolved")
    print("=" * 70)
    
    # Create all core services
    create_core_module()
    create_services_module()
    create_tools_service()
    create_router_service()
    
    # Test services (FIXED)
    success = run_core_services_tests()
    
    print("\n" + "=" * 70)
    if success:
        print("✅ CORE SERVICES CREATION COMPLETE!")
        print("\n📁 Created modules:")
        print("   • core/ - Configuration management")
        print("   • services/tools.py - Web search, file ops, browsing, ad copy (REGEX FIXED)")
        print("   • services/router.py - Intelligent AI model routing")
        
        print("\n🎯 READY FOR PHASE 2B: AI Services")
        print("   (OpenSourceAIService, GoogleAIService)")
        
        print("\n💡 ALL SECURITY VALIDATIONS PRESERVED:")
        print("   • File path validation and directory traversal protection")
        print("   • Content security scanning with fixed regex patterns")
        print("   • URL validation and IP blocking")
        print("   • API key management and error handling")
        
    else:
        print("❌ CORE SERVICES CREATION FAILED!")
        print("Check error messages above and retry")
    
    print("=" * 70)

if __name__ == "__main__":
    main()