"""
Tool service for external integrations (web search, file operations, etc.) - COMPLETE VERSION
ALL original tool functionality and security from main.py preserved
"""

import os
import re
import json
import asyncio
import aiohttp
import structlog
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from playwright.async_api import async_playwright

from ..core.decorators import async_retry_with_backoff

logger = structlog.get_logger()

class ToolService:
    """Service for executing external tools and integrations - COMPLETE ORIGINAL"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.workspace_path = os.path.join(os.getcwd(), "workspace")
        
        # Ensure workspace directory exists - ORIGINAL LOGIC
        if not os.path.exists(self.workspace_path):
            os.makedirs(self.workspace_path)
            logger.info("Workspace directory created", path=self.workspace_path)
        
        # Security configuration - COMPLETE ORIGINAL CONFIGURATION
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
        """Enhanced filename validation for security - COMPLETE ORIGINAL VALIDATION"""
        if not filename or not isinstance(filename, str):
            logger.warning("Invalid filename type", filename=filename, type=type(filename))
            return False
        
        # Check for path traversal attempts - ORIGINAL SECURITY
        if ".." in filename or filename.startswith("/") or filename.startswith("\"):
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
            return False
        
        return True

    def _validate_content(self, content: str) -> bool:
        """Validate content for security and size limits - COMPLETE ORIGINAL VALIDATION"""
        if not isinstance(content, str):
            logger.warning("Invalid content type", type=type(content))
            return False
        
        # Check content length - ORIGINAL VALIDATION
        if len(content) > self.max_content_length:
            logger.warning("Content too large", size=len(content), max_size=self.max_content_length)
            return False
        
        # Check for potentially malicious content patterns - ORIGINAL SECURITY
        malicious_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',  # JavaScript URLs
            r'data:.*base64',  # Base64 data URLs
            r'eval\s*\(',  # eval() calls
            r'exec\s*\(',  # exec() calls
            r'system\s*\(',  # system() calls
            r'shell_exec\s*\(',  # shell_exec() calls
            r'passthru\s*\(',  # passthru() calls
            r'proc_open\s*\(',  # proc_open() calls
        ]
        
        for pattern in malicious_patterns:
            if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                logger.warning("Potentially malicious content detected", pattern=pattern)
                return False
        
        return True

    def _validate_url(self, url: str) -> bool:
        """Validate URL for security - COMPLETE ORIGINAL VALIDATION"""
        if not url or not isinstance(url, str):
            logger.warning("Invalid URL type", url=url, type=type(url))
            return False
        
        # Must start with http or https - ORIGINAL VALIDATION
        if not url.startswith(('http://', 'https://')):
            logger.warning("Invalid URL protocol", url=url)
            return False
        
        # Check URL length - ORIGINAL VALIDATION
        if len(url) > 2048:
            logger.warning("URL too long", url=url, length=len(url))
            return False
        
        # Block internal/private IPs and localhost - ORIGINAL SECURITY
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
        """Load API keys from configuration - ORIGINAL FUNCTIONALITY"""
        self.serper_api_key = self.config.get("serper_api_key")
        self.copyshark_api_token = self.config.get("copyshark_api_token")
        
        if not self.serper_api_key:
            logger.warning("SERPER_API_KEY not configured - web search will fail")
        if not self.copyshark_api_token:
            logger.warning("COPYSHARK_API_TOKEN not configured - ad copy generation will fail")

    @async_retry_with_backoff()
    async def web_search(self, query: str) -> Dict:
        """Search the web using Serper API with comprehensive error handling - COMPLETE ORIGINAL"""
        logger.info("Web search requested", query=query[:100])
        
        # Input validation - COMPLETE ORIGINAL VALIDATION
        if not query or not isinstance(query, str) or len(query.strip()) == 0:
            raise ValueError("Invalid search query: query must be a non-empty string")
        
        if len(query) > 500:  # Reasonable query length limit
            raise ValueError(f"Search query too long: {len(query)} characters (max 500)")
        
        # Load API keys - ORIGINAL LOGIC
        self.load_api_keys()
        if not self.serper_api_key:
            raise ValueError("SERPER_API_KEY not configured. Please set it in your environment variables.")
        
        # Prepare request - ORIGINAL CONFIGURATION
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
        """Browse website and extract content using Playwright with security measures - COMPLETE ORIGINAL"""
        logger.info("Website browsing requested", url=url)
        
        # Enhanced URL validation - ORIGINAL VALIDATION
        if not self._validate_url(url):
            raise ValueError("Invalid or blocked URL provided")
        
        try:
            async with async_playwright() as p:
                # Launch browser with security settings - ORIGINAL CONFIGURATION
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
                
                # Create new page with security context - ORIGINAL SECURITY
                page = await browser.new_page()
                
                try:
                    # Set user agent and disable JavaScript for security - ORIGINAL SECURITY
                    await page.set_user_agent(
                        "Mozilla/5.0 (compatible; AI-Portal-Bot/26.2.0; +https://ai-portal.com/bot)"
                    )
                    
                    # Disable JavaScript to prevent potential security issues - ORIGINAL SECURITY
                    await page.set_javascript_enabled(False)
                    
                    # Set viewport - ORIGINAL CONFIGURATION
                    await page.set_viewport_size({"width": 1280, "height": 720})
                    
                    # Navigate to the URL with timeout - ORIGINAL CONFIGURATION
                    await page.goto(
                        url, 
                        timeout=15000, 
                        wait_until='domcontentloaded'
                    )
                    
                    # Extract page content - ORIGINAL EXTRACTION LOGIC
                    title = await page.title()
                    content = await page.evaluate("document.body.innerText")
                    
                    # Limit content size for security and performance - ORIGINAL SECURITY
                    max_content_size = 100000  # 100KB limit
                    if len(content) > max_content_size:
                        content = content[:max_content_size] + "... (content truncated for security and performance)"
                        logger.warning("Content truncated due to size", 
                                     url=url, 
                                     original_size=len(content),
                                     truncated_size=max_content_size)
                    
                    # Get meta description if available - ORIGINAL FUNCTIONALITY
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
        """Save content to file with comprehensive validation and security - COMPLETE ORIGINAL"""
        logger.info("File save requested", filename=filename, content_length=len(content))
        
        # Enhanced validation - ORIGINAL VALIDATION
        if not self._validate_filename(filename):
            raise ValueError("Invalid filename or extension not allowed")
        
        if not self._validate_content(content):
            raise ValueError("Invalid content or content too large")
        
        # Construct file path - ORIGINAL LOGIC
        file_path = os.path.join(self.workspace_path, filename)
        
        # Additional security: ensure file path is within workspace (prevent directory traversal) - ORIGINAL SECURITY
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
        
        # Check if file already exists and handle appropriately - ORIGINAL LOGIC
        file_exists = os.path.exists(file_path)
        if file_exists:
            logger.warning("File already exists, will overwrite", filename=filename)
        
        try:
            # Write file with proper encoding - ORIGINAL LOGIC
            with open(file_path, 'w', encoding='utf-8', newline='') as f:
                f.write(content)
            
            # Get file statistics - ORIGINAL FUNCTIONALITY
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
        """Generate advertising copy using CopyShark API with comprehensive validation - COMPLETE ORIGINAL"""
        logger.info("Ad copy generation requested", 
                   product=productName[:50], 
                   audience=audience[:50],
                   niche=niche[:50] if niche else "None")
        
        # Input validation - COMPLETE ORIGINAL VALIDATION
        if not productName or not isinstance(productName, str) or len(productName.strip()) == 0:
            raise ValueError("Invalid product name: must be a non-empty string")
        
        if not audience or not isinstance(audience, str) or len(audience.strip()) == 0:
            raise ValueError("Invalid audience: must be a non-empty string")
        
        # Length validation - ORIGINAL VALIDATION
        if len(productName) > 200:
            raise ValueError(f"Product name too long: {len(productName)} characters (max 200)")
        
        if len(audience) > 500:
            raise ValueError(f"Audience description too long: {len(audience)} characters (max 500)")
        
        if niche and len(niche) > 200:
            raise ValueError(f"Niche description too long: {len(niche)} characters (max 200)")
        
        # Load API configuration - ORIGINAL LOGIC
        self.load_api_keys()
        base_url = self.config.get("copyshark_service", {}).get("base_url")
        
        if not base_url:
            raise ValueError("CopyShark service base URL not configured")
        
        if not self.copyshark_api_token:
            raise ValueError("CopyShark API token not configured")
        
        # Prepare request - ORIGINAL CONFIGURATION
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