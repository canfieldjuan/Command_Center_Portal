"""
OpenRouter AI service integration for multiple LLM providers - COMPLETE VERSION
ALL original OpenRouter functionality from main.py preserved
"""

import asyncio
import aiohttp
import json
import structlog
from typing import Dict, List

from ..core.decorators import async_retry_with_backoff
from ..schemas.chat import ChatCompletionResponse

logger = structlog.get_logger()

class OpenSourceAIService:
    """OpenRouter service for accessing multiple AI models through a single API - COMPLETE ORIGINAL"""
    
    def __init__(self, config: Dict):
        self.api_key = config.get('openrouter_api_key')
        self.base_url = "https://openrouter.ai/api/v1"
        self.timeout = 180  # 3 minutes timeout
        
        if not self.api_key:
            logger.warning("OpenRouter API key not configured")
        else:
            logger.info("OpenSourceAIService initialized with API key")

    @async_retry_with_backoff()
    async def _api_call(self, endpoint: str, payload: Dict):
        """Make API call to OpenRouter with comprehensive error handling - COMPLETE ORIGINAL"""
        if not self.api_key:
            raise ValueError("OpenRouter API key not configured. Please set OPENROUTER_API_KEY environment variable.")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://ai-portal.com",
            "X-Title": "AI Portal"
        }
        
        url = f"{self.base_url}{endpoint}"
        
        logger.debug("OpenRouter API call", endpoint=endpoint, payload_size=len(json.dumps(payload)))
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    result = await response.json()
                    
                    logger.debug("OpenRouter API call successful", 
                                endpoint=endpoint,
                                status=response.status)
                    
                    return result
                    
        except aiohttp.ClientResponseError as e:
            error_detail = "Unknown error"
            try:
                error_body = await e.response.json()
                error_detail = error_body.get('error', {}).get('message', str(e))
            except:
                error_detail = str(e)
            
            logger.error("OpenRouter API error", 
                        endpoint=endpoint,
                        status=e.status,
                        error=error_detail)
            
            if e.status == 401:
                raise ValueError("OpenRouter API authentication failed. Check your API key.")
            elif e.status == 429:
                raise ValueError("OpenRouter API rate limit exceeded. Please retry later.")
            elif e.status == 402:
                raise ValueError("OpenRouter API billing issue. Check your account balance.")
            else:
                raise ValueError(f"OpenRouter API error: {error_detail}")
                
        except asyncio.TimeoutError:
            logger.error("OpenRouter API timeout", endpoint=endpoint)
            raise ValueError(f"OpenRouter API call timed out after {self.timeout} seconds")
        except Exception as e:
            logger.error("OpenRouter API unexpected error", endpoint=endpoint, error=str(e))
            raise ValueError(f"OpenRouter API call failed: {str(e)}")

    async def chat_completion(self, messages: List[Dict], model: str) -> ChatCompletionResponse:
        """Generate chat completion using OpenRouter - COMPLETE ORIGINAL"""
        logger.info("Chat completion requested", model=model, message_count=len(messages))
        
        # Validate inputs - ORIGINAL VALIDATION
        if not messages or not isinstance(messages, list):
            raise ValueError("Messages must be a non-empty list")
        
        if not model or not isinstance(model, str):
            raise ValueError("Model must be a non-empty string")
        
        # Validate message format - ORIGINAL VALIDATION
        for i, message in enumerate(messages):
            if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
                raise ValueError(f"Message {i} must have 'role' and 'content' fields")
            
            if message['role'] not in ['system', 'user', 'assistant']:
                raise ValueError(f"Message {i} has invalid role: {message['role']}")
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 0.9,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }
        
        try:
            response = await self._api_call("/chat/completions", payload)
            
            if not response.get('choices') or len(response['choices']) == 0:
                raise ValueError("No response choices returned from OpenRouter")
            
            choice = response['choices'][0]
            if not choice.get('message') or not choice['message'].get('content'):
                raise ValueError("Invalid response format from OpenRouter")
            
            result = ChatCompletionResponse(
                type='text',
                response=choice['message']['content']
            )
            
            logger.info("Chat completion successful", 
                       model=model,
                       response_length=len(result.response))
            
            return result
            
        except Exception as e:
            logger.error("Chat completion failed", model=model, error=str(e))
            raise

    async def image_generation(self, prompt: str, model: str) -> ChatCompletionResponse:
        """Generate image using OpenRouter - COMPLETE ORIGINAL"""
        logger.info("Image generation requested", model=model, prompt_length=len(prompt))
        
        # Validate inputs - ORIGINAL VALIDATION
        if not prompt or not isinstance(prompt, str) or len(prompt.strip()) == 0:
            raise ValueError("Prompt must be a non-empty string")
        
        if not model or not isinstance(model, str):
            raise ValueError("Model must be a non-empty string")
        
        if len(prompt) > 4000:
            raise ValueError(f"Prompt too long: {len(prompt)} characters (max 4000)")
        
        payload = {
            "model": model,
            "prompt": prompt.strip(),
            "n": 1,
            "size": "1024x1024",
            "quality": "standard",
            "response_format": "url"
        }
        
        try:
            response = await self._api_call("/images/generations", payload)
            
            if not response.get('data') or len(response['data']) == 0:
                raise ValueError("No image data returned from OpenRouter")
            
            image_data = response['data'][0]
            if not image_data.get('url'):
                raise ValueError("No image URL returned from OpenRouter")
            
            result = ChatCompletionResponse(
                type='image',
                response=image_data['url']
            )
            
            logger.info("Image generation successful", 
                       model=model,
                       image_url=result.response)
            
            return result
            
        except Exception as e:
            logger.error("Image generation failed", model=model, error=str(e))
            raise

    async def determine_function_calls(self, prompt: str, tools: List[Dict], model: str) -> List[Dict]:
        """Determine if function calls are needed based on the prompt - COMPLETE ORIGINAL"""
        logger.info("Function call determination requested", 
                   model=model,
                   prompt_length=len(prompt),
                   tools_count=len(tools))
        
        # Validate inputs - ORIGINAL VALIDATION
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")
        
        if not isinstance(tools, list):
            raise ValueError("Tools must be a list")
        
        if not model or not isinstance(model, str):
            raise ValueError("Model must be a non-empty string")
        
        # If no tools available, return empty list - ORIGINAL LOGIC
        if len(tools) == 0:
            logger.info("No tools available for function calling")
            return []
        
        # Create comprehensive system prompt for function determination - ORIGINAL LOGIC
        tools_description = json.dumps(tools, indent=2)
        
        sys_prompt = f"""You are a function call router. Analyze the user's request and determine if it requires calling any of these available tools:

{tools_description}

Rules:
1. If the user's request can be fulfilled using one or more of these tools, respond with a JSON array of function calls
2. Each function call should have the format: {{"name": "tool_name", "arguments": {{"param": "value"}}}}
3. If no tools are needed, respond with an empty array: []
4. ONLY respond with the JSON array, no other text
5. Ensure all required parameters are included in the arguments
6. Use appropriate parameter values based on the user's request

Example responses:
- []: for requests that don't need tools
- [{{"name": "web_search", "arguments": {{"query": "latest AI news"}}}}]: for search requests
- [{{"name": "save_to_file", "arguments": {{"filename": "report.txt", "content": "content here"}}}}]: for file operations"""

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self.chat_completion(messages, model)
            
            # Parse the JSON response - ORIGINAL PARSING LOGIC
            try:
                function_calls = json.loads(response.response.strip())
                
                # Validate that it's a list - ORIGINAL VALIDATION
                if not isinstance(function_calls, list):
                    logger.warning("Function call response not a list, returning empty list")
                    return []
                
                # Validate each function call - ORIGINAL VALIDATION
                valid_calls = []
                for call in function_calls:
                    if isinstance(call, dict) and 'name' in call and 'arguments' in call:
                        # Verify the tool name exists - ORIGINAL VALIDATION
                        tool_names = [tool['name'] for tool in tools]
                        if call['name'] in tool_names:
                            valid_calls.append(call)
                        else:
                            logger.warning("Unknown tool name in function call", 
                                         tool_name=call['name'],
                                         available_tools=tool_names)
                    else:
                        logger.warning("Invalid function call format", call=call)
                
                logger.info("Function call determination successful", 
                           function_calls_count=len(valid_calls))
                
                return valid_calls
                
            except json.JSONDecodeError as e:
                logger.warning("Failed to parse function call JSON response", 
                             response=response.response[:200],
                             error=str(e))
                return []
                
        except Exception as e:
            logger.error("Function call determination failed", 
                        model=model,
                        error=str(e))
            return []