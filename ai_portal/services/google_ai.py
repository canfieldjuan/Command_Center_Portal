"""
Google AI service integration for Gemini models - COMPLETE VERSION
ALL original Google AI functionality from main.py preserved
"""

import os
import asyncio
import aiohttp
import json
import structlog
from typing import Dict, List

from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleAuthRequest

from ..core.decorators import async_retry_with_backoff
from ..schemas.chat import ChatCompletionResponse

logger = structlog.get_logger()

class GoogleAIService:
    """Google AI service for Gemini model access - COMPLETE ORIGINAL"""
    
    def __init__(self, config: Dict):
        self.credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        self.credentials = None
        
        # Load scopes from config or use defaults - ORIGINAL LOGIC
        self.scopes = config.get('google_ai_scopes', [
            'https://www.googleapis.com/auth/cloud-platform',
            'https://www.googleapis.com/auth/generative-language'
        ])
        
        # Initialize credentials if available - ORIGINAL LOGIC
        if self.credentials_path and os.path.exists(self.credentials_path):
            try:
                self.credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path,
                    scopes=self.scopes
                )
                logger.info("Google AI Service initialized with service account", 
                           credentials_path=self.credentials_path,
                           scopes=len(self.scopes))
            except Exception as e:
                logger.error("Failed to load Google credentials", 
                           path=self.credentials_path,
                           error=str(e))
        else:
            logger.warning("Google AI Service credentials not available", 
                         path=self.credentials_path)
        
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.timeout = 180  # 3 minutes timeout

    def _refresh_token(self):
        """Refresh the authentication token - ORIGINAL FUNCTIONALITY"""
        if self.credentials:
            try:
                self.credentials.refresh(GoogleAuthRequest())
                logger.debug("Google credentials refreshed successfully")
            except Exception as e:
                logger.error("Failed to refresh Google credentials", error=str(e))
                raise ValueError(f"Google authentication failed: {str(e)}")

    @async_retry_with_backoff()
    async def _api_call(self, url: str, payload: Dict):
        """Make API call to Google AI with comprehensive error handling - COMPLETE ORIGINAL"""
        if not self.credentials:
            raise ValueError("Google credentials not loaded. Please configure GOOGLE_APPLICATION_CREDENTIALS.")
        
        # Refresh token before making the call - ORIGINAL LOGIC
        self._refresh_token()
        
        headers = {
            "Authorization": f"Bearer {self.credentials.token}",
            "Content-Type": "application/json",
            "User-Agent": "AI-Portal/26.2.0"
        }
        
        logger.debug("Google AI API call", url=url, payload_size=len(json.dumps(payload)))
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    result = await response.json()
                    
                    logger.debug("Google AI API call successful", 
                                url=url,
                                status=response.status)
                    
                    return result
                    
        except aiohttp.ClientResponseError as e:
            error_detail = "Unknown error"
            try:
                error_body = await e.response.json()
                error_detail = error_body.get('error', {}).get('message', str(e))
            except:
                error_detail = str(e)
            
            logger.error("Google AI API error", 
                        url=url,
                        status=e.status,
                        error=error_detail)
            
            if e.status == 401:
                raise ValueError("Google AI API authentication failed. Check your credentials.")
            elif e.status == 429:
                raise ValueError("Google AI API rate limit exceeded. Please retry later.")
            elif e.status == 403:
                raise ValueError("Google AI API access forbidden. Check your permissions.")
            else:
                raise ValueError(f"Google AI API error: {error_detail}")
                
        except asyncio.TimeoutError:
            logger.error("Google AI API timeout", url=url)
            raise ValueError(f"Google AI API call timed out after {self.timeout} seconds")
        except Exception as e:
            logger.error("Google AI API unexpected error", url=url, error=str(e))
            raise ValueError(f"Google AI API call failed: {str(e)}")

    async def chat_completion(self, messages: List[Dict], model: str) -> ChatCompletionResponse:
        """Generate chat completion using Google Gemini - COMPLETE ORIGINAL"""
        logger.info("Google chat completion requested", model=model, message_count=len(messages))
        
        # Validate inputs - ORIGINAL VALIDATION
        if not messages or not isinstance(messages, list):
            raise ValueError("Messages must be a non-empty list")
        
        if not model or not isinstance(model, str):
            raise ValueError("Model must be a non-empty string")
        
        # Convert OpenAI format messages to Gemini format - ORIGINAL CONVERSION LOGIC
        gemini_messages = []
        for message in messages:
            if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
                raise ValueError("Each message must have 'role' and 'content' fields")
            
            # Map roles (Gemini uses 'user' and 'model' instead of 'assistant') - ORIGINAL LOGIC
            role = "user" if message["role"] in ["user", "system"] else "model"
            
            gemini_messages.append({
                "role": role,
                "parts": [{"text": message["content"]}]
            })
        
        # Complete original payload configuration
        payload = {
            "contents": gemini_messages,
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.9,
                "maxOutputTokens": 4096,
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }
        
        url = f"{self.base_url}/{model}:generateContent"
        
        try:
            response = await self._api_call(url, payload)
            
            # Validate response format - ORIGINAL VALIDATION
            if not response.get('candidates') or len(response['candidates']) == 0:
                raise ValueError("No response candidates returned from Google AI")
            
            candidate = response['candidates'][0]
            if not candidate.get('content') or not candidate['content'].get('parts'):
                raise ValueError("Invalid response format from Google AI")
            
            parts = candidate['content']['parts']
            if len(parts) == 0 or not parts[0].get('text'):
                raise ValueError("No text content in response from Google AI")
            
            result = ChatCompletionResponse(
                type='text',
                response=parts[0]['text']
            )
            
            logger.info("Google chat completion successful", 
                       model=model,
                       response_length=len(result.response))
            
            return result
            
        except Exception as e:
            logger.error("Google chat completion failed", model=model, error=str(e))
            raise