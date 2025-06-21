# AI Portal -- Version 26.2.0 -- COMPLETE LEARNING MACHINE BUILD
# Build Time: 2025-06-21 - Complete Single File Implementation
# All functionality from v26.1.2 + Memory Learning System + Enhanced Error Handling

import asyncio
import os
import json
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from functools import wraps
from typing import Dict, List, Any, Optional, Callable, TypeVar
import uuid
from pathlib import Path
import pickle

from dotenv import load_dotenv
import aiohttp
import uvicorn
import yaml
import structlog
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import urllib.parse

from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleAuthRequest
from playwright.async_api import async_playwright

from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, Float, ForeignKey, JSON, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.sql import func
from sqlalchemy.exc import IntegrityError

# Memory Learning System imports
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Load environment variables ---
load_dotenv()

# --- Setup logging ---
structlog.configure(processors=[structlog.processors.JSONRenderer()])
logger = structlog.get_logger()
Base = declarative_base()

# --- Resilience Decorator ---
F = TypeVar('F', bound=Callable[..., Any])

def async_retry_with_backoff(retries: int = 4, initial_delay: float = 1.0, backoff: float = 2.0):
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            for i in range(retries):
                try:
                    return await func(*args, **kwargs)
                except aiohttp.ClientResponseError as e:
                    if e.status in [429, 500, 502, 503, 504] and i < retries - 1:
                        logger.warning("API call failed, retrying...", 
                                     attempt=i + 1, delay=delay, status=e.status)
                        await asyncio.sleep(delay)
                        delay *= backoff
                    else: 
                        raise
                except Exception: 
                    raise
        return wrapper
    return decorator

# --- DATABASE MODELS ---
class Project(Base):
    __tablename__ = 'projects'
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    user_id = Column(String(100), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    chat_history = relationship("ChatHistory", back_populates="project", cascade="all, delete-orphan")

class Persona(Base):
    __tablename__ = 'personas'
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    system_prompt = Column(Text, nullable=False)
    model_preference = Column(String(100))
    user_id = Column(String(100), nullable=False, index=True)
    __table_args__ = (UniqueConstraint('user_id', 'name', name='_user_id_name_uc'),)
    chat_history = relationship("ChatHistory", back_populates="persona")

class ChatHistory(Base):
    __tablename__ = 'chat_history'
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(PG_UUID(as_uuid=True), ForeignKey('projects.id', ondelete='CASCADE'), nullable=False)
    persona_id = Column(PG_UUID(as_uuid=True), ForeignKey('personas.id', ondelete='SET NULL'), nullable=True)
    user_id = Column(String(100), nullable=False)
    message = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    response_type = Column(String(50), default='text')
    model_used = Column(String(100), nullable=False)
    cost = Column(Float, default=0.0)
    response_time = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    project = relationship("Project", back_populates="chat_history")
    persona = relationship("Persona", back_populates="chat_history")

class ProjectSettings(Base):
    __tablename__ = 'project_settings'
    project_id = Column(PG_UUID(as_uuid=True), ForeignKey('projects.id', ondelete='CASCADE'), primary_key=True)
    active_persona_id = Column(PG_UUID(as_uuid=True), ForeignKey('personas.id', ondelete='SET NULL'), nullable=True)
    context_length = Column(Integer, default=10)
    settings = Column(JSON, default={})

# --- Pydantic Models ---
class ProjectRequest(BaseModel):
    name: str
    description: Optional[str] = None
    user_id: str = "anonymous"

class ProjectResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    user_id: str
    created_at: datetime
    updated_at: datetime

class PersonaRequest(BaseModel):
    name: str
    system_prompt: str
    model_preference: Optional[str] = None
    user_id: str = "anonymous"

class PersonaResponse(BaseModel):
    id: str
    name: str
    system_prompt: str
    model_preference: Optional[str]
    user_id: str

class ChatRequest(BaseModel):
    message: str
    user_id: str = "anonymous"
    project_id: Optional[str] = None
    persona_id: Optional[str] = None
    task_type: str = "auto"
    user_tier: str = "free"

class ChatCompletionResponse(BaseModel):
    type: str
    response: str

class ChatResponse(BaseModel):
    success: bool
    response: str
    response_type: str
    model: str
    reasoning: str
    project_id: str

class ObjectiveRequest(BaseModel):
    objective: str
    user_id: str = "anonymous"
    project_id: Optional[str] = None

# === MEMORY LEARNING SYSTEM ===
class MemoryService:
    """
    Persistent learning memory system for the AI Portal agent.
    Stores and retrieves task success patterns, failure corrections, and optimization insights.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory_dir = Path(config.get('memory_dir', './agent_memory'))
        self.embedding_model_name = config.get('embedding_model', 'all-MiniLM-L6-v2')
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
        self.max_memory_results = config.get('max_memory_results', 10)
        
        # Initialize directories
        self.memory_dir.mkdir(exist_ok=True)
        (self.memory_dir / 'plans').mkdir(exist_ok=True)
        (self.memory_dir / 'tasks').mkdir(exist_ok=True)
        (self.memory_dir / 'failures').mkdir(exist_ok=True)
        (self.memory_dir / 'insights').mkdir(exist_ok=True)
        (self.memory_dir / 'embeddings').mkdir(exist_ok=True)
        
        # Embedding model (will be loaded during initialization)
        self.embedding_model = None
        
        logger.info("Memory service configured", 
                   memory_dir=str(self.memory_dir),
                   embedding_model=self.embedding_model_name)

    async def initialize(self):
        """Initialize the memory system and load embedding model"""
        try:
            logger.info("Initializing memory system embedding model")
            
            # Load sentence transformer model in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            self.embedding_model = await loop.run_in_executor(
                None, 
                lambda: SentenceTransformer(self.embedding_model_name)
            )
            
            logger.info("Memory system initialized successfully", 
                       model=self.embedding_model_name)
            
            # Load existing memory statistics
            stats = await self.get_memory_stats()
            logger.info("Memory system ready", **stats)
            
        except Exception as e:
            logger.error("Failed to initialize memory system", error=str(e))
            raise ValueError(f"Memory system initialization failed: {str(e)}")

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        if not self.embedding_model:
            raise ValueError("Embedding model not initialized")
        
        try:
            embedding = self.embedding_model.encode([text])
            return embedding[0]
        except Exception as e:
            logger.error("Failed to generate embedding", text=text[:100], error=str(e))
            raise

    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            return float(similarity)
        except Exception as e:
            logger.error("Failed to calculate similarity", error=str(e))
            return 0.0

    def _save_memory_item(self, category: str, item_id: str, data: Dict[str, Any], embedding: np.ndarray):
        """Save memory item with embedding"""
        try:
            # Save data
            data_path = self.memory_dir / category / f"{item_id}.json"
            with open(data_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            # Save embedding
            embedding_path = self.memory_dir / 'embeddings' / f"{category}_{item_id}.npy"
            np.save(embedding_path, embedding)
            
            logger.debug("Memory item saved", category=category, item_id=item_id)
            
        except Exception as e:
            logger.error("Failed to save memory item", 
                        category=category, 
                        item_id=item_id, 
                        error=str(e))
            raise

    def _load_memory_items(self, category: str) -> List[Dict[str, Any]]:
        """Load all memory items from a category"""
        try:
            category_path = self.memory_dir / category
            items = []
            
            for json_file in category_path.glob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    item_id = json_file.stem
                    embedding_path = self.memory_dir / 'embeddings' / f"{category}_{item_id}.npy"
                    
                    if embedding_path.exists():
                        embedding = np.load(embedding_path)
                        data['_embedding'] = embedding
                        data['_item_id'] = item_id
                        items.append(data)
                
                except Exception as e:
                    logger.warning("Failed to load memory item", 
                                 file=str(json_file), 
                                 error=str(e))
            
            logger.debug("Memory items loaded", category=category, count=len(items))
            return items
            
        except Exception as e:
            logger.error("Failed to load memory items", category=category, error=str(e))
            return []

    async def store_successful_plan(self, objective: str, plan: List[Dict], 
                                  execution_results: Dict, user_id: str) -> str:
        """Store a successful plan execution for future reference"""
        try:
            plan_id = str(uuid.uuid4())
            timestamp = datetime.utcnow()
            
            # Create plan memory item
            plan_data = {
                "plan_id": plan_id,
                "objective": objective,
                "plan": plan,
                "execution_results": execution_results,
                "user_id": user_id,
                "success_rate": execution_results.get("success_rate", 0),
                "total_steps": execution_results.get("total_steps", 0),
                "successful_steps": execution_results.get("successful_steps", 0),
                "execution_time": execution_results.get("total_execution_time", 0),
                "timestamp": timestamp.isoformat(),
                "memory_type": "successful_plan"
            }
            
            # Generate embedding for the objective
            embedding = self._generate_embedding(objective)
            
            # Save to memory
            self._save_memory_item('plans', plan_id, plan_data, embedding)
            
            logger.info("Successful plan stored in memory", 
                       plan_id=plan_id,
                       objective=objective[:100],
                       success_rate=plan_data["success_rate"])
            
            return plan_id
            
        except Exception as e:
            logger.error("Failed to store successful plan", 
                        objective=objective[:100],
                        error=str(e))
            raise

    async def query_similar_plans(self, objective: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Query memory for similar successful plans"""
        try:
            # Generate embedding for query objective
            query_embedding = self._generate_embedding(objective)
            
            # Load all stored plans
            plans = self._load_memory_items('plans')
            
            # Calculate similarities
            similar_plans = []
            for plan in plans:
                if '_embedding' in plan:
                    similarity = self._calculate_similarity(query_embedding, plan['_embedding'])
                    if similarity >= self.similarity_threshold:
                        plan['similarity'] = similarity
                        similar_plans.append(plan)
            
            # Sort by similarity and limit results
            similar_plans.sort(key=lambda x: x['similarity'], reverse=True)
            similar_plans = similar_plans[:limit]
            
            # Clean up embeddings before returning
            for plan in similar_plans:
                plan.pop('_embedding', None)
                plan.pop('_item_id', None)
            
            logger.debug("Similar plans found", 
                        query=objective[:50],
                        found_count=len(similar_plans))
            
            return similar_plans
            
        except Exception as e:
            logger.error("Failed to query similar plans", 
                        objective=objective[:100],
                        error=str(e))
            return []

    async def store_task_success(self, task: Dict, result: str, persona_used: str) -> str:
        """Store successful task execution"""
        try:
            task_id = str(uuid.uuid4())
            timestamp = datetime.utcnow()
            
            task_data = {
                "task_id": task_id,
                "task": task,
                "result": result,
                "persona_used": persona_used,
                "timestamp": timestamp.isoformat(),
                "memory_type": "task_success"
            }
            
            # Generate embedding for task description
            task_description = task.get('description', str(task))
            embedding = self._generate_embedding(task_description)
            
            # Save to memory
            self._save_memory_item('tasks', task_id, task_data, embedding)
            
            logger.debug("Task success stored", 
                        task_id=task_id,
                        task_description=task_description[:50])
            
            return task_id
            
        except Exception as e:
            logger.error("Failed to store task success", 
                        task=str(task)[:100],
                        error=str(e))
            raise

    async def store_task_failure(self, task: Dict, failed_result: str, 
                               failure_reason: str, corrective_action: Dict) -> str:
        """Store task failure and its correction for learning"""
        try:
            failure_id = str(uuid.uuid4())
            timestamp = datetime.utcnow()
            
            failure_data = {
                "failure_id": failure_id,
                "original_task": task,
                "failed_result": failed_result,
                "failure_reason": failure_reason,
                "corrective_action": corrective_action,
                "timestamp": timestamp.isoformat(),
                "memory_type": "task_failure"
            }
            
            # Generate embedding for task description + failure reason
            task_description = task.get('description', str(task))
            search_text = f"{task_description} {failure_reason}"
            embedding = self._generate_embedding(search_text)
            
            # Save to memory
            self._save_memory_item('failures', failure_id, failure_data, embedding)
            
            logger.debug("Task failure stored", 
                        failure_id=failure_id,
                        task_description=task_description[:50],
                        failure_reason=failure_reason[:50])
            
            return failure_id
            
        except Exception as e:
            logger.error("Failed to store task failure", 
                        task=str(task)[:100],
                        error=str(e))
            raise

    async def query_similar_failures(self, task: Dict, failure_reason: str, 
                                   limit: int = 3) -> List[Dict[str, Any]]:
        """Query memory for similar past failures and their corrections"""
        try:
            # Generate embedding for current failure
            task_description = task.get('description', str(task))
            search_text = f"{task_description} {failure_reason}"
            query_embedding = self._generate_embedding(search_text)
            
            # Load all stored failures
            failures = self._load_memory_items('failures')
            
            # Calculate similarities
            similar_failures = []
            for failure in failures:
                if '_embedding' in failure:
                    similarity = self._calculate_similarity(query_embedding, failure['_embedding'])
                    if similarity >= self.similarity_threshold:
                        failure['similarity'] = similarity
                        similar_failures.append(failure)
            
            # Sort by similarity and limit results
            similar_failures.sort(key=lambda x: x['similarity'], reverse=True)
            similar_failures = similar_failures[:limit]
            
            # Clean up embeddings before returning
            for failure in similar_failures:
                failure.pop('_embedding', None)
                failure.pop('_item_id', None)
            
            logger.debug("Similar failures found", 
                        query=search_text[:50],
                        found_count=len(similar_failures))
            
            return similar_failures
            
        except Exception as e:
            logger.error("Failed to query similar failures", 
                        task=str(task)[:100],
                        error=str(e))
            return []

    async def store_insight(self, insight: str, context: Dict, user_id: str) -> str:
        """Store a learned insight"""
        try:
            insight_id = str(uuid.uuid4())
            timestamp = datetime.utcnow()
            
            insight_data = {
                "insight_id": insight_id,
                "insight": insight,
                "context": context,
                "user_id": user_id,
                "timestamp": timestamp.isoformat(),
                "memory_type": "insight"
            }
            
            # Generate embedding for insight
            embedding = self._generate_embedding(insight)
            
            # Save to memory
            self._save_memory_item('insights', insight_id, insight_data, embedding)
            
            logger.info("Insight stored in memory", 
                       insight_id=insight_id,
                       insight=insight[:100])
            
            return insight_id
            
        except Exception as e:
            logger.error("Failed to store insight", 
                        insight=insight[:100],
                        error=str(e))
            raise

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        try:
            stats = {
                "total_plans": len(list((self.memory_dir / 'plans').glob("*.json"))),
                "total_tasks": len(list((self.memory_dir / 'tasks').glob("*.json"))),
                "total_failures": len(list((self.memory_dir / 'failures').glob("*.json"))),
                "total_insights": len(list((self.memory_dir / 'insights').glob("*.json"))),
                "memory_dir": str(self.memory_dir),
                "embedding_model": self.embedding_model_name,
                "similarity_threshold": self.similarity_threshold
            }
            
            # Calculate memory size
            total_size = sum(f.stat().st_size for f in self.memory_dir.rglob('*') if f.is_file())
            stats["total_memory_size_mb"] = round(total_size / (1024 * 1024), 2)
            
            return stats
            
        except Exception as e:
            logger.error("Failed to get memory stats", error=str(e))
            return {"error": str(e)}

    async def clear_memory(self, memory_type: str = "all", confirm_phrase: str = "") -> Dict[str, Any]:
        """Clear memory (requires confirmation)"""
        if confirm_phrase != "CONFIRM_CLEAR_MEMORY":
            return {
                "error": "Confirmation phrase required",
                "required_phrase": "CONFIRM_CLEAR_MEMORY"
            }
        
        try:
            cleared_counts = {}
            
            if memory_type == "all" or memory_type == "plans":
                plans_cleared = len(list((self.memory_dir / 'plans').glob("*.json")))
                for f in (self.memory_dir / 'plans').glob("*.json"):
                    f.unlink()
                cleared_counts["plans"] = plans_cleared
            
            if memory_type == "all" or memory_type == "tasks":
                tasks_cleared = len(list((self.memory_dir / 'tasks').glob("*.json")))
                for f in (self.memory_dir / 'tasks').glob("*.json"):
                    f.unlink()
                cleared_counts["tasks"] = tasks_cleared
            
            if memory_type == "all" or memory_type == "failures":
                failures_cleared = len(list((self.memory_dir / 'failures').glob("*.json")))
                for f in (self.memory_dir / 'failures').glob("*.json"):
                    f.unlink()
                cleared_counts["failures"] = failures_cleared
            
            if memory_type == "all" or memory_type == "insights":
                insights_cleared = len(list((self.memory_dir / 'insights').glob("*.json")))
                for f in (self.memory_dir / 'insights').glob("*.json"):
                    f.unlink()
                cleared_counts["insights"] = insights_cleared
            
            # Clear embeddings
            if memory_type == "all":
                embeddings_cleared = len(list((self.memory_dir / 'embeddings').glob("*.npy")))
                for f in (self.memory_dir / 'embeddings').glob("*.npy"):
                    f.unlink()
                cleared_counts["embeddings"] = embeddings_cleared
            
            logger.info("Memory cleared", 
                       memory_type=memory_type,
                       cleared_counts=cleared_counts)
            
            return {
                "status": "memory_cleared",
                "memory_type": memory_type,
                "cleared_counts": cleared_counts,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to clear memory", 
                        memory_type=memory_type,
                        error=str(e))
            return {"error": f"Failed to clear memory: {str(e)}"}

# === SERVICE AND ROUTER CLASSES ===
class ConfigManager:
    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = config_file
        self.config = {}
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
        
        # Database configuration with validation
        db_password = os.environ.get("SUPABASE_PASSWORD")
        if not db_password:
            raise ValueError("SUPABASE_PASSWORD environment variable not found. Please set it in your .env file.")
        
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
        if ".." in filename or filename.startswith("/") or filename.startswith("\\"):
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
        """Validate content for security and size limits"""
        if not isinstance(content, str):
            logger.warning("Invalid content type", type=type(content))
            return False
        
        # Check content length
        if len(content) > self.max_content_length:
            logger.warning("Content too large", size=len(content), max_size=self.max_content_length)
            return False
        
        # Check for potentially malicious content patterns
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

class OpenSourceAIService:
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
        """Make API call to OpenRouter with comprehensive error handling"""
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
        """Generate chat completion using OpenRouter"""
        logger.info("Chat completion requested", model=model, message_count=len(messages))
        
        # Validate inputs
        if not messages or not isinstance(messages, list):
            raise ValueError("Messages must be a non-empty list")
        
        if not model or not isinstance(model, str):
            raise ValueError("Model must be a non-empty string")
        
        # Validate message format
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
        """Generate image using OpenRouter"""
        logger.info("Image generation requested", model=model, prompt_length=len(prompt))
        
        # Validate inputs
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
        """Determine if function calls are needed based on the prompt"""
        logger.info("Function call determination requested", 
                   model=model,
                   prompt_length=len(prompt),
                   tools_count=len(tools))
        
        # Validate inputs
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")
        
        if not isinstance(tools, list):
            raise ValueError("Tools must be a list")
        
        if not model or not isinstance(model, str):
            raise ValueError("Model must be a non-empty string")
        
        # If no tools available, return empty list
        if len(tools) == 0:
            logger.info("No tools available for function calling")
            return []
        
        # Create comprehensive system prompt for function determination
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
            
            # Parse the JSON response
            try:
                function_calls = json.loads(response.response.strip())
                
                # Validate that it's a list
                if not isinstance(function_calls, list):
                    logger.warning("Function call response not a list, returning empty list")
                    return []
                
                # Validate each function call
                valid_calls = []
                for call in function_calls:
                    if isinstance(call, dict) and 'name' in call and 'arguments' in call:
                        # Verify the tool name exists
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

class GoogleAIService:
    def __init__(self, config: Dict):
        self.credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        self.credentials = None
        
        # Load scopes from config or use defaults
        self.scopes = config.get('google_ai_scopes', [
            'https://www.googleapis.com/auth/cloud-platform',
            'https://www.googleapis.com/auth/generative-language'
        ])
        
        # Initialize credentials if available
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
        """Refresh the authentication token"""
        if self.credentials:
            try:
                self.credentials.refresh(GoogleAuthRequest())
                logger.debug("Google credentials refreshed successfully")
            except Exception as e:
                logger.error("Failed to refresh Google credentials", error=str(e))
                raise ValueError(f"Google authentication failed: {str(e)}")

    @async_retry_with_backoff()
    async def _api_call(self, url: str, payload: Dict):
        """Make API call to Google AI with comprehensive error handling"""
        if not self.credentials:
            raise ValueError("Google credentials not loaded. Please configure GOOGLE_APPLICATION_CREDENTIALS.")
        
        # Refresh token before making the call
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
        """Generate chat completion using Google Gemini"""
        logger.info("Google chat completion requested", model=model, message_count=len(messages))
        
        # Validate inputs
        if not messages or not isinstance(messages, list):
            raise ValueError("Messages must be a non-empty list")
        
        if not model or not isinstance(model, str):
            raise ValueError("Model must be a non-empty string")
        
        # Convert OpenAI format messages to Gemini format
        gemini_messages = []
        for message in messages:
            if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
                raise ValueError("Each message must have 'role' and 'content' fields")
            
            # Map roles (Gemini uses 'user' and 'model' instead of 'assistant')
            role = "user" if message["role"] in ["user", "system"] else "model"
            
            gemini_messages.append({
                "role": role,
                "parts": [{"text": message["content"]}]
            })
        
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
            
            # Validate response format
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

    def route(self, task_type: str, user_tier: str = "free", persona: Optional[Persona] = None):
        """Route task to appropriate service and model"""
        logger.debug("Routing request", 
                    task_type=task_type,
                    user_tier=user_tier,
                    persona_name=persona.name if persona else None)
        
        # Check persona preference first
        if persona and persona.model_preference:
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
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
    def __init__(self, config_file: str = "config.yaml"):
        logger.info("Initializing UnifiedAIPortal", config_file=config_file)
        
        # Load configuration
        self.config_manager = ConfigManager(config_file)
        self.config = self.config_manager.config
        
        # Initialize FastAPI application
        self.app = FastAPI(
            title="AI Portal - Learning Machine",
            version="26.2.0",
            description="Advanced AI orchestration system with persistent learning capabilities",
            lifespan=lifespan
        )
        
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
        
        # Initialize database
        db_url = self.config.get('database_url')
        if not db_url:
            raise ValueError("Database URL not configured. Please check your environment variables.")
        
        logger.info("Connecting to database", url_masked=db_url[:50] + "...")
        self.db_engine = create_engine(
            db_url,
            pool_size=10,
            max_overflow=20,
            pool_recycle=3600,
            echo=False  # Set to True for SQL debugging
        )
        self.DbSession = sessionmaker(bind=self.db_engine)
        
        # Create database tables
        try:
            Base.metadata.create_all(self.db_engine)
            logger.info("Database tables created/verified successfully")
        except Exception as e:
            logger.error("Database table creation failed", error=str(e))
            raise
        
        # Setup application
        self.setup_app()
        
        logger.info("UnifiedAIPortal initialization complete")

    async def initialize_memory_system(self):
        """Initialize the agent's persistent learning memory"""
        logger.info("Initializing memory system")
        try:
            self.memory_service = MemoryService(self.config)
            await self.memory_service.initialize()
            logger.info("AGENT LEARNING BRAIN ACTIVATED")
        except Exception as e:
            logger.error("Failed to initialize memory system", error=str(e))
            logger.warning("Continuing without persistent memory - system will be amnesiac")

    def setup_app(self):
        """Setup FastAPI application with middleware and routes"""
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
        
        # Setup routes
        self.setup_routes()
        
        logger.info("FastAPI application setup complete")

    def _get_or_create_default_project(self, session, user_id: str) -> uuid.UUID:
        """Get or create default project for user"""
        logger.debug("Getting or creating default project", user_id=user_id)
        
        try:
            project = session.query(Project).filter(
                Project.user_id == user_id,
                Project.name == "Default Project"
            ).first()
            
            if not project:
                logger.info("Creating default project for user", user_id=user_id)
                project = Project(
                    name="Default Project",
                    description="Automatically created default project",
                    user_id=user_id
                )
                session.add(project)
                session.commit()
                session.refresh(project)
                logger.info("Default project created", user_id=user_id, project_id=str(project.id))
            
            return project.id
            
        except Exception as e:
            logger.error("Failed to get or create default project", 
                        user_id=user_id,
                        error=str(e))
            session.rollback()
            raise

    async def execute_tool_call(self, tool_name: str, arguments: Dict) -> Any:
        """Execute a tool call with comprehensive error handling"""
        logger.info("Executing tool call", tool=tool_name, arguments=arguments)
        
        # Validate tool name
        tool_service = self.services.get('tools')
        if not tool_service:
            raise ValueError("Tool service not available")
        
        if not hasattr(tool_service, tool_name):
            available_tools = [method for method in dir(tool_service) 
                             if not method.startswith('_') and callable(getattr(tool_service, method))]
            raise ValueError(f"Tool '{tool_name}' not found. Available tools: {available_tools}")
        
        # Validate arguments
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a dictionary")
        
        try:
            # Execute the tool
            tool_method = getattr(tool_service, tool_name)
            result = await tool_method(**arguments)
            
            logger.info("Tool call executed successfully", 
                       tool=tool_name,
                       result_type=type(result).__name__)
            
            return result
            
        except TypeError as e:
            # Handle incorrect arguments
            logger.error("Tool call failed due to incorrect arguments", 
                        tool=tool_name,
                        arguments=arguments,
                        error=str(e))
            raise ValueError(f"Tool '{tool_name}' called with incorrect arguments: {str(e)}")
        except Exception as e:
            logger.error("Tool call execution failed", 
                        tool=tool_name,
                        error=str(e))
            raise ValueError(f"Tool '{tool_name}' execution failed: {str(e)}")

    def _save_chat_history(self, request: ChatRequest, response: ChatCompletionResponse, 
                          project_id: uuid.UUID, model_used: str, response_time: float):
        """Save chat history to database with comprehensive error handling"""
        logger.debug("Saving chat history", 
                    project_id=str(project_id),
                    model=model_used,
                    response_time=response_time)
        
        with self.DbSession() as session:
            try:
                # Convert persona_id to UUID if provided
                persona_uuid = None
                if request.persona_id:
                    try:
                        persona_uuid = uuid.UUID(request.persona_id)
                    except ValueError:
                        logger.warning("Invalid persona_id format", persona_id=request.persona_id)
                
                # Create chat history entry
                history = ChatHistory(
                    project_id=project_id,
                    persona_id=persona_uuid,
                    user_id=request.user_id,
                    message=request.message,
                    response=response.response,
                    response_type=response.type,
                    model_used=model_used,
                    cost=0.0,  # TODO: Implement cost calculation
                    response_time=response_time
                )
                
                session.add(history)
                session.commit()
                
                logger.debug("Chat history saved successfully", 
                           history_id=str(history.id))
                
            except Exception as e:
                logger.error("Failed to save chat history", 
                           project_id=str(project_id),
                           error=str(e))
                session.rollback()
                # Don't raise exception as this is background operation

    async def run_orchestration_plan(self, plan: List[Dict], user_id: str, project_id: uuid.UUID) -> Dict:
        """
        Execute the full Reflexive Swarm orchestration with Master Planner, 
        Persona Dispatcher, and Critic Agent - COMPLETE ORIGINAL IMPLEMENTATION WITH MEMORY ENHANCEMENT
        """
        logger.info("Starting orchestration plan execution", 
                   plan_steps=len(plan),
                   user_id=user_id,
                   project_id=str(project_id))
        
        results = []
        adaptive_corrections = 0
        max_corrections = 3
        start_time = time.time()
        
        with self.DbSession() as session:
            try:
                for step_index, step in enumerate(plan):
                    step_start_time = time.time()
                    logger.info(f"Executing step {step_index + 1}/{len(plan)}", step=step)
                    
                    # Persona Dispatcher - Select best persona for this task
                    personas = session.query(Persona).filter(Persona.user_id == user_id).all()
                    selected_persona = await self._dispatch_persona(step, personas)
                    
                    if selected_persona:
                        logger.info("Persona selected for step", 
                                  step=step_index + 1,
                                  persona=selected_persona.name)
                    else:
                        logger.info("No specific persona selected, using default execution", 
                                  step=step_index + 1)
                    
                    # Execute the task with selected persona
                    task_result = await self._execute_task_with_persona(step, selected_persona, user_id)
                    
                    # Critic Agent - Validate the result
                    critic_verdict = await self._critic_agent_validate(step, task_result)
                    
                    step_execution_time = time.time() - step_start_time
                    
                    if critic_verdict["status"] == "PASS":
                        logger.info(f"Step {step_index + 1} PASSED validation", 
                                  verdict=critic_verdict["reasoning"],
                                  execution_time=step_execution_time)
                        
                        # Store successful task in memory
                        if self.memory_service:
                            try:
                                await self.memory_service.store_task_success(
                                    task=step,
                                    result=task_result,
                                    persona_used=selected_persona.name if selected_persona else "default"
                                )
                                logger.debug("Task success stored in memory", step=step_index + 1)
                            except Exception as e:
                                logger.error("Failed to store task success in memory", 
                                           step=step_index + 1,
                                           error=str(e))
                        
                        results.append({
                            "step": step_index + 1,
                            "task": step,
                            "result": task_result,
                            "persona": selected_persona.name if selected_persona else "Default",
                            "status": "SUCCESS",
                            "critic_verdict": critic_verdict,
                            "execution_time": step_execution_time
                        })
                        
                    else:
                        logger.warning(f"Step {step_index + 1} FAILED validation", 
                                     verdict=critic_verdict["reasoning"],
                                     execution_time=step_execution_time)
                        
                        # Adaptive Execution Loop with Memory-Enhanced Correction
                        if adaptive_corrections < max_corrections:
                            logger.info("Attempting adaptive correction", 
                                      step=step_index + 1,
                                      correction_attempt=adaptive_corrections + 1)
                            
                            # Query memory for similar past failures
                            similar_failures = []
                            if self.memory_service:
                                try:
                                    similar_failures = await self.memory_service.query_similar_failures(
                                        task=step,
                                        failure_reason=critic_verdict["reasoning"],
                                        limit=3
                                    )
                                    if similar_failures:
                                        logger.info("Found similar failures in memory", 
                                                  count=len(similar_failures),
                                                  step=step_index + 1)
                                except Exception as e:
                                    logger.error("Failed to query similar failures", 
                                               step=step_index + 1,
                                               error=str(e))
                            
                            # Generate memory-informed corrective task
                            corrective_task = await self._generate_memory_informed_corrective_task(
                                step, task_result, critic_verdict["reasoning"], similar_failures
                            )
                            
                            logger.info("Generated corrective task", 
                                      step=step_index + 1,
                                      corrective_task=corrective_task.get("description", "")[:100])
                            
                            # Execute corrective task
                            corrective_start_time = time.time()
                            corrective_result = await self._execute_task_with_persona(
                                corrective_task, selected_persona, user_id
                            )
                            
                            # Re-validate
                            corrective_verdict = await self._critic_agent_validate(
                                corrective_task, corrective_result
                            )
                            
                            corrective_execution_time = time.time() - corrective_start_time
                            total_step_time = time.time() - step_start_time
                            
                            # Store failure and correction in memory
                            if self.memory_service:
                                try:
                                    await self.memory_service.store_task_failure(
                                        task=step,
                                        failed_result=task_result,
                                        failure_reason=critic_verdict["reasoning"],
                                        corrective_action=corrective_task
                                    )
                                    logger.debug("Task failure and correction stored in memory", 
                                               step=step_index + 1)
                                except Exception as e:
                                    logger.error("Failed to store task failure in memory", 
                                               step=step_index + 1,
                                               error=str(e))
                            
                            correction_status = "CORRECTED" if corrective_verdict["status"] == "PASS" else "FAILED"
                            
                            results.append({
                                "step": step_index + 1,
                                "original_task": step,
                                "corrective_task": corrective_task,
                                "result": corrective_result,
                                "persona": selected_persona.name if selected_persona else "Default",
                                "status": correction_status,
                                "critic_verdict": corrective_verdict,
                                "adaptive_correction": True,
                                "memory_informed": len(similar_failures) > 0,
                                "execution_time": total_step_time,
                                "correction_time": corrective_execution_time
                            })
                            
                            adaptive_corrections += 1
                            
                            if corrective_verdict["status"] == "PASS":
                                logger.info(f"Step {step_index + 1} CORRECTED successfully", 
                                          correction_attempt=adaptive_corrections)
                            else:
                                logger.warning(f"Step {step_index + 1} correction FAILED", 
                                             correction_attempt=adaptive_corrections)
                        else:
                            # Max corrections reached, log failure and continue
                            logger.error(f"Step {step_index + 1} FAILED after max corrections", 
                                       max_corrections=max_corrections)
                            
                            results.append({
                                "step": step_index + 1,
                                "task": step,
                                "result": task_result,
                                "persona": selected_persona.name if selected_persona else "Default",
                                "status": "FAILED_MAX_CORRECTIONS",
                                "critic_verdict": critic_verdict,
                                "execution_time": step_execution_time
                            })
                
                # Calculate final statistics
                total_execution_time = time.time() - start_time
                successful_steps = len([r for r in results if r["status"] in ["SUCCESS", "CORRECTED"]])
                failed_steps = len([r for r in results if "FAILED" in r["status"]])
                
                orchestration_result = {
                    "status": "completed",
                    "total_steps": len(plan),
                    "successful_steps": successful_steps,
                    "failed_steps": failed_steps,
                    "adaptive_corrections_used": adaptive_corrections,
                    "total_execution_time": total_execution_time,
                    "success_rate": successful_steps / len(plan) if len(plan) > 0 else 0,
                    "results": results
                }
                
                logger.info("Orchestration plan execution completed", 
                           total_steps=len(plan),
                           successful_steps=successful_steps,
                           failed_steps=failed_steps,
                           total_time=total_execution_time,
                           success_rate=orchestration_result["success_rate"])
                
                return orchestration_result
                
            except Exception as e:
                logger.error("Orchestration plan execution failed", 
                           error=str(e),
                           completed_steps=len(results))
                raise

    async def _dispatch_persona(self, task: Dict, personas: List[Persona]) -> Optional[Persona]:
        """
        Persona Dispatcher - Analyze task and select most appropriate specialist persona
        """
        logger.debug("Dispatching persona for task", 
                    task_description=task.get('description', '')[:100],
                    available_personas=len(personas))
        
        if not personas:
            logger.warning("No personas available for dispatch")
            return None
        
        # Get routing information for dispatcher
        try:
            dispatcher_route = self.router.route('simple_qa', 'free')
            dispatcher_service = self.services.get(dispatcher_route['service'])
            
            if not dispatcher_service:
                logger.error("Dispatcher service not available", service=dispatcher_route['service'])
                return None
            
        except Exception as e:
            logger.error("Failed to route dispatcher request", error=str(e))
            return None
        
        # Prepare persona descriptions
        persona_descriptions = []
        for p in personas:
            # Truncate system prompt for better readability
            prompt_summary = p.system_prompt[:100] + "..." if len(p.system_prompt) > 100 else p.system_prompt
            persona_descriptions.append(f"- {p.name}: {prompt_summary}")
        
        # Create dispatch prompt
        task_description = task.get('description', task.get('task', str(task)))
        
        dispatch_prompt = f"""
        You are a Persona Dispatcher. Analyze this task and select the most appropriate specialist:
        
        Task: {task_description}
        
        Available Personas:
        {chr(10).join(persona_descriptions)}
        
        Respond with ONLY the exact persona name that best matches this task.
        If no persona is specifically suitable, respond with "NONE".
        """
        
        messages = [{"role": "user", "content": dispatch_prompt}]
        
        try:
            completion = await dispatcher_service.chat_completion(messages, dispatcher_route['model'])
            selected_name = completion.response.strip()
            
            logger.debug("Persona dispatcher response", selected_name=selected_name)
            
            # Handle explicit NONE response
            if selected_name.upper() == "NONE":
                logger.info("Persona dispatcher returned NONE - no suitable specialist found")
                return None
            
            # Find matching persona
            for persona in personas:
                if persona.name.lower() == selected_name.lower():
                    logger.info("Persona dispatched successfully", 
                              task_description=task_description[:50],
                              persona=persona.name)
                    return persona
            
            # No exact match found - log warning and return None
            available_names = [p.name for p in personas]
            logger.warning("Persona dispatcher returned unmatched name", 
                          selected_name=selected_name, 
                          available_personas=available_names)
            return None
            
        except Exception as e:
            logger.error("Persona dispatch failed", error=str(e))
            return None

    async def _execute_task_with_persona(self, task: Dict, persona: Optional[Persona], user_id: str) -> str:
        """
        Execute a task using the selected persona or default execution if no persona available
        """
        task_description = task.get('description', task.get('task', str(task)))
        task_type = task.get('type', 'simple_qa')
        
        logger.debug("Executing task with persona", 
                    task_description=task_description[:100],
                    task_type=task_type,
                    persona=persona.name if persona else "default")
        
        try:
            # Check if task requires tools
            function_router_service = self.services['openrouter']
            fc_route = self.router.route('function_routing', 'free')
            
            tool_calls = await function_router_service.determine_function_calls(
                task_description,
                self.config.get('available_tools', []),
                fc_route['model']
            )
            
            if tool_calls:
                logger.info("Task requires tool execution", 
                          task_description=task_description[:50],
                          tool_calls_count=len(tool_calls))
                
                # Execute tool calls
                tool_results = await asyncio.gather(*[
                    self.execute_tool_call(call['name'], call['arguments'])
                    for call in tool_calls
                ], return_exceptions=True)
                
                # Handle any tool execution errors
                successful_results = []
                for i, result in enumerate(tool_results):
                    if isinstance(result, Exception):
                        logger.error("Tool call failed", 
                                   tool_call=tool_calls[i],
                                   error=str(result))
                        successful_results.append({"error": str(result), "tool": tool_calls[i]['name']})
                    else:
                        successful_results.append(result)
                
                # Use persona to interpret tool results if available
                if persona:
                    interpretation_prompt = f"""
                    Task: {task_description}
                    
                    Tool Results: {json.dumps(successful_results, indent=2, default=str)}
                    
                    Based on your expertise, provide a comprehensive response to the task using these results.
                    If any tools failed, work around the failures and provide the best possible response.
                    """
                    
                    route_decision = self.router.route(task_type, 'free', persona)
                    service = self.services.get(route_decision['service'])
                    
                    messages = [
                        {"role": "system", "content": persona.system_prompt},
                        {"role": "user", "content": interpretation_prompt}
                    ]
                    
                    completion = await service.chat_completion(messages, route_decision['model'])
                    return completion.response
                else:
                    # No persona available - provide direct tool results with basic interpretation
                    logger.info("No persona available for task interpretation - using default execution")
                    route_decision = self.router.route(task_type, 'free')
                    service = self.services.get(route_decision['service'])
                    
                    interpretation_prompt = f"""
                    Task: {task_description}
                    
                    Tool Results: {json.dumps(successful_results, indent=2, default=str)}
                    
                    Provide a comprehensive response to the task using these tool results.
                    If any tools failed, work around the failures and provide the best possible response.
                    """
                    
                    messages = [{"role": "user", "content": interpretation_prompt}]
                    completion = await service.chat_completion(messages, route_decision['model'])
                    return completion.response
            else:
                # Standard AI response with or without persona
                logger.debug("Task does not require tools, using direct AI completion")
                
                if persona:
                    route_decision = self.router.route(task_type, 'free', persona)
                    service = self.services.get(route_decision['service'])
                    
                    messages = [
                        {"role": "system", "content": persona.system_prompt},
                        {"role": "user", "content": task_description}
                    ]
                else:
                    # No persona available - use default execution
                    logger.debug("No persona available for task - using default execution")
                    route_decision = self.router.route(task_type, 'free')
                    service = self.services.get(route_decision['service'])
                    
                    messages = [{"role": "user", "content": task_description}]
                
                completion = await service.chat_completion(messages, route_decision['model'])
                return completion.response
                
        except Exception as e:
            logger.error("Task execution failed", 
                        task_description=task_description[:50],
                        error=str(e))
            raise ValueError(f"Task execution failed: {str(e)}")

    async def _critic_agent_validate(self, task: Dict, result: str) -> Dict:
        """
        Critic Agent - Validate task result against objective with comprehensive analysis
        """
        task_description = task.get('description', task.get('task', str(task)))
        task_objective = task.get('objective', 'Complete the task successfully')
        
        logger.debug("Critic agent validating result", 
                    task_description=task_description[:100],
                    result_length=len(result))
        
        try:
            critic_route = self.router.route('simple_qa', 'free')
            critic_service = self.services.get(critic_route['service'])
            
            validation_prompt = f"""
            You are a Critic Agent. Your job is to evaluate if a task result meets its objective.
            
            TASK: {task_description}
            OBJECTIVE: {task_objective}
            
            RESULT TO EVALUATE:
            {result}
            
            Analyze if the result successfully accomplishes the task objective.
            Consider:
            1. Does the result directly address the task requirements?
            2. Is the result complete and comprehensive?
            3. Is the quality sufficient for the objective?
            4. Are there any obvious errors or omissions?
            
            Respond with EXACTLY this format:
            STATUS: [PASS or FAIL]
            REASONING: [Detailed explanation of why it passes or fails]
            
            Be strict but fair in your evaluation. A result should PASS only if it genuinely meets the objective.
            """
            
            messages = [{"role": "user", "content": validation_prompt}]
            
            completion = await critic_service.chat_completion(messages, critic_route['model'])
            response = completion.response
            
            # Parse the response
            status = "FAIL"  # Default to fail for safety
            reasoning = "Could not parse critic response"
            
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('STATUS:'):
                    status_text = line.split('STATUS:')[1].strip().upper()
                    if status_text in ['PASS', 'FAIL']:
                        status = status_text
                elif line.startswith('REASONING:'):
                    reasoning = line.split('REASONING:')[1].strip()
            
            # Validate that we got a proper response
            if status not in ['PASS', 'FAIL']:
                logger.warning("Invalid critic status, defaulting to FAIL", 
                             parsed_status=status,
                             response=response[:200])
                status = 'FAIL'
                reasoning = "Critic agent provided invalid status format"
            
            result_dict = {
                "status": status,
                "reasoning": reasoning,
                "full_response": response
            }
            
            logger.debug("Critic agent validation completed", 
                        status=status,
                        reasoning=reasoning[:100])
            
            return result_dict
            
        except Exception as e:
            logger.error("Critic agent validation failed", 
                        task_description=task_description[:50],
                        error=str(e))
            return {
                "status": "FAIL",
                "reasoning": f"Critic agent error: {str(e)}",
                "full_response": ""
            }

    async def _generate_memory_informed_corrective_task(self, original_task: Dict, failed_result: str, 
                                                      failure_reason: str, similar_failures: List[Dict]) -> Dict:
        """
        Generate a corrective task informed by past similar failures and their successful corrections
        """
        logger.debug("Generating memory-informed corrective task", 
                    original_task=original_task.get('description', '')[:50],
                    similar_failures_count=len(similar_failures))
        
        try:
            corrective_route = self.router.route('simple_qa', 'free')
            corrective_service = self.services.get(corrective_route['service'])
            
            original_description = original_task.get('description', original_task.get('task', str(original_task)))
            
            # Build memory context from similar failures
            memory_context = ""
            if similar_failures:
                memory_context = "\n\nLEARNED FROM PAST FAILURES:\n"
                for i, failure in enumerate(similar_failures[:2]):  # Use top 2 similar failures
                    memory_context += f"Past Failure {i+1}:\n"
                    memory_context += f"- Task: {failure.get('original_task', {}).get('description', '')}\n"
                    memory_context += f"- Failure Reason: {failure.get('failure_reason', '')}\n"
                    memory_context += f"- Successful Correction: {failure.get('corrective_action', {}).get('description', '')}\n\n"
                memory_context += "Apply these lessons to create a better corrective task.\n"
            
            corrective_prompt = f"""
            You are a Memory-Enhanced Adaptive Task Generator with access to past failure corrections.
            A task has failed validation and needs correction using learned experience.
            
            ORIGINAL TASK: {original_description}
            FAILED RESULT: {failed_result}
            FAILURE REASON: {failure_reason}
            {memory_context}
            
            Generate a corrective task that:
            1. Addresses the specific failure reason
            2. Learns from past similar failures
            3. Is more specific and targeted than the original
            4. Applies proven correction strategies
            5. Has clear, measurable success criteria
            
            Respond with EXACTLY this format:
            CORRECTIVE_TASK: [Description of the corrective task]
            OBJECTIVE: [Clear success criteria]
            TYPE: [task type: simple_qa, code_generation, etc.]
            MEMORY_APPLIED: [true/false - whether past failures informed this correction]
            """
            
            messages = [{"role": "user", "content": corrective_prompt}]
            completion = await corrective_service.chat_completion(messages, corrective_route['model'])
            response = completion.response
            
            # Parse the response
            corrective_task = original_description  # Fallback
            objective = "Complete the corrective task"
            task_type = "simple_qa"
            memory_applied = len(similar_failures) > 0
            
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('CORRECTIVE_TASK:'):
                    corrective_task = line.split('CORRECTIVE_TASK:')[1].strip()
                elif line.startswith('OBJECTIVE:'):
                    objective = line.split('OBJECTIVE:')[1].strip()
                elif line.startswith('TYPE:'):
                    task_type = line.split('TYPE:')[1].strip()
                elif line.startswith('MEMORY_APPLIED:'):
                    memory_applied = line.split('MEMORY_APPLIED:')[1].strip().lower() == 'true'
            
            result_dict = {
                "description": corrective_task,
                "objective": objective,
                "type": task_type,
                "is_corrective": True,
                "memory_informed": memory_applied,
                "original_task": original_task,
                "similar_failures_count": len(similar_failures)
            }
            
            logger.debug("Memory-informed corrective task generated", 
                        corrective_task=corrective_task[:100],
                        memory_applied=memory_applied)
            
            return result_dict
            
        except Exception as e:
            logger.error("Memory-informed corrective task generation failed", error=str(e))
            # Fallback to standard corrective task generation
            return await self._generate_corrective_task(original_task, failed_result, failure_reason)

    async def _generate_corrective_task(self, original_task: Dict, failed_result: str, failure_reason: str) -> Dict:
        """
        Generate a corrective task when the Critic Agent reports failure - ORIGINAL FALLBACK METHOD
        """
        logger.debug("Generating standard corrective task", 
                    original_task=original_task.get('description', '')[:50])
        
        try:
            corrective_route = self.router.route('simple_qa', 'free')
            corrective_service = self.services.get(corrective_route['service'])
            
            original_description = original_task.get('description', original_task.get('task', str(original_task)))
            
            corrective_prompt = f"""
            You are an Adaptive Task Generator. A task has failed validation and needs correction.
            
            ORIGINAL TASK: {original_description}
            FAILED RESULT: {failed_result}
            FAILURE REASON: {failure_reason}
            
            Generate a corrective task that addresses the specific failure reason.
            The corrective task should be more specific and targeted than the original.
            Include clear success criteria to avoid the same failure.
            
            Respond with EXACTLY this format:
            CORRECTIVE_TASK: [Description of the corrective task]
            OBJECTIVE: [Clear success criteria]
            TYPE: [task type: simple_qa, code_generation, etc.]
            """
            
            messages = [{"role": "user", "content": corrective_prompt}]
            completion = await corrective_service.chat_completion(messages, corrective_route['model'])
            response = completion.response
            
            # Parse the response
            corrective_task = original_description  # Fallback
            objective = "Complete the corrective task"
            task_type = "simple_qa"
            
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('CORRECTIVE_TASK:'):
                    corrective_task = line.split('CORRECTIVE_TASK:')[1].strip()
                elif line.startswith('OBJECTIVE:'):
                    objective = line.split('OBJECTIVE:')[1].strip()
                elif line.startswith('TYPE:'):
                    task_type = line.split('TYPE:')[1].strip()
            
            result_dict = {
                "description": corrective_task,
                "objective": objective,
                "type": task_type,
                "is_corrective": True,
                "original_task": original_task
            }
            
            logger.debug("Standard corrective task generated", 
                        corrective_task=corrective_task[:100])
            
            return result_dict
            
        except Exception as e:
            logger.error("Corrective task generation failed", error=str(e))
            return {
                "description": f"Retry: {original_description}",
                "objective": "Complete the task successfully", 
                "type": "simple_qa",
                "is_corrective": True,
                "original_task": original_task
            }

    def _save_orchestration_history(self, request: ObjectiveRequest, results: Dict, 
                                   project_id: uuid.UUID, execution_time: float):
        """
        Save orchestration execution history for analysis and improvement
        """
        logger.debug("Saving orchestration history", 
                    project_id=str(project_id),
                    execution_time=execution_time)
        
        with self.DbSession() as session:
            try:
                # Save as a special chat history entry
                history = ChatHistory(
                    project_id=project_id,
                    persona_id=None,
                    user_id=request.user_id,
                    message=f"ORCHESTRATION: {request.objective}",
                    response=json.dumps(results, indent=2, default=str),
                    response_type='orchestration',
                    model_used='memory_enhanced_reflexive_swarm_v26.2.0',
                    response_time=execution_time
                )
                session.add(history)
                session.commit()
                logger.info("Orchestration history saved", 
                           project_id=str(project_id),
                           execution_time=execution_time)
            except Exception as e:
                logger.error("Failed to save orchestration history", 
                           project_id=str(project_id),
                           error=str(e))
                session.rollback()

    def setup_routes(self):
        """Setup all API routes with comprehensive error handling"""
        logger.info("Setting up API routes")

        @self.app.post("/projects", response_model=ProjectResponse)
        async def create_project(request: ProjectRequest):
            """Create a new project"""
            logger.info("Creating new project", name=request.name, user_id=request.user_id)
            
            with self.DbSession() as session:
                try:
                    project = Project(
                        name=request.name,
                        description=request.description,
                        user_id=request.user_id
                    )
                    session.add(project)
                    session.commit()
                    session.refresh(project)
                    
                    logger.info("Project created successfully", 
                               project_id=str(project.id),
                               name=project.name)
                    
                    return ProjectResponse(
                        id=str(project.id),
                        name=project.name,
                        description=project.description,
                        user_id=project.user_id,
                        created_at=project.created_at,
                        updated_at=project.updated_at
                    )
                except Exception as e:
                    session.rollback()
                    logger.error("Failed to create project", 
                               name=request.name,
                               error=str(e))
                    raise HTTPException(status_code=500, detail="Failed to create project")

        @self.app.get("/projects", response_model=List[ProjectResponse])
        async def list_projects(user_id: str):
            """List all projects for a user"""
            logger.debug("Listing projects", user_id=user_id)
            
            with self.DbSession() as session:
                try:
                    projects = session.query(Project).filter(Project.user_id == user_id).all()
                    
                    result = [
                        ProjectResponse(
                            id=str(p.id),
                            name=p.name,
                            description=p.description,
                            user_id=p.user_id,
                            created_at=p.created_at,
                            updated_at=p.updated_at
                        ) for p in projects
                    ]
                    
                    logger.debug("Projects listed successfully", 
                               user_id=user_id,
                               count=len(result))
                    
                    return result
                except Exception as e:
                    logger.error("Failed to list projects", 
                               user_id=user_id,
                               error=str(e))
                    raise HTTPException(status_code=500, detail="Failed to list projects")

        @self.app.post("/personas", status_code=201)
        async def create_persona(request: PersonaRequest):
            """Create a new persona"""
            logger.info("Creating new persona", 
                       name=request.name,
                       user_id=request.user_id)
            
            with self.DbSession() as session:
                try:
                    persona = Persona(
                        name=request.name,
                        system_prompt=request.system_prompt,
                        model_preference=request.model_preference,
                        user_id=request.user_id
                    )
                    session.add(persona)
                    session.commit()
                    session.refresh(persona)
                    
                    logger.info("Persona created successfully", 
                               persona_id=str(persona.id),
                               name=persona.name)
                    
                    return {"id": str(persona.id), "name": persona.name}
                    
                except IntegrityError:
                    session.rollback()
                    logger.warning("Persona name already exists", 
                                 name=request.name,
                                 user_id=request.user_id)
                    raise HTTPException(
                        status_code=409,
                        detail=f"Persona with name '{request.name}' already exists for this user."
                    )
                except Exception as e:
                    session.rollback()
                    logger.error("Failed to create persona", 
                               name=request.name,
                               error=str(e))
                    raise HTTPException(status_code=500, detail="Failed to create persona")

        @self.app.get("/personas", response_model=List[PersonaResponse])
        async def list_personas(user_id: str):
            """List all personas for a user"""
            logger.debug("Listing personas", user_id=user_id)
            
            with self.DbSession() as session:
                try:
                    personas = session.query(Persona).filter(Persona.user_id == user_id).all()
                    
                    result = [
                        PersonaResponse(
                            id=str(p.id),
                            name=p.name,
                            system_prompt=p.system_prompt,
                            model_preference=p.model_preference,
                            user_id=p.user_id
                        ) for p in personas
                    ]
                    
                    logger.debug("Personas listed successfully", 
                               user_id=user_id,
                               count=len(result))
                    
                    return result
                except Exception as e:
                    logger.error("Failed to list personas", 
                               user_id=user_id,
                               error=str(e))
                    raise HTTPException(status_code=500, detail="Failed to list personas")

        @self.app.post("/chat", response_model=ChatResponse)
        async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
            """Main chat endpoint with tool calling and persona support"""
            start_time = time.time()
            logger.info("Chat request received", 
                       user_id=request.user_id,
                       message_length=len(request.message),
                       task_type=request.task_type)
            
            with self.DbSession() as session:
                try:
                    # Get persona if specified
                    persona = None
                    if request.persona_id:
                        try:
                            persona = session.query(Persona).filter(
                                Persona.id == uuid.UUID(request.persona_id)
                            ).first()
                            if persona:
                                logger.debug("Persona loaded for chat", persona=persona.name)
                        except ValueError:
                            logger.warning("Invalid persona_id format", persona_id=request.persona_id)
                    
                    # Get or create project
                    project_id = (
                        uuid.UUID(request.project_id) if request.project_id 
                        else self._get_or_create_default_project(session, request.user_id)
                    )
                    
                    # --- Tool/Function Calling Logic ---
                    function_router_service = self.services['openrouter']
                    fc_route = self.router.route('function_routing', request.user_tier)
                    
                    tool_calls = await function_router_service.determine_function_calls(
                        request.message,
                        self.config.get('available_tools', []),
                        fc_route['model']
                    )
                    
                    if tool_calls:
                        logger.info("Executing tool calls", count=len(tool_calls))
                        # Execute tool calls
                        results = await asyncio.gather(*[
                            self.execute_tool_call(call['name'], call['arguments'])
                            for call in tool_calls
                        ], return_exceptions=True)
                        
                        # Handle tool execution results
                        processed_results = []
                        for i, result in enumerate(results):
                            if isinstance(result, Exception):
                                logger.error("Tool call failed", 
                                           tool=tool_calls[i]['name'],
                                           error=str(result))
                                processed_results.append({
                                    "tool": tool_calls[i]['name'],
                                    "error": str(result)
                                })
                            else:
                                processed_results.append(result)
                        
                        final_response_str = json.dumps(processed_results, indent=2, default=str)
                        final_response_type = "tool_response"
                        route_decision = fc_route
                    else:
                        # --- Standard Chat Logic ---
                        task_type = request.task_type
                        if task_type == "auto":
                            # Auto-detect task type based on message content
                            message_lower = request.message.lower()
                            if any(k in message_lower for k in ["image", "photo", "picture", "generate image"]):
                                task_type = "image_generation"
                            elif any(k in message_lower for k in ["code", "script", "program", "function", "class"]):
                                task_type = "code_generation"
                            else:
                                task_type = "simple_qa"
                            
                            logger.debug("Auto-detected task type", 
                                       original_type=request.task_type,
                                       detected_type=task_type)
                        
                        route_decision = self.router.route(task_type, request.user_tier, persona)
                        service = self.services.get(route_decision['service'])
                        
                        if task_type == "image_generation":
                            completion = await service.image_generation(request.message, route_decision['model'])
                        else:
                            messages = []
                            if persona:
                                messages.append({"role": "system", "content": persona.system_prompt})
                            messages.append({"role": "user", "content": request.message})
                            
                            completion = await service.chat_completion(messages, route_decision['model'])
                        
                        final_response_str = completion.response
                        final_response_type = completion.type
                    
                    response_time = time.time() - start_time
                    
                    # Save chat history in background
                    mock_completion = ChatCompletionResponse(
                        type=final_response_type,
                        response=final_response_str
                    )
                    
                    background_tasks.add_task(
                        self._save_chat_history,
                        request,
                        mock_completion,
                        project_id,
                        route_decision['model'],
                        response_time
                    )
                    
                    logger.info("Chat request completed successfully", 
                               user_id=request.user_id,
                               response_time=response_time,
                               response_type=final_response_type)
                    
                    return ChatResponse(
                        success=True,
                        response=final_response_str,
                        response_type=final_response_type,
                        model=route_decision['model'],
                        reasoning=route_decision.get('reasoning', ''),
                        project_id=str(project_id)
                    )
                    
                except Exception as e:
                    response_time = time.time() - start_time
                    logger.error("Chat request failed", 
                               user_id=request.user_id,
                               error=str(e),
                               response_time=response_time)
                    raise HTTPException(status_code=500, detail=f"Chat request failed: {str(e)}")

        @self.app.post("/objectives/execute")
        async def execute_objective(request: ObjectiveRequest, background_tasks: BackgroundTasks):
            """
            Full Memory-Enhanced Agentic Orchestration Engine - The Learning Reflexive Swarm
            Master Planner -> Persona Dispatcher -> Critic Agent -> Adaptive Execution Loop
            ORIGINAL ORCHESTRATION PRESERVED AS PRIMARY SYSTEM + Memory Learning
            """
            start_time = time.time()
            logger.info("Memory-enhanced objective orchestration started", 
                       objective=request.objective,
                       user_id=request.user_id)
            
            with self.DbSession() as session:
                try:
                    project_id = (
                        uuid.UUID(request.project_id) if request.project_id 
                        else self._get_or_create_default_project(session, request.user_id)
                    )
                    
                    # === MEMORY-ENHANCED MASTER PLANNER AGENT ===
                    logger.info("Activating Memory-Enhanced Master Planner Agent")
                    planner_route = self.router.route('code_generation', 'free')
                    planner_service = self.services.get(planner_route['service'])
                    
                    # STEP 1: Query memory for similar past plans
                    similar_plans = []
                    if self.memory_service:
                        try:
                            similar_plans = await self.memory_service.query_similar_plans(request.objective, limit=3)
                            if similar_plans:
                                logger.info("Found similar plans in memory", count=len(similar_plans))
                        except Exception as e:
                            logger.error("Failed to query memory for similar plans", error=str(e))
                    
                    # STEP 2: Create memory-informed planning prompt
                    memory_context = ""
                    if similar_plans:
                        memory_context = "\n\nRELEVANT PAST EXPERIENCE:\n"
                        for i, plan in enumerate(similar_plans[:2]):  # Use top 2 similar plans
                            memory_context += f"Past Objective: {plan.get('objective', '')}\n"
                            memory_context += f"Success Rate: {plan.get('success_rate', 0):.2f}\n"
                            memory_context += f"Key Steps: {json.dumps(plan.get('plan', [])[:3])}\n\n"  # First 3 steps
                        memory_context += "Use this experience to create a better plan.\n"
                    
                    planning_prompt = f"""
                    You are the Memory-Enhanced Master Planner Agent with access to past successful operations.
                    Your job is to break down complex objectives into executable steps using learned experience.
                    
                    OBJECTIVE: {request.objective}
                    {memory_context}
                    
                    Create a detailed execution plan that improves on past experience. Each step should be:
                    1. Specific and actionable
                    2. Have clear success criteria
                    3. Be executable by AI agents with available tools
                    4. Learn from past successes and failures
                    
                    Available tools: web_search, browse_website, save_to_file, generateAdCopy
                    
                    Respond with EXACTLY this JSON format:
                    {{
                        "plan": [
                            {{
                                "step": 1,
                                "description": "Detailed task description",
                                "objective": "Specific success criteria", 
                                "type": "simple_qa|code_generation|image_generation",
                                "estimated_difficulty": "low|medium|high",
                                "dependencies": [],
                                "learned_from_memory": true/false
                            }}
                        ],
                        "total_steps": 5,
                        "estimated_duration": "15 minutes",
                        "success_criteria": "Overall objective completion criteria",
                        "memory_informed": {len(similar_plans) > 0}
                    }}
                    
                    Make the plan comprehensive, achievable, and informed by past experience.
                    """
                    
                    planner_messages = [{"role": "user", "content": planning_prompt}]
                    planner_completion = await planner_service.chat_completion(planner_messages, planner_route['model'])
                    
                    # Parse the plan
                    try:
                        plan_data = json.loads(planner_completion.response)
                        execution_plan = plan_data.get("plan", [])
                        logger.info("Memory-Enhanced Master Planner created execution plan", 
                                  total_steps=len(execution_plan),
                                  memory_informed=plan_data.get("memory_informed", False))
                    except json.JSONDecodeError:
                        logger.error("Failed to parse Master Planner response", 
                                   response=planner_completion.response[:500])
                        
                        # Fallback plan
                        execution_plan = [{
                            "step": 1,
                            "description": request.objective,
                            "objective": "Complete the requested objective",
                            "type": "simple_qa",
                            "estimated_difficulty": "medium",
                            "dependencies": []
                        }]
                    
                    # === EXECUTE THE MEMORY-ENHANCED ORCHESTRATION PLAN ===
                    logger.info("Starting Memory-Enhanced Reflexive Swarm execution")
                    orchestration_results = await self.run_orchestration_plan(
                        execution_plan, request.user_id, project_id
                    )
                    
                    execution_time = time.time() - start_time
                    
                    # === STORE SUCCESSFUL PLAN IN MEMORY ===
                    if (self.memory_service and 
                        orchestration_results.get("successful_steps", 0) > 0):
                        try:
                            await self.memory_service.store_successful_plan(
                                objective=request.objective,
                                plan=execution_plan,
                                execution_results=orchestration_results,
                                user_id=request.user_id
                            )
                            logger.info("Plan stored in agent memory for future learning")
                        except Exception as e:
                            logger.error("Failed to store plan in memory", error=str(e))
                    
                    # === FINAL SYNTHESIS ===
                    # Use Master Planner to synthesize final results
                    synthesis_prompt = f"""
                    You are the Memory-Enhanced Master Planner Agent completing an orchestration cycle.
                    
                    ORIGINAL OBJECTIVE: {request.objective}
                    
                    EXECUTION RESULTS:
                    {json.dumps(orchestration_results, indent=2, default=str)}
                    
                    Provide a comprehensive summary of:
                    1. What was accomplished
                    2. Key deliverables created
                    3. Any remaining tasks or recommendations
                    4. Overall success assessment
                    5. Learning insights for future similar objectives
                    
                    Be detailed and actionable in your summary.
                    """
                    
                    synthesis_messages = [{"role": "user", "content": synthesis_prompt}]
                    synthesis_completion = await planner_service.chat_completion(
                        synthesis_messages, planner_route['model']
                    )
                    
                    # Save orchestration history
                    background_tasks.add_task(
                        self._save_orchestration_history,
                        request, orchestration_results, project_id, execution_time
                    )
                    
                    logger.info("Memory-enhanced objective orchestration completed", 
                               objective=request.objective,
                               execution_time=execution_time,
                               success_rate=orchestration_results.get("success_rate", 0))
                    
                    return {
                        "status": "orchestration_complete",
                        "objective": request.objective,
                        "project_id": str(project_id),
                        "execution_plan": execution_plan,
                        "orchestration_results": orchestration_results,
                        "final_synthesis": synthesis_completion.response,
                        "performance_metrics": {
                            "total_execution_time": execution_time,
                            "total_steps": orchestration_results["total_steps"],
                            "successful_steps": orchestration_results["successful_steps"],
                            "failed_steps": orchestration_results["failed_steps"],
                            "adaptive_corrections": orchestration_results["adaptive_corrections_used"],
                            "success_rate": orchestration_results.get("success_rate", 0),
                            "memory_informed": len(similar_plans) > 0
                        },
                        "system_info": {
                            "master_planner_model": planner_route['model'],
                            "orchestration_engine": "Memory-Enhanced Reflexive Swarm v26.2.0",
                            "agents_used": ["Memory-Enhanced Master Planner", "Persona Dispatcher", "Critic Agent", "Learning Memory System"],
                            "learning_active": self.memory_service is not None
                        }
                    }
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    logger.error("Orchestration failed", 
                               objective=request.objective,
                               error=str(e),
                               execution_time=execution_time)
                    return {
                        "status": "orchestration_failed",
                        "objective": request.objective,
                        "error": str(e),
                        "project_id": str(project_id),
                        "execution_time": execution_time,
                        "fallback_message": "The orchestration system encountered an error. Please try with a simpler objective or check system configuration."
                    }

        @self.app.get("/memory/stats")
        async def get_memory_stats():
            """Get agent memory statistics"""
            if not self.memory_service:
                return {
                    "error": "Memory system not initialized",
                    "suggestion": "Memory system requires proper configuration and dependencies"
                }
            
            try:
                stats = await self.memory_service.get_memory_stats()
                return {
                    "status": "memory_active",
                    "agent_learning": True,
                    "memory_stats": stats,
                    "capabilities": [
                        "Plan Learning",
                        "Task Success Memory", 
                        "Failure Pattern Recognition",
                        "Corrective Action Learning",
                        "Insight Storage"
                    ]
                }
            except Exception as e:
                logger.error("Failed to get memory stats", error=str(e))
                return {"error": f"Memory stats unavailable: {str(e)}"}
        
        @self.app.post("/memory/insights")
        async def store_insight(insight_data: dict):
            """Store a learned insight in agent memory"""
            if not self.memory_service:
                raise HTTPException(status_code=503, detail="Memory system not initialized")
            
            insight = insight_data.get("insight", "")
            context = insight_data.get("context", {})
            user_id = insight_data.get("user_id", "anonymous")
            
            if not insight or len(insight.strip()) == 0:
                raise HTTPException(status_code=400, detail="Insight text required")
            
            try:
                memory_id = await self.memory_service.store_insight(insight, context, user_id)
                
                return {
                    "status": "insight_stored",
                    "memory_id": memory_id,
                    "insight": insight,
                    "learning_enabled": True
                }
            except Exception as e:
                logger.error("Failed to store insight", error=str(e))
                raise HTTPException(status_code=500, detail=f"Failed to store insight: {str(e)}")
        
        @self.app.post("/memory/clear")
        async def clear_memory(clear_request: dict):
            """Clear agent memory (requires confirmation)"""
            if not self.memory_service:
                raise HTTPException(status_code=503, detail="Memory system not initialized")
            
            memory_type = clear_request.get("memory_type")
            confirm_phrase = clear_request.get("confirm_phrase", "")
            
            try:
                result = await self.memory_service.clear_memory(memory_type, confirm_phrase)
                return result
            except Exception as e:
                logger.error("Failed to clear memory", error=str(e))
                raise HTTPException(status_code=500, detail=f"Failed to clear memory: {str(e)}")

        @self.app.get("/health")
        async def health_check():
            """Basic health check endpoint"""
            return {
                "status": "healthy",
                "version": "26.2.0",
                "timestamp": datetime.utcnow().isoformat(),
                "system": "AI Portal Learning Machine"
            }
        
        @self.app.get("/projects/{project_id}/history")
        async def get_project_history(project_id: str, user_id: str, limit: int = 50):
            """Get chat history for a specific project"""
            logger.debug("Getting project history", project_id=project_id, user_id=user_id)
            
            with self.DbSession() as session:
                try:
                    project_uuid = uuid.UUID(project_id)
                    history = session.query(ChatHistory).filter(
                        ChatHistory.project_id == project_uuid,
                        ChatHistory.user_id == user_id
                    ).order_by(ChatHistory.created_at.desc()).limit(limit).all()
                    
                    result = [
                        {
                            "id": str(h.id),
                            "message": h.message,
                            "response": h.response,
                            "response_type": h.response_type,
                            "model_used": h.model_used,
                            "response_time": h.response_time,
                            "created_at": h.created_at.isoformat(),
                            "persona_id": str(h.persona_id) if h.persona_id else None
                        } for h in history
                    ]
                    
                    logger.debug("Project history retrieved", 
                               project_id=project_id,
                               count=len(result))
                    
                    return result
                except ValueError:
                    raise HTTPException(status_code=400, detail="Invalid project ID format")
                except Exception as e:
                    logger.error("Failed to get project history", 
                               project_id=project_id,
                               error=str(e))
                    raise HTTPException(status_code=500, detail="Failed to retrieve project history")
        
        @self.app.delete("/projects/{project_id}")
        async def delete_project(project_id: str, user_id: str):
            """Delete a project and all associated data"""
            logger.info("Deleting project", project_id=project_id, user_id=user_id)
            
            with self.DbSession() as session:
                try:
                    project_uuid = uuid.UUID(project_id)
                    project = session.query(Project).filter(
                        Project.id == project_uuid,
                        Project.user_id == user_id
                    ).first()
                    
                    if not project:
                        raise HTTPException(status_code=404, detail="Project not found")
                    
                    project_name = project.name
                    session.delete(project)
                    session.commit()
                    
                    logger.info("Project deleted successfully", 
                               project_id=project_id,
                               name=project_name)
                    
                    return {"status": "deleted", "project_id": project_id, "name": project_name}
                    
                except ValueError:
                    raise HTTPException(status_code=400, detail="Invalid project ID format")
                except HTTPException:
                    raise
                except Exception as e:
                    session.rollback()
                    logger.error("Failed to delete project", 
                               project_id=project_id,
                               error=str(e))
                    raise HTTPException(status_code=500, detail="Failed to delete project")
        
        @self.app.delete("/personas/{persona_id}")
        async def delete_persona(persona_id: str, user_id: str):
            """Delete a persona"""
            logger.info("Deleting persona", persona_id=persona_id, user_id=user_id)
            
            with self.DbSession() as session:
                try:
                    persona_uuid = uuid.UUID(persona_id)
                    persona = session.query(Persona).filter(
                        Persona.id == persona_uuid,
                        Persona.user_id == user_id
                    ).first()
                    
                    if not persona:
                        raise HTTPException(status_code=404, detail="Persona not found")
                    
                    persona_name = persona.name
                    session.delete(persona)
                    session.commit()
                    
                    logger.info("Persona deleted successfully", 
                               persona_id=persona_id,
                               name=persona_name)
                    
                    return {"status": "deleted", "persona_id": persona_id, "name": persona_name}
                    
                except ValueError:
                    raise HTTPException(status_code=400, detail="Invalid persona ID format")
                except HTTPException:
                    raise
                except Exception as e:
                    session.rollback()
                    logger.error("Failed to delete persona", 
                               persona_id=persona_id,
                               error=str(e))
                    raise HTTPException(status_code=500, detail="Failed to delete persona")
            
        @self.app.get("/system/status")
        async def system_status():
            """Get comprehensive system status and configuration"""
            logger.debug("System status check requested")
            
            with self.DbSession() as session:
                try:
                    # Get database statistics
                    project_count = session.query(Project).count()
                    persona_count = session.query(Persona).count()
                    chat_count = session.query(ChatHistory).count()
                    
                    # Get orchestration statistics
                    orchestration_count = session.query(ChatHistory).filter(
                        ChatHistory.response_type == 'orchestration'
                    ).count()
                    
                    # Check service availability
                    services_status = {
                        "openrouter": bool(self.config.get("openrouter_api_key")),
                        "google": bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")),
                        "serper": bool(self.config.get("serper_api_key")),
                        "copyshark": bool(self.config.get("copyshark_api_token"))
                    }
                    
                    # Memory system status
                    memory_status = {
                        "initialized": self.memory_service is not None,
                        "learning_active": self.memory_service is not None
                    }
                    
                    if self.memory_service:
                        try:
                            memory_stats = await self.memory_service.get_memory_stats()
                            memory_status.update(memory_stats)
                        except Exception as e:
                            logger.error("Failed to get memory stats for status", error=str(e))
                            memory_status["error"] = str(e)
                    
                    status_response = {
                        "status": "operational",
                        "version": "26.2.0",
                        "system": "AI Portal Learning Machine",
                        "timestamp": datetime.utcnow().isoformat(),
                        "database": {
                            "connected": True,
                            "projects": project_count,
                            "personas": persona_count,
                            "chat_history_entries": chat_count,
                            "orchestration_executions": orchestration_count
                        },
                        "services": services_status,
                        "memory_system": memory_status,
                        "orchestration_agents": [
                            "Memory-Enhanced Master Planner",
                            "Persona Dispatcher", 
                            "Critic Agent",
                            "Adaptive Execution Loop",
                            "Learning Memory System"
                        ],
                        "available_tools": [tool["name"] for tool in self.config.get("available_tools", [])],
                        "model_tiers": list(self.config.get("model_tiers", {}).keys()),
                        "features": [
                            "Multi-agent orchestration",
                            "Persona-based AI routing",
                            "Tool integration",
                            "Persistent learning memory",
                            "Adaptive error correction",
                            "Real-time web search",
                            "Website content extraction",
                            "File workspace management",
                            "Ad copy generation"
                        ],
                        "performance": {
                            "total_projects": project_count,
                            "total_personas": persona_count,
                            "total_interactions": chat_count,
                            "learning_enabled": self.memory_service is not None
                        }
                    }
                    
                    logger.debug("System status check completed successfully")
                    return status_response
                    
                except Exception as e:
                    logger.error("System status check failed", error=str(e))
                    return {
                        "status": "degraded",
                        "error": str(e),
                        "version": "26.2.0",
                        "timestamp": datetime.utcnow().isoformat(),
                        "system": "AI Portal Learning Machine"
                    }
        
        logger.info("API routes setup completed successfully")

    def run(self):
        """Run the AI Portal application"""
        logger.info("Starting AI Portal Learning Machine", version="26.2.0")
        
        # Pre-flight checks
        logger.info("Performing pre-flight system checks")
        
        # Check critical environment variables
        required_vars = ["SUPABASE_PASSWORD"]
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        
        if missing_vars:
            logger.error("Missing required environment variables", missing=missing_vars)
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        # Check database connection
        try:
            with self.DbSession() as session:
                session.execute("SELECT 1")
            logger.info("Database connection verified")
        except Exception as e:
            logger.error("Database connection failed", error=str(e))
            raise ValueError(f"Database connection failed: {str(e)}")
        
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
        
        # Start the server
        logger.info("Launching FastAPI server", host="0.0.0.0", port=8000)
        
        try:
            uvicorn.run(
                self.app, 
                host="0.0.0.0", 
                port=8000,
                log_level="info",
                access_log=True,
                reload=False  # Set to True for development
            )
        except Exception as e:
            logger.error("Failed to start server", error=str(e))
            raise

if __name__ == "__main__":
    try:
        # Initialize and run the portal
        portal = UnifiedAIPortal()
        portal.run()
    except KeyboardInterrupt:
        logger.info("AI Portal shutdown requested by user")
    except Exception as e:
        logger.error("AI Portal startup failed", error=str(e))
        print(f"FATAL ERROR: {str(e)}")
        exit(1)