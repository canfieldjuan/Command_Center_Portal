"""
Memory learning system for persistent AI learning and improvement - COMPLETE VERSION
ALL original memory functionality from main.py preserved (the sophisticated learning brain!)
"""

import asyncio
import json
import uuid
import numpy as np
import structlog
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = structlog.get_logger()

class MemoryService:
    """
    Persistent learning memory system for the AI Portal agent.
    Stores and retrieves task success patterns, failure corrections, and optimization insights.
    COMPLETE ORIGINAL IMPLEMENTATION - THE LEARNING BRAIN!
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory_dir = Path(config.get('memory_dir', './agent_memory'))
        self.embedding_model_name = config.get('embedding_model', 'all-MiniLM-L6-v2')
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
        self.max_memory_results = config.get('max_memory_results', 10)
        
        # Initialize directories - ORIGINAL STRUCTURE
        self.memory_dir.mkdir(exist_ok=True)
        (self.memory_dir / 'plans').mkdir(exist_ok=True)
        (self.memory_dir / 'tasks').mkdir(exist_ok=True)
        (self.memory_dir / 'failures').mkdir(exist_ok=True)
        (self.memory_dir / 'insights').mkdir(exist_ok=True)
        (self.memory_dir / 'embeddings').mkdir(exist_ok=True)
        
        # Embedding model (will be loaded during initialization) - ORIGINAL LOGIC
        self.embedding_model = None
        
        logger.info("Memory service configured", 
                   memory_dir=str(self.memory_dir),
                   embedding_model=self.embedding_model_name)

    async def initialize(self):
        """Initialize the memory system and load embedding model - COMPLETE ORIGINAL"""
        try:
            logger.info("Initializing memory system embedding model")
            
            # Load sentence transformer model in a separate thread to avoid blocking - ORIGINAL LOGIC
            loop = asyncio.get_event_loop()
            self.embedding_model = await loop.run_in_executor(
                None, 
                lambda: SentenceTransformer(self.embedding_model_name)
            )
            
            logger.info("Memory system initialized successfully", 
                       model=self.embedding_model_name)
            
            # Load existing memory statistics - ORIGINAL FUNCTIONALITY
            stats = await self.get_memory_stats()
            logger.info("Memory system ready", **stats)
            
        except Exception as e:
            logger.error("Failed to initialize memory system", error=str(e))
            raise ValueError(f"Memory system initialization failed: {str(e)}")

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text - ORIGINAL FUNCTIONALITY"""
        if not self.embedding_model:
            raise ValueError("Embedding model not initialized")
        
        try:
            embedding = self.embedding_model.encode([text])
            return embedding[0]
        except Exception as e:
            logger.error("Failed to generate embedding", text=text[:100], error=str(e))
            raise

    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings - ORIGINAL FUNCTIONALITY"""
        try:
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            return float(similarity)
        except Exception as e:
            logger.error("Failed to calculate similarity", error=str(e))
            return 0.0

    def _save_memory_item(self, category: str, item_id: str, data: Dict[str, Any], embedding: np.ndarray):
        """Save memory item with embedding - ORIGINAL FUNCTIONALITY"""
        try:
            # Save data - ORIGINAL LOGIC
            data_path = self.memory_dir / category / f"{item_id}.json"
            with open(data_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            # Save embedding - ORIGINAL LOGIC
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
        """Load all memory items from a category - ORIGINAL FUNCTIONALITY"""
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
        """Store a successful plan execution for future reference - COMPLETE ORIGINAL"""
        try:
            plan_id = str(uuid.uuid4())
            timestamp = datetime.utcnow()
            
            # Create plan memory item - ORIGINAL DATA STRUCTURE
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
            
            # Generate embedding for the objective - ORIGINAL LOGIC
            embedding = self._generate_embedding(objective)
            
            # Save to memory - ORIGINAL LOGIC
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
        """Query memory for similar successful plans - COMPLETE ORIGINAL"""
        try:
            # Generate embedding for query objective - ORIGINAL LOGIC
            query_embedding = self._generate_embedding(objective)
            
            # Load all stored plans - ORIGINAL LOGIC
            plans = self._load_memory_items('plans')
            
            # Calculate similarities - ORIGINAL ALGORITHM
            similar_plans = []
            for plan in plans:
                if '_embedding' in plan:
                    similarity = self._calculate_similarity(query_embedding, plan['_embedding'])
                    if similarity >= self.similarity_threshold:
                        plan['similarity'] = similarity
                        similar_plans.append(plan)
            
            # Sort by similarity and limit results - ORIGINAL LOGIC
            similar_plans.sort(key=lambda x: x['similarity'], reverse=True)
            similar_plans = similar_plans[:limit]
            
            # Clean up embeddings before returning - ORIGINAL LOGIC
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
        """Store successful task execution - COMPLETE ORIGINAL"""
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
            
            # Generate embedding for task description - ORIGINAL LOGIC
            task_description = task.get('description', str(task))
            embedding = self._generate_embedding(task_description)
            
            # Save to memory - ORIGINAL LOGIC
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
        """Store task failure and its correction for learning - COMPLETE ORIGINAL"""
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
            
            # Generate embedding for task description + failure reason - ORIGINAL LOGIC
            task_description = task.get('description', str(task))
            search_text = f"{task_description} {failure_reason}"
            embedding = self._generate_embedding(search_text)
            
            # Save to memory - ORIGINAL LOGIC
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
        """Query memory for similar past failures and their corrections - COMPLETE ORIGINAL"""
        try:
            # Generate embedding for current failure - ORIGINAL LOGIC
            task_description = task.get('description', str(task))
            search_text = f"{task_description} {failure_reason}"
            query_embedding = self._generate_embedding(search_text)
            
            # Load all stored failures - ORIGINAL LOGIC
            failures = self._load_memory_items('failures')
            
            # Calculate similarities - ORIGINAL ALGORITHM
            similar_failures = []
            for failure in failures:
                if '_embedding' in failure:
                    similarity = self._calculate_similarity(query_embedding, failure['_embedding'])
                    if similarity >= self.similarity_threshold:
                        failure['similarity'] = similarity
                        similar_failures.append(failure)
            
            # Sort by similarity and limit results - ORIGINAL LOGIC
            similar_failures.sort(key=lambda x: x['similarity'], reverse=True)
            similar_failures = similar_failures[:limit]
            
            # Clean up embeddings before returning - ORIGINAL LOGIC
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
        """Store a learned insight - COMPLETE ORIGINAL"""
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
            
            # Generate embedding for insight - ORIGINAL LOGIC
            embedding = self._generate_embedding(insight)
            
            # Save to memory - ORIGINAL LOGIC
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
        """Get memory system statistics - COMPLETE ORIGINAL"""
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
            
            # Calculate memory size - ORIGINAL FUNCTIONALITY
            total_size = sum(f.stat().st_size for f in self.memory_dir.rglob('*') if f.is_file())
            stats["total_memory_size_mb"] = round(total_size / (1024 * 1024), 2)
            
            return stats
            
        except Exception as e:
            logger.error("Failed to get memory stats", error=str(e))
            return {"error": str(e)}

    async def clear_memory(self, memory_type: str = "all", confirm_phrase: str = "") -> Dict[str, Any]:
        """Clear memory (requires confirmation) - COMPLETE ORIGINAL"""
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
            
            # Clear embeddings - ORIGINAL LOGIC
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