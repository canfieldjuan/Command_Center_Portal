"""
Chat conversation API endpoints - COMPLETE VERSION
ALL original chat functionality from main.py preserved
"""

import time
import uuid
import json
import asyncio
import structlog
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from sqlalchemy.orm import Session

from ..schemas.chat import ChatRequest, ChatResponse, ChatCompletionResponse
from ..models.project import Project
from ..models.persona import Persona
from ..models.chat_history import ChatHistory
from ..models.base import get_db_session

logger = structlog.get_logger()
router = APIRouter(prefix="/chat", tags=["chat"])

# These will be injected by the main application
_services = None
_router = None
_config = None

def inject_dependencies(services, intelligent_router, config):
    """Inject service dependencies - called by main app"""
    global _services, _router, _config
    _services = services
    _router = intelligent_router
    _config = config

def get_db():
    """Dependency to get database session"""
    with get_db_session() as session:
        yield session

async def execute_tool_call(tool_name: str, arguments: dict):
    """Execute a tool call with comprehensive error handling - COMPLETE ORIGINAL"""
    logger.info("Executing tool call", tool=tool_name, arguments=arguments)
    
    # Validate tool name
    tool_service = _services.get('tools')
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

def save_chat_history_bg(request: ChatRequest, response: ChatCompletionResponse, 
                        project_id: uuid.UUID, model_used: str, response_time: float):
    """Save chat history to database - background task - COMPLETE ORIGINAL"""
    logger.debug("Saving chat history", 
                project_id=str(project_id),
                model=model_used,
                response_time=response_time)
    
    try:
        with get_db_session() as session:
            ChatHistory.create_entry(
                session=session,
                project_id=project_id,
                user_id=request.user_id,
                message=request.message,
                response=response.response,
                response_type=response.type,
                model_used=model_used,
                response_time=response_time,
                persona_id=request.persona_id,
                cost=0.0  # TODO: Implement cost calculation
            )
            
    except Exception as e:
        logger.error("Failed to save chat history", 
                   project_id=str(project_id),
                   error=str(e))
        # Don't raise exception as this is background operation

@router.post("", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks, 
                       db: Session = Depends(get_db)):
    """Main chat endpoint with tool calling and persona support - COMPLETE ORIGINAL"""
    start_time = time.time()
    logger.info("Chat request received", 
               user_id=request.user_id,
               message_length=len(request.message),
               task_type=request.task_type)
    
    if not _services or not _router or not _config:
        raise HTTPException(status_code=500, detail="Service dependencies not initialized")
    
    try:
        # Get persona if specified
        persona = None
        if request.persona_id:
            try:
                persona = db.query(Persona).filter(
                    Persona.id == uuid.UUID(request.persona_id)
                ).first()
                if persona:
                    logger.debug("Persona loaded for chat", persona=persona.name)
            except ValueError:
                logger.warning("Invalid persona_id format", persona_id=request.persona_id)
        
        # Get or create project
        project_id = (
            uuid.UUID(request.project_id) if request.project_id 
            else Project.get_or_create_default(db, request.user_id)
        )
        
        # --- Tool/Function Calling Logic - COMPLETE ORIGINAL ---
        function_router_service = _services['openrouter']
        fc_route = _router.route('function_routing', request.user_tier)
        
        tool_calls = await function_router_service.determine_function_calls(
            request.message,
            _config.get('available_tools', []),
            fc_route['model']
        )
        
        if tool_calls:
            logger.info("Executing tool calls", count=len(tool_calls))
            # Execute tool calls
            results = await asyncio.gather(*[
                execute_tool_call(call['name'], call['arguments'])
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
            # --- Standard Chat Logic - COMPLETE ORIGINAL ---
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
            
            route_decision = _router.route(task_type, request.user_tier, persona)
            service = _services.get(route_decision['service'])
            
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
            save_chat_history_bg,
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