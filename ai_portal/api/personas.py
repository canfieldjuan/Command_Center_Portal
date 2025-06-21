"""
Persona management API endpoints - COMPLETE VERSION
ALL original persona functionality from main.py preserved
"""

import uuid
import structlog
from typing import List
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from ..schemas.persona import PersonaRequest, PersonaResponse, PersonaUpdateRequest
from ..models.persona import Persona
from ..models.base import get_db_session

logger = structlog.get_logger()
router = APIRouter(prefix="/personas", tags=["personas"])

def get_db():
    """Dependency to get database session"""
    with get_db_session() as session:
        yield session

@router.post("", response_model=PersonaResponse, status_code=201)
async def create_persona(request: PersonaRequest, db: Session = Depends(get_db)):
    """Create a new persona - COMPLETE ORIGINAL"""
    logger.info("Creating new persona", 
               name=request.name,
               user_id=request.user_id)
    
    try:
        persona = Persona.create_with_validation(
            session=db,
            name=request.name,
            system_prompt=request.system_prompt,
            model_preference=request.model_preference,
            user_id=request.user_id
        )
        
        return PersonaResponse(
            id=str(persona.id),
            name=persona.name,
            system_prompt=persona.system_prompt,
            model_preference=persona.model_preference,
            user_id=persona.user_id
        )
        
    except IntegrityError:
        db.rollback()
        logger.warning("Persona name already exists", 
                     name=request.name,
                     user_id=request.user_id)
        raise HTTPException(
            status_code=409,
            detail=f"Persona with name '{request.name}' already exists for this user."
        )
    except ValueError as e:
        db.rollback()
        logger.error("Persona validation failed", 
                   name=request.name,
                   error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        db.rollback()
        logger.error("Failed to create persona", 
                   name=request.name,
                   error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create persona")

@router.get("", response_model=List[PersonaResponse])
async def list_personas(user_id: str, db: Session = Depends(get_db)):
    """List all personas for a user - COMPLETE ORIGINAL"""
    logger.debug("Listing personas", user_id=user_id)
    
    try:
        personas = db.query(Persona).filter(Persona.user_id == user_id).all()
        
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

@router.get("/{persona_id}", response_model=PersonaResponse)
async def get_persona(persona_id: str, user_id: str, db: Session = Depends(get_db)):
    """Get specific persona by ID - COMPLETE ORIGINAL"""
    logger.debug("Getting persona", persona_id=persona_id, user_id=user_id)
    
    try:
        persona_uuid = uuid.UUID(persona_id)
        persona = db.query(Persona).filter(
            Persona.id == persona_uuid,
            Persona.user_id == user_id
        ).first()
        
        if not persona:
            raise HTTPException(status_code=404, detail="Persona not found")
        
        return PersonaResponse(
            id=str(persona.id),
            name=persona.name,
            system_prompt=persona.system_prompt,
            model_preference=persona.model_preference,
            user_id=persona.user_id
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid persona ID format")
    except Exception as e:
        logger.error("Failed to get persona", 
                   persona_id=persona_id,
                   error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve persona")

@router.put("/{persona_id}", response_model=PersonaResponse)
async def update_persona(persona_id: str, request: PersonaUpdateRequest,
                        user_id: str, db: Session = Depends(get_db)):
    """Update existing persona - COMPLETE ORIGINAL"""
    logger.info("Updating persona", persona_id=persona_id, user_id=user_id)
    
    try:
        persona_uuid = uuid.UUID(persona_id)
        persona = db.query(Persona).filter(
            Persona.id == persona_uuid,
            Persona.user_id == user_id
        ).first()
        
        if not persona:
            raise HTTPException(status_code=404, detail="Persona not found")
        
        # Validate and update fields if provided
        if request.name is not None:
            if not persona.validate_name(request.name):
                raise HTTPException(status_code=400, detail="Invalid persona name")
            persona.name = request.name.strip()
            
        if request.system_prompt is not None:
            if not persona.validate_system_prompt(request.system_prompt):
                raise HTTPException(status_code=400, detail="Invalid system prompt")
            persona.system_prompt = request.system_prompt.strip()
            
        if request.model_preference is not None:
            if not persona.validate_model_preference(request.model_preference):
                raise HTTPException(status_code=400, detail="Invalid model preference")
            persona.model_preference = request.model_preference.strip() if request.model_preference.strip() else None
        
        db.commit()
        db.refresh(persona)
        
        logger.info("Persona updated successfully", 
                   persona_id=persona_id,
                   name=persona.name)
        
        return PersonaResponse(
            id=str(persona.id),
            name=persona.name,
            system_prompt=persona.system_prompt,
            model_preference=persona.model_preference,
            user_id=persona.user_id
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid persona ID format")
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error("Failed to update persona", 
                   persona_id=persona_id,
                   error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update persona")

@router.delete("/{persona_id}")
async def delete_persona(persona_id: str, user_id: str, db: Session = Depends(get_db)):
    """Delete a persona - COMPLETE ORIGINAL"""
    logger.info("Deleting persona", persona_id=persona_id, user_id=user_id)
    
    try:
        persona_uuid = uuid.UUID(persona_id)
        persona = db.query(Persona).filter(
            Persona.id == persona_uuid,
            Persona.user_id == user_id
        ).first()
        
        if not persona:
            raise HTTPException(status_code=404, detail="Persona not found")
        
        persona_name = persona.name
        db.delete(persona)
        db.commit()
        
        logger.info("Persona deleted successfully", 
                   persona_id=persona_id,
                   name=persona_name)
        
        return {"status": "deleted", "persona_id": persona_id, "name": persona_name}
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid persona ID format")
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error("Failed to delete persona", 
                   persona_id=persona_id,
                   error=str(e))
        raise HTTPException(status_code=500, detail="Failed to delete persona")