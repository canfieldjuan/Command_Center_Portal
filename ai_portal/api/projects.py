"""
Project management API endpoints - COMPLETE VERSION
ALL original project functionality from main.py preserved
"""

import uuid
import structlog
from typing import List
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from ..schemas.project import ProjectRequest, ProjectResponse, ProjectUpdateRequest
from ..models.project import Project
from ..models.chat_history import ChatHistory
from ..models.base import get_db_session

logger = structlog.get_logger()
router = APIRouter(prefix="/projects", tags=["projects"])

def get_db():
    """Dependency to get database session"""
    with get_db_session() as session:
        yield session

@router.post("", response_model=ProjectResponse, status_code=201)
async def create_project(request: ProjectRequest, db: Session = Depends(get_db)):
    """Create a new project - COMPLETE ORIGINAL"""
    logger.info("Creating new project", name=request.name, user_id=request.user_id)
    
    try:
        project = Project(
            name=request.name,
            description=request.description,
            user_id=request.user_id
        )
        db.add(project)
        db.commit()
        db.refresh(project)
        
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
        db.rollback()
        logger.error("Failed to create project", 
                   name=request.name,
                   error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create project")

@router.get("", response_model=List[ProjectResponse])
async def list_projects(user_id: str, db: Session = Depends(get_db)):
    """List all projects for a user - COMPLETE ORIGINAL"""
    logger.debug("Listing projects", user_id=user_id)
    
    try:
        projects = db.query(Project).filter(Project.user_id == user_id).all()
        
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

@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: str, user_id: str, db: Session = Depends(get_db)):
    """Get specific project by ID - COMPLETE ORIGINAL"""
    logger.debug("Getting project", project_id=project_id, user_id=user_id)
    
    try:
        project_uuid = uuid.UUID(project_id)
        project = db.query(Project).filter(
            Project.id == project_uuid,
            Project.user_id == user_id
        ).first()
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        return ProjectResponse(
            id=str(project.id),
            name=project.name,
            description=project.description,
            user_id=project.user_id,
            created_at=project.created_at,
            updated_at=project.updated_at
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid project ID format")
    except Exception as e:
        logger.error("Failed to get project", 
                   project_id=project_id,
                   error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve project")

@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(project_id: str, request: ProjectUpdateRequest, 
                        user_id: str, db: Session = Depends(get_db)):
    """Update existing project - COMPLETE ORIGINAL"""
    logger.info("Updating project", project_id=project_id, user_id=user_id)
    
    try:
        project_uuid = uuid.UUID(project_id)
        project = db.query(Project).filter(
            Project.id == project_uuid,
            Project.user_id == user_id
        ).first()
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Update fields if provided
        if request.name is not None:
            project.name = request.name
        if request.description is not None:
            project.description = request.description
        
        db.commit()
        db.refresh(project)
        
        logger.info("Project updated successfully", 
                   project_id=project_id,
                   name=project.name)
        
        return ProjectResponse(
            id=str(project.id),
            name=project.name,
            description=project.description,
            user_id=project.user_id,
            created_at=project.created_at,
            updated_at=project.updated_at
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid project ID format")
    except Exception as e:
        db.rollback()
        logger.error("Failed to update project", 
                   project_id=project_id,
                   error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update project")

@router.delete("/{project_id}")
async def delete_project(project_id: str, user_id: str, db: Session = Depends(get_db)):
    """Delete a project and all associated data - COMPLETE ORIGINAL"""
    logger.info("Deleting project", project_id=project_id, user_id=user_id)
    
    try:
        project_uuid = uuid.UUID(project_id)
        project = db.query(Project).filter(
            Project.id == project_uuid,
            Project.user_id == user_id
        ).first()
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        project_name = project.name
        db.delete(project)
        db.commit()
        
        logger.info("Project deleted successfully", 
                   project_id=project_id,
                   name=project_name)
        
        return {"status": "deleted", "project_id": project_id, "name": project_name}
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid project ID format")
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error("Failed to delete project", 
                   project_id=project_id,
                   error=str(e))
        raise HTTPException(status_code=500, detail="Failed to delete project")

@router.get("/{project_id}/history")
async def get_project_history(project_id: str, user_id: str, limit: int = 50,
                             db: Session = Depends(get_db)):
    """Get chat history for a specific project - COMPLETE ORIGINAL"""
    logger.debug("Getting project history", project_id=project_id, user_id=user_id)
    
    try:
        project_uuid = uuid.UUID(project_id)
        history = ChatHistory.get_project_history(db, project_uuid, user_id, limit)
        
        logger.debug("Project history retrieved", 
                   project_id=project_id,
                   count=len(history))
        
        return history
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid project ID format")
    except Exception as e:
        logger.error("Failed to get project history", 
                   project_id=project_id,
                   error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve project history")