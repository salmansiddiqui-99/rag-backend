"""Pydantic models for Module"""
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field
from datetime import datetime


class ModuleBase(BaseModel):
    """Base module schema"""
    name: str = Field(..., min_length=20, max_length=200)
    description: str = Field(..., min_length=20, max_length=500)
    order: int = Field(..., ge=1, le=4)


class ModuleCreate(ModuleBase):
    """Schema for creating a module"""
    pass


class Module(ModuleBase):
    """Full module schema"""
    id: UUID
    created_at: datetime

    class Config:
        from_attributes = True


class ModuleUpdate(BaseModel):
    """Schema for updating a module"""
    name: Optional[str] = Field(None, min_length=20, max_length=200)
    description: Optional[str] = Field(None, min_length=20, max_length=500)
    order: Optional[int] = Field(None, ge=1, le=4)


class ModuleResponse(Module):
    """Schema for module API responses"""
    pass
