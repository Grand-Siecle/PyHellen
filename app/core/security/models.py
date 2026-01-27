"""Security models for token-based authentication."""

from datetime import datetime
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class TokenScope(str, Enum):
    """Available token permission scopes."""
    READ = "read"           # Can use tagging endpoints
    WRITE = "write"         # Can modify cache
    ADMIN = "admin"         # Full access including token management


class TokenCreate(BaseModel):
    """Request model for creating a new token."""
    name: str = Field(..., min_length=1, max_length=100, description="Token name/description")
    scopes: List[TokenScope] = Field(
        default=[TokenScope.READ],
        description="Permission scopes for this token"
    )
    expires_days: Optional[int] = Field(
        None,
        ge=1,
        le=365,
        description="Days until token expires (None = never)"
    )


class Token(BaseModel):
    """Token model returned from database."""
    id: int
    name: str
    token_hash: str
    scopes: List[TokenScope]
    created_at: datetime
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime]
    is_active: bool

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """Response when a new token is created (includes plain token once)."""
    id: int
    name: str
    token: str = Field(..., description="Plain token - save this, it won't be shown again!")
    scopes: List[TokenScope]
    created_at: datetime
    expires_at: Optional[datetime]


class TokenInfo(BaseModel):
    """Token info without sensitive data."""
    id: int
    name: str
    scopes: List[TokenScope]
    created_at: datetime
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime]
    is_active: bool


class AuthStatus(BaseModel):
    """Authentication status response."""
    authenticated: bool
    auth_enabled: bool
    token_name: Optional[str] = None
    scopes: List[TokenScope] = []
