"""
SQLModel models for PyHellen database.

These models define the database schema and can be used directly
as Pydantic models for API validation.
"""

from datetime import datetime
from typing import List, Optional
from sqlmodel import SQLModel, Field, Relationship


# ==================
# Model Management
# ==================

class ModelBase(SQLModel):
    """Base model fields shared between create and read."""
    code: str = Field(unique=True, index=True)
    name: str
    description: Optional[str] = None
    pie_module: str
    is_active: bool = Field(default=True, index=True)
    is_builtin: bool = Field(default=False)
    batch_size: Optional[int] = None
    priority: int = Field(default=0)


class Model(ModelBase, table=True):
    """NLP Model definition stored in database."""
    __tablename__ = "models"

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    files: List["ModelFile"] = Relationship(back_populates="model", cascade_delete=True)
    metrics: Optional["ModelMetrics"] = Relationship(back_populates="model", cascade_delete=True)
    cache_entries: List["CacheEntry"] = Relationship(back_populates="model", cascade_delete=True)


class ModelCreate(ModelBase):
    """Schema for creating a new model."""
    pass


class ModelRead(ModelBase):
    """Schema for reading a model."""
    id: int
    created_at: datetime
    updated_at: datetime


class ModelUpdate(SQLModel):
    """Schema for updating a model."""
    name: Optional[str] = None
    description: Optional[str] = None
    pie_module: Optional[str] = None
    batch_size: Optional[int] = None
    priority: Optional[int] = None


# ==================
# Model Files
# ==================

class ModelFileBase(SQLModel):
    """Base fields for model files."""
    filename: str
    url: str
    size_bytes: Optional[int] = None
    checksum: Optional[str] = None


class ModelFile(ModelFileBase, table=True):
    """Model file metadata."""
    __tablename__ = "model_files"

    id: Optional[int] = Field(default=None, primary_key=True)
    model_id: int = Field(foreign_key="models.id", index=True)
    is_downloaded: bool = Field(default=False)
    downloaded_at: Optional[datetime] = None

    # Relationship
    model: Optional[Model] = Relationship(back_populates="files")


# ==================
# Model Metrics
# ==================

class ModelMetrics(SQLModel, table=True):
    """Metrics for model usage."""
    __tablename__ = "model_metrics"

    id: Optional[int] = Field(default=None, primary_key=True)
    model_id: int = Field(foreign_key="models.id", unique=True)

    load_count: int = Field(default=0)
    load_time_total_ms: float = Field(default=0.0)
    last_loaded_at: Optional[datetime] = None

    process_count: int = Field(default=0)
    process_time_total_ms: float = Field(default=0.0)
    last_used_at: Optional[datetime] = None

    download_count: int = Field(default=0)
    download_time_total_ms: float = Field(default=0.0)
    download_size_bytes: int = Field(default=0)

    error_count: int = Field(default=0)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationship
    model: Optional[Model] = Relationship(back_populates="metrics")

    @property
    def avg_load_time_ms(self) -> float:
        return self.load_time_total_ms / self.load_count if self.load_count > 0 else 0

    @property
    def avg_process_time_ms(self) -> float:
        return self.process_time_total_ms / self.process_count if self.process_count > 0 else 0


# ==================
# Cache
# ==================

class CacheEntry(SQLModel, table=True):
    """Cached tagging result."""
    __tablename__ = "cache_entries"

    id: Optional[int] = Field(default=None, primary_key=True)
    cache_key: str = Field(unique=True, index=True)
    model_id: int = Field(foreign_key="models.id", index=True)
    text_hash: str
    text_preview: Optional[str] = None
    result_json: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime = Field(index=True)
    hit_count: int = Field(default=0)
    last_hit_at: Optional[datetime] = None
    size_bytes: int = Field(default=0)

    # Relationship
    model: Optional[Model] = Relationship(back_populates="cache_entries")


# ==================
# Tokens
# ==================

class Token(SQLModel, table=True):
    """API access token."""
    __tablename__ = "tokens"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    token_hash: str = Field(unique=True, index=True)
    scopes: str  # Comma-separated list of scopes
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    is_active: bool = Field(default=True, index=True)


class TokenCreate(SQLModel):
    """Schema for creating a token."""
    name: str
    scopes: List[str] = Field(default_factory=lambda: ["read"])
    expires_in_days: Optional[int] = None


# ==================
# Request Log
# ==================

class RequestLog(SQLModel, table=True):
    """HTTP request log entry."""
    __tablename__ = "request_log"

    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow, index=True)
    model_id: Optional[int] = Field(default=None, foreign_key="models.id")
    endpoint: str
    method: str
    token_id: Optional[int] = Field(default=None, foreign_key="tokens.id")
    text_length: Optional[int] = None
    batch_size: Optional[int] = None
    processing_time_ms: Optional[float] = None
    from_cache: bool = Field(default=False)
    status_code: int
    error_message: Optional[str] = None
    client_ip: Optional[str] = None


# ==================
# Audit Log
# ==================

class AuditLog(SQLModel, table=True):
    """Audit trail entry."""
    __tablename__ = "audit_log"

    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow, index=True)
    action: str = Field(index=True)
    actor_token_id: Optional[int] = Field(default=None, foreign_key="tokens.id")
    target_type: Optional[str] = None
    target_id: Optional[str] = None
    details_json: Optional[str] = None
    client_ip: Optional[str] = None
    success: bool = Field(default=True)


# ==================
# App State
# ==================

class AppState(SQLModel, table=True):
    """Key-value application state."""
    __tablename__ = "app_state"

    key: str = Field(primary_key=True)
    value_json: str
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ==================
# Schema Migrations
# ==================

class SchemaMigration(SQLModel, table=True):
    """Track applied database migrations."""
    __tablename__ = "schema_migrations"

    version: int = Field(primary_key=True)
    applied_at: datetime = Field(default_factory=datetime.utcnow)
    description: Optional[str] = None
