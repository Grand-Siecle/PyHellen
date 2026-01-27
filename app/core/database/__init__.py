"""
Database module for PyHellen API.
Provides unified SQLite database management with SQLModel ORM.

Usage:
    from app.core.database import get_db_engine, ModelRepository

    engine = get_db_engine()
    model_repo = ModelRepository()
    models = model_repo.get_all()
"""

# Engine and session management
from app.core.database.engine import DatabaseEngine, get_db_engine, get_session

# SQLModel models
from app.core.database.models import (
    Model, ModelCreate, ModelRead, ModelUpdate, ModelFile, ModelMetrics,
    Token, TokenCreate,
    CacheEntry, RequestLog, AuditLog, AppState,
)

# Repositories
from app.core.database.repositories.model_repo import ModelRepository
from app.core.database.repositories.token_repo import TokenRepository, TokenScope
from app.core.database.repositories.cache_repo import CacheRepository
from app.core.database.repositories.metrics_repo import MetricsRepository
from app.core.database.repositories.audit_repo import AuditRepository, AuditAction
from app.core.database.repositories.request_log_repo import RequestLogRepository

# Backwards compatibility aliases
DatabaseManager = DatabaseEngine
get_db_manager = get_db_engine

__all__ = [
    # Engine
    "DatabaseEngine",
    "get_db_engine",
    "get_session",
    # Legacy aliases
    "DatabaseManager",
    "get_db_manager",
    # Models
    "Model",
    "ModelCreate",
    "ModelRead",
    "ModelUpdate",
    "ModelFile",
    "ModelMetrics",
    "Token",
    "TokenCreate",
    "CacheEntry",
    "RequestLog",
    "AuditLog",
    "AppState",
    # Repositories
    "ModelRepository",
    "TokenRepository",
    "CacheRepository",
    "MetricsRepository",
    "AuditRepository",
    "RequestLogRepository",
    # Constants
    "TokenScope",
    "AuditAction",
]
