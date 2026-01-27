"""Admin routes for token, model, and security management."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status

from app.core.security import (
    AuthManager,
    get_auth_manager,
    require_admin,
    Token,
    TokenCreate,
    TokenScope,
)
from app.core.security.models import TokenResponse, TokenInfo, AuthStatus
from app.core.database import get_db_manager, ModelRepository, AuditRepository, RequestLogRepository, MetricsRepository
from app.core.database.repositories.audit_repo import AuditAction
from app.core.logger import logger
from app.schemas.models import (
    ModelCreate,
    ModelUpdate,
    ModelInfo,
    ModelDetailInfo,
    ModelListResponse,
    ModelStatistics,
    ModelFileCreate,
    ModelFileInfo,
)

router = APIRouter()


# ============================================
# Dependency injection for repositories
# ============================================

def get_model_repo() -> ModelRepository:
    """Get ModelRepository instance."""
    return ModelRepository()


def get_audit_repo() -> AuditRepository:
    """Get AuditRepository instance."""
    return AuditRepository()


def get_request_log_repo() -> RequestLogRepository:
    """Get RequestLogRepository instance."""
    return RequestLogRepository()


def get_metrics_repo() -> MetricsRepository:
    """Get MetricsRepository instance."""
    return MetricsRepository()


# ============================================
# Authentication endpoints
# ============================================

@router.get("/auth/status", response_model=AuthStatus)
async def get_auth_status(
    token: Optional[Token] = Depends(require_admin),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """
    Get current authentication status.

    Returns whether authentication is enabled and current token info if authenticated.
    """
    return AuthStatus(
        authenticated=token is not None,
        auth_enabled=auth_manager.enabled,
        token_name=token.name if token else None,
        scopes=token.scopes if token else []
    )


# ============================================
# Token management endpoints
# ============================================

@router.get("/tokens", response_model=List[TokenInfo])
async def list_tokens(
    _: Optional[Token] = Depends(require_admin),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """
    List all tokens.

    Requires admin privileges. Returns token metadata without sensitive data.
    """
    if not auth_manager.enabled or not auth_manager.db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Authentication is not enabled"
        )

    tokens = auth_manager.db.list_tokens()
    return [
        TokenInfo(
            id=t.id,
            name=t.name,
            scopes=t.scopes,
            created_at=t.created_at,
            expires_at=t.expires_at,
            last_used_at=t.last_used_at,
            is_active=t.is_active
        )
        for t in tokens
    ]


@router.post("/tokens", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def create_token(
    token_data: TokenCreate,
    current_token: Optional[Token] = Depends(require_admin),
    auth_manager: AuthManager = Depends(get_auth_manager),
    audit_repo: AuditRepository = Depends(get_audit_repo)
):
    """
    Create a new API token.

    Requires admin privileges.

    **IMPORTANT**: The plain token is only returned ONCE in this response.
    Store it securely - it cannot be retrieved again!

    **Scopes**:
    - `read`: Can use tagging endpoints (GET/POST /api/tag, /api/batch, etc.)
    - `write`: Can modify cache (POST /api/cache/clear, /api/cache/cleanup)
    - `admin`: Full access including token management
    """
    if not auth_manager.enabled or not auth_manager.db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Authentication is not enabled"
        )

    try:
        token_obj, plain_token = auth_manager.create_token(
            name=token_data.name,
            scopes=token_data.scopes,
            expires_days=token_data.expires_days
        )

        # Audit log
        audit_repo.log(
            action=AuditAction.TOKEN_CREATED,
            actor_token_id=current_token.id if current_token else None,
            target_type="token",
            target_id=str(token_obj.id),
            details={"name": token_data.name, "scopes": [s.value for s in token_data.scopes]}
        )

        logger.info(f"Created token '{token_data.name}' with scopes: {[s.value for s in token_data.scopes]}")

        return TokenResponse(
            id=token_obj.id,
            name=token_obj.name,
            token=plain_token,
            scopes=token_obj.scopes,
            created_at=token_obj.created_at,
            expires_at=token_obj.expires_at
        )

    except Exception as e:
        logger.error(f"Error creating token: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create token"
        )


@router.delete("/tokens/{token_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_token(
    token_id: int,
    current_token: Optional[Token] = Depends(require_admin),
    auth_manager: AuthManager = Depends(get_auth_manager),
    audit_repo: AuditRepository = Depends(get_audit_repo)
):
    """
    Revoke a token by ID.

    Requires admin privileges. Revoked tokens cannot be used for authentication.
    """
    if not auth_manager.enabled or not auth_manager.db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Authentication is not enabled"
        )

    # Prevent revoking own token
    if current_token and current_token.id == token_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot revoke your own token"
        )

    success = auth_manager.db.revoke_token(token_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Token with id {token_id} not found"
        )

    # Audit log
    audit_repo.log(
        action=AuditAction.TOKEN_REVOKED,
        actor_token_id=current_token.id if current_token else None,
        target_type="token",
        target_id=str(token_id)
    )

    logger.info(f"Revoked token id={token_id}")


@router.delete("/tokens/{token_id}/permanent", status_code=status.HTTP_204_NO_CONTENT)
async def delete_token_permanently(
    token_id: int,
    current_token: Optional[Token] = Depends(require_admin),
    auth_manager: AuthManager = Depends(get_auth_manager),
    audit_repo: AuditRepository = Depends(get_audit_repo)
):
    """
    Permanently delete a token.

    Requires admin privileges. This action cannot be undone.
    """
    if not auth_manager.enabled or not auth_manager.db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Authentication is not enabled"
        )

    # Prevent deleting own token
    if current_token and current_token.id == token_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own token"
        )

    success = auth_manager.db.delete_token(token_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Token with id {token_id} not found"
        )

    # Audit log
    audit_repo.log(
        action=AuditAction.TOKEN_DELETED,
        actor_token_id=current_token.id if current_token else None,
        target_type="token",
        target_id=str(token_id)
    )

    logger.info(f"Permanently deleted token id={token_id}")


@router.post("/tokens/cleanup")
async def cleanup_expired_tokens(
    current_token: Optional[Token] = Depends(require_admin),
    auth_manager: AuthManager = Depends(get_auth_manager),
    audit_repo: AuditRepository = Depends(get_audit_repo)
):
    """
    Remove all expired tokens from the database.

    Requires admin privileges.
    """
    if not auth_manager.enabled or not auth_manager.db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Authentication is not enabled"
        )

    count = auth_manager.db.cleanup_expired()

    # Audit log
    audit_repo.log(
        action=AuditAction.TOKEN_EXPIRED_CLEANUP,
        actor_token_id=current_token.id if current_token else None,
        target_type="token",
        details={"removed_count": count}
    )

    return {"removed_tokens": count, "message": f"Cleaned up {count} expired tokens"}


@router.get("/tokens/stats")
async def get_token_stats(
    _: Optional[Token] = Depends(require_admin),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """
    Get token statistics.

    Requires admin privileges.
    """
    if not auth_manager.enabled or not auth_manager.db:
        return {
            "auth_enabled": False,
            "message": "Authentication is not enabled"
        }

    stats = auth_manager.db.get_token_count()
    stats["auth_enabled"] = True
    return stats


# ============================================
# Model management endpoints
# ============================================

@router.get("/models", response_model=ModelListResponse)
async def list_models(
    include_inactive: bool = False,
    _: Optional[Token] = Depends(require_admin),
    model_repo: ModelRepository = Depends(get_model_repo)
):
    """
    List all models.

    Requires admin privileges.

    Args:
        include_inactive: Include inactive models in the list
    """
    models = model_repo.get_all(include_inactive=include_inactive)
    stats = model_repo.get_statistics()

    return ModelListResponse(
        models=[
            ModelInfo(
                id=m.id,
                code=m.code,
                name=m.name,
                description=m.description,
                pie_module=m.pie_module,
                is_active=m.is_active,
                is_builtin=m.is_builtin,
                batch_size=m.batch_size,
                priority=m.priority,
                created_at=m.created_at,
                updated_at=m.updated_at
            )
            for m in models
        ],
        total=stats["total"],
        active=stats["active"]
    )


@router.get("/models/stats", response_model=ModelStatistics)
async def get_model_statistics(
    _: Optional[Token] = Depends(require_admin),
    model_repo: ModelRepository = Depends(get_model_repo)
):
    """
    Get model statistics.

    Requires admin privileges.
    """
    return model_repo.get_statistics()


@router.get("/models/{code}", response_model=ModelDetailInfo)
async def get_model(
    code: str,
    _: Optional[Token] = Depends(require_admin),
    model_repo: ModelRepository = Depends(get_model_repo)
):
    """
    Get detailed information about a specific model.

    Requires admin privileges.
    """
    model = model_repo.get_by_code(code)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{code}' not found"
        )

    files = model_repo.get_model_files(code)
    total_size = sum(f.size_bytes or 0 for f in files if f.is_downloaded)

    # Get runtime info from model_manager
    from app.core.model_manager import model_manager

    model_status = model_manager.get_model_status(code)
    device = model_manager.device
    has_custom_processor = code in model_manager.iterator_processors

    # Get metrics
    metrics = None
    if model_manager._metrics and code in model_manager._metrics.models:
        metrics = model_manager._metrics.models[code].to_dict()

    return ModelDetailInfo(
        id=model.id,
        code=model.code,
        name=model.name,
        description=model.description,
        pie_module=model.pie_module,
        is_active=model.is_active,
        is_builtin=model.is_builtin,
        batch_size=model.batch_size,
        priority=model.priority,
        created_at=model.created_at,
        updated_at=model.updated_at,
        files=[
            ModelFileInfo(
                id=f.id,
                filename=f.filename,
                url=f.url,
                size_bytes=f.size_bytes,
                checksum=f.checksum,
                is_downloaded=f.is_downloaded,
                downloaded_at=f.downloaded_at
            )
            for f in files
        ],
        total_size_mb=round(total_size / (1024 * 1024), 2),
        status=model_status,
        device=device,
        has_custom_processor=has_custom_processor,
        metrics=metrics
    )


@router.post("/models", response_model=ModelInfo, status_code=status.HTTP_201_CREATED)
async def create_model(
    model_data: ModelCreate,
    current_token: Optional[Token] = Depends(require_admin),
    model_repo: ModelRepository = Depends(get_model_repo),
    audit_repo: AuditRepository = Depends(get_audit_repo)
):
    """
    Create a new model.

    Requires admin privileges.

    This allows adding custom models that are available in pie_extended
    but not included in the default set.
    """
    # Check if model already exists
    existing = model_repo.get_by_code(model_data.code)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model '{model_data.code}' already exists"
        )

    try:
        model = model_repo.create(
            code=model_data.code,
            name=model_data.name,
            pie_module=model_data.pie_module,
            description=model_data.description,
            batch_size=model_data.batch_size,
            priority=model_data.priority
        )

        # Audit log
        audit_repo.log(
            action=AuditAction.MODEL_CREATED,
            actor_token_id=current_token.id if current_token else None,
            target_type="model",
            target_id=model_data.code,
            details={"name": model_data.name, "pie_module": model_data.pie_module}
        )

        logger.info(f"Created model '{model_data.code}'")

        return ModelInfo(
            id=model.id,
            code=model.code,
            name=model.name,
            description=model.description,
            pie_module=model.pie_module,
            is_active=model.is_active,
            is_builtin=model.is_builtin,
            batch_size=model.batch_size,
            priority=model.priority,
            created_at=model.created_at,
            updated_at=model.updated_at
        )

    except Exception as e:
        logger.error(f"Error creating model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create model"
        )


@router.patch("/models/{code}", response_model=ModelInfo)
async def update_model(
    code: str,
    model_data: ModelUpdate,
    current_token: Optional[Token] = Depends(require_admin),
    model_repo: ModelRepository = Depends(get_model_repo),
    audit_repo: AuditRepository = Depends(get_audit_repo)
):
    """
    Update a model's properties.

    Requires admin privileges.
    """
    model = model_repo.get_by_code(code)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{code}' not found"
        )

    updated = model_repo.update(
        code=code,
        name=model_data.name,
        description=model_data.description,
        pie_module=model_data.pie_module,
        batch_size=model_data.batch_size,
        priority=model_data.priority
    )

    # Audit log
    audit_repo.log(
        action=AuditAction.MODEL_UPDATED,
        actor_token_id=current_token.id if current_token else None,
        target_type="model",
        target_id=code,
        details=model_data.model_dump(exclude_none=True)
    )

    return ModelInfo(
        id=updated.id,
        code=updated.code,
        name=updated.name,
        description=updated.description,
        pie_module=updated.pie_module,
        is_active=updated.is_active,
        is_builtin=updated.is_builtin,
        batch_size=updated.batch_size,
        priority=updated.priority,
        created_at=updated.created_at,
        updated_at=updated.updated_at
    )


@router.delete("/models/{code}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    code: str,
    current_token: Optional[Token] = Depends(require_admin),
    model_repo: ModelRepository = Depends(get_model_repo),
    audit_repo: AuditRepository = Depends(get_audit_repo)
):
    """
    Delete a model.

    Requires admin privileges. Builtin models cannot be deleted.
    """
    model = model_repo.get_by_code(code)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{code}' not found"
        )

    if model.is_builtin:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete builtin models. Use deactivate instead."
        )

    success = model_repo.delete(code)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete model"
        )

    # Audit log
    audit_repo.log(
        action=AuditAction.MODEL_DELETED,
        actor_token_id=current_token.id if current_token else None,
        target_type="model",
        target_id=code
    )

    logger.info(f"Deleted model '{code}'")


@router.post("/models/{code}/activate", response_model=ModelInfo)
async def activate_model(
    code: str,
    current_token: Optional[Token] = Depends(require_admin),
    model_repo: ModelRepository = Depends(get_model_repo),
    audit_repo: AuditRepository = Depends(get_audit_repo)
):
    """
    Activate a model.

    Requires admin privileges.
    """
    model = model_repo.get_by_code(code)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{code}' not found"
        )

    if model.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{code}' is already active"
        )

    model_repo.activate(code)

    # Audit log
    audit_repo.log(
        action=AuditAction.MODEL_ACTIVATED,
        actor_token_id=current_token.id if current_token else None,
        target_type="model",
        target_id=code
    )

    return ModelInfo(
        id=model.id,
        code=model.code,
        name=model.name,
        description=model.description,
        pie_module=model.pie_module,
        is_active=True,
        is_builtin=model.is_builtin,
        batch_size=model.batch_size,
        priority=model.priority,
        created_at=model.created_at,
        updated_at=model.updated_at
    )


@router.post("/models/{code}/deactivate", response_model=ModelInfo)
async def deactivate_model(
    code: str,
    current_token: Optional[Token] = Depends(require_admin),
    model_repo: ModelRepository = Depends(get_model_repo),
    audit_repo: AuditRepository = Depends(get_audit_repo)
):
    """
    Deactivate a model.

    Requires admin privileges. Deactivated models cannot be used for tagging.
    """
    model = model_repo.get_by_code(code)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{code}' not found"
        )

    if not model.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{code}' is already inactive"
        )

    model_repo.deactivate(code)

    # Unload from memory if loaded
    from app.core.model_manager import model_manager
    if code in model_manager.taggers:
        await model_manager.unload_model(code)

    # Audit log
    audit_repo.log(
        action=AuditAction.MODEL_DEACTIVATED,
        actor_token_id=current_token.id if current_token else None,
        target_type="model",
        target_id=code
    )

    return ModelInfo(
        id=model.id,
        code=model.code,
        name=model.name,
        description=model.description,
        pie_module=model.pie_module,
        is_active=False,
        is_builtin=model.is_builtin,
        batch_size=model.batch_size,
        priority=model.priority,
        created_at=model.created_at,
        updated_at=model.updated_at
    )


@router.post("/models/{code}/files", response_model=ModelFileInfo, status_code=status.HTTP_201_CREATED)
async def add_model_file(
    code: str,
    file_data: ModelFileCreate,
    _: Optional[Token] = Depends(require_admin),
    model_repo: ModelRepository = Depends(get_model_repo)
):
    """
    Add a file to a model.

    Requires admin privileges. This is useful for custom models
    that need specific files to be downloaded.
    """
    model = model_repo.get_by_code(code)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{code}' not found"
        )

    file = model_repo.add_model_file(
        code=code,
        filename=file_data.filename,
        url=file_data.url,
        size_bytes=file_data.size_bytes,
        checksum=file_data.checksum
    )

    if not file:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add model file"
        )

    return ModelFileInfo(
        id=file.id,
        filename=file.filename,
        url=file.url,
        size_bytes=file.size_bytes,
        checksum=file.checksum,
        is_downloaded=file.is_downloaded,
        downloaded_at=file.downloaded_at
    )


# ============================================
# Audit log endpoints
# ============================================

@router.get("/audit")
async def get_audit_log(
    limit: int = 100,
    offset: int = 0,
    action: Optional[str] = None,
    _: Optional[Token] = Depends(require_admin),
    audit_repo: AuditRepository = Depends(get_audit_repo)
):
    """
    Get audit log entries.

    Requires admin privileges.

    Args:
        limit: Maximum number of entries to return
        offset: Number of entries to skip
        action: Filter by action type
    """
    if action:
        entries = audit_repo.get_by_action(action, limit=limit)
    else:
        entries = audit_repo.get_recent(limit=limit, offset=offset)

    return {
        "entries": [
            {
                "id": e.id,
                "timestamp": e.timestamp.isoformat(),
                "action": e.action,
                "actor_token_id": e.actor_token_id,
                "target_type": e.target_type,
                "target_id": e.target_id,
                "details": e.details,
                "client_ip": e.client_ip,
                "success": e.success
            }
            for e in entries
        ],
        "count": len(entries)
    }


@router.get("/audit/stats")
async def get_audit_statistics(
    hours: int = 24,
    _: Optional[Token] = Depends(require_admin),
    audit_repo: AuditRepository = Depends(get_audit_repo)
):
    """
    Get audit statistics.

    Requires admin privileges.

    Args:
        hours: Number of hours to look back
    """
    return audit_repo.get_statistics(hours=hours)


@router.post("/audit/cleanup")
async def cleanup_audit_log(
    days: int = 90,
    _: Optional[Token] = Depends(require_admin),
    audit_repo: AuditRepository = Depends(get_audit_repo)
):
    """
    Clean up old audit log entries.

    Requires admin privileges.

    Args:
        days: Remove entries older than this many days
    """
    count = audit_repo.cleanup(days=days)
    return {"removed_entries": count, "message": f"Cleaned up {count} audit entries older than {days} days"}


# ============================================
# Request log endpoints
# ============================================

@router.get("/requests")
async def get_request_log(
    limit: int = 100,
    offset: int = 0,
    model: Optional[str] = None,
    _: Optional[Token] = Depends(require_admin),
    request_log_repo: RequestLogRepository = Depends(get_request_log_repo)
):
    """
    Get request log entries.

    Requires admin privileges.

    Args:
        limit: Maximum number of entries to return
        offset: Number of entries to skip
        model: Filter by model code
    """
    if model:
        entries = request_log_repo.get_by_model(model, limit=limit)
    else:
        entries = request_log_repo.get_recent(limit=limit, offset=offset)

    return {
        "entries": [
            {
                "id": e.id,
                "timestamp": e.timestamp.isoformat(),
                "endpoint": e.endpoint,
                "method": e.method,
                "model_id": e.model_id,
                "token_id": e.token_id,
                "text_length": e.text_length,
                "processing_time_ms": e.processing_time_ms,
                "from_cache": e.from_cache,
                "status_code": e.status_code,
                "error_message": e.error_message,
                "client_ip": e.client_ip
            }
            for e in entries
        ],
        "count": len(entries)
    }


@router.get("/requests/stats")
async def get_request_statistics(
    hours: int = 24,
    _: Optional[Token] = Depends(require_admin),
    request_log_repo: RequestLogRepository = Depends(get_request_log_repo)
):
    """
    Get request statistics.

    Requires admin privileges.

    Args:
        hours: Number of hours to look back
    """
    return request_log_repo.get_statistics(hours=hours)


@router.get("/requests/errors")
async def get_request_errors(
    limit: int = 100,
    _: Optional[Token] = Depends(require_admin),
    request_log_repo: RequestLogRepository = Depends(get_request_log_repo)
):
    """
    Get recent error requests.

    Requires admin privileges.
    """
    entries = request_log_repo.get_errors(limit=limit)

    return {
        "entries": [
            {
                "id": e.id,
                "timestamp": e.timestamp.isoformat(),
                "endpoint": e.endpoint,
                "method": e.method,
                "status_code": e.status_code,
                "error_message": e.error_message,
                "client_ip": e.client_ip
            }
            for e in entries
        ],
        "count": len(entries)
    }


@router.post("/requests/cleanup")
async def cleanup_request_log(
    days: int = 30,
    _: Optional[Token] = Depends(require_admin),
    request_log_repo: RequestLogRepository = Depends(get_request_log_repo)
):
    """
    Clean up old request log entries.

    Requires admin privileges.

    Args:
        days: Remove entries older than this many days
    """
    count = request_log_repo.cleanup(days=days)
    return {"removed_entries": count, "message": f"Cleaned up {count} request log entries older than {days} days"}


# ============================================
# Metrics endpoints (persistent)
# ============================================

@router.get("/metrics/persistent")
async def get_persistent_metrics(
    _: Optional[Token] = Depends(require_admin),
    metrics_repo: MetricsRepository = Depends(get_metrics_repo)
):
    """
    Get persistent metrics from database.

    Returns historical metrics that persist across server restarts.
    Requires admin privileges.
    """
    all_metrics = metrics_repo.get_all()
    global_stats = metrics_repo.get_global_statistics()

    return {
        "global": global_stats,
        "models": {
            code: metrics.to_dict()
            for code, metrics in all_metrics.items()
        }
    }


@router.get("/metrics/persistent/{model}")
async def get_model_persistent_metrics(
    model: str,
    _: Optional[Token] = Depends(require_admin),
    metrics_repo: MetricsRepository = Depends(get_metrics_repo)
):
    """
    Get persistent metrics for a specific model.

    Requires admin privileges.
    """
    metrics = metrics_repo.get_by_model(model)
    if not metrics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No metrics found for model '{model}'"
        )

    return metrics.to_dict()


@router.post("/metrics/reset")
async def reset_metrics(
    model: Optional[str] = None,
    _: Optional[Token] = Depends(require_admin),
    metrics_repo: MetricsRepository = Depends(get_metrics_repo)
):
    """
    Reset metrics.

    Requires admin privileges.

    Args:
        model: If specified, reset only this model's metrics. Otherwise reset all.
    """
    if model:
        success = metrics_repo.reset(model)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model}' not found"
            )
        return {"message": f"Reset metrics for model '{model}'"}
    else:
        count = metrics_repo.reset_all()
        return {"message": f"Reset metrics for {count} models"}
