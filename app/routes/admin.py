"""Admin routes for token and security management."""

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
from app.core.logger import logger

router = APIRouter()


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
    _: Optional[Token] = Depends(require_admin),
    auth_manager: AuthManager = Depends(get_auth_manager)
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
    auth_manager: AuthManager = Depends(get_auth_manager)
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

    logger.info(f"Revoked token id={token_id}")


@router.delete("/tokens/{token_id}/permanent", status_code=status.HTTP_204_NO_CONTENT)
async def delete_token_permanently(
    token_id: int,
    current_token: Optional[Token] = Depends(require_admin),
    auth_manager: AuthManager = Depends(get_auth_manager)
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

    logger.info(f"Permanently deleted token id={token_id}")


@router.post("/tokens/cleanup")
async def cleanup_expired_tokens(
    _: Optional[Token] = Depends(require_admin),
    auth_manager: AuthManager = Depends(get_auth_manager)
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
