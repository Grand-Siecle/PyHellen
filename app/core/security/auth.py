"""Authentication manager and FastAPI dependencies."""

import secrets
from typing import Optional, List
from fastapi import Depends, HTTPException, Header, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.security.database import TokenDatabase
from app.core.security.models import Token, TokenScope
from app.core.logger import logger


# HTTP Bearer scheme for OpenAPI docs
security_scheme = HTTPBearer(auto_error=False)


class AuthManager:
    """
    Manages authentication for the application.

    Supports optional authentication - when disabled, all requests are allowed.
    When enabled, validates Bearer tokens against the database.
    """

    _instance: Optional["AuthManager"] = None

    def __init__(
        self,
        enabled: bool = False,
        secret_key: str = "",
        db_path: str = "tokens.db",
        auto_create_admin: bool = True
    ):
        self.enabled = enabled
        self.secret_key = secret_key
        self._db: Optional[TokenDatabase] = None
        self._db_path = db_path
        self._auto_create_admin = auto_create_admin
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the auth manager (lazy initialization)."""
        if self._initialized:
            return

        if self.enabled:
            if not self.secret_key:
                raise ValueError("SECRET_KEY is required when authentication is enabled")

            self._db = TokenDatabase(self._db_path)

            # Create initial admin token if no tokens exist
            if self._auto_create_admin:
                stats = self._db.get_token_count()
                if stats["total"] == 0:
                    self._create_initial_admin_token()

            logger.info("Authentication enabled")
        else:
            logger.warning("Authentication DISABLED - API is publicly accessible")

        self._initialized = True

    def _create_initial_admin_token(self) -> None:
        """Create initial admin token on first run."""
        token_obj, plain_token = self._db.create_token(
            name="Initial Admin Token",
            scopes=[TokenScope.READ, TokenScope.WRITE, TokenScope.ADMIN],
            secret_key=self.secret_key,
            expires_days=None  # Never expires
        )

        # Log the token prominently - this is the only time it's visible
        logger.warning("=" * 60)
        logger.warning("INITIAL ADMIN TOKEN CREATED")
        logger.warning("Save this token - it will NOT be shown again!")
        logger.warning(f"Token: {plain_token}")
        logger.warning("=" * 60)

    @property
    def db(self) -> Optional[TokenDatabase]:
        """Get the token database."""
        return self._db

    def validate_token(self, token: str) -> Optional[Token]:
        """Validate a token and return token data if valid."""
        if not self.enabled or not self._db:
            return None
        return self._db.validate_token(token, self.secret_key)

    def create_token(
        self,
        name: str,
        scopes: List[TokenScope],
        expires_days: Optional[int] = None
    ) -> tuple:
        """Create a new token."""
        if not self._db:
            raise RuntimeError("Token database not initialized")
        return self._db.create_token(name, scopes, self.secret_key, expires_days)

    @classmethod
    def get_instance(cls) -> "AuthManager":
        """Get the singleton instance."""
        if cls._instance is None:
            raise RuntimeError("AuthManager not initialized. Call setup_auth() first.")
        return cls._instance

    @classmethod
    def setup(
        cls,
        enabled: bool,
        secret_key: str,
        db_path: str = "tokens.db",
        auto_create_admin: bool = True
    ) -> "AuthManager":
        """Setup and return the singleton instance."""
        cls._instance = cls(
            enabled=enabled,
            secret_key=secret_key,
            db_path=db_path,
            auto_create_admin=auto_create_admin
        )
        cls._instance.initialize()
        return cls._instance


def get_auth_manager() -> AuthManager:
    """FastAPI dependency to get the auth manager."""
    return AuthManager.get_instance()


async def _extract_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> Optional[str]:
    """Extract token from Authorization header or X-API-Key header."""
    if credentials and credentials.credentials:
        return credentials.credentials
    if x_api_key:
        return x_api_key
    return None


async def optional_auth(
    token: Optional[str] = Depends(_extract_token),
    auth_manager: AuthManager = Depends(get_auth_manager)
) -> Optional[Token]:
    """
    Optional authentication - returns Token if valid, None if no auth.

    Use this for endpoints that work differently based on auth status.
    """
    if not auth_manager.enabled:
        return None

    if not token:
        return None

    return auth_manager.validate_token(token)


async def require_auth(
    token: Optional[str] = Depends(_extract_token),
    auth_manager: AuthManager = Depends(get_auth_manager)
) -> Optional[Token]:
    """
    Require authentication if enabled.

    - If auth disabled: returns None (allows access)
    - If auth enabled but no token: raises 401
    - If auth enabled with invalid token: raises 401
    - If auth enabled with valid token: returns Token
    """
    if not auth_manager.enabled:
        return None

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )

    token_data = auth_manager.validate_token(token)
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )

    return token_data


async def require_admin(
    token_data: Optional[Token] = Depends(require_auth),
    auth_manager: AuthManager = Depends(get_auth_manager)
) -> Optional[Token]:
    """
    Require admin scope if authentication is enabled.

    - If auth disabled: returns None (allows access)
    - If auth enabled: requires valid token with ADMIN scope
    """
    if not auth_manager.enabled:
        return None

    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    if TokenScope.ADMIN not in token_data.scopes:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )

    return token_data


def require_scope(required_scope: TokenScope):
    """
    Factory for scope-checking dependencies.

    Usage:
        @router.get("/endpoint", dependencies=[Depends(require_scope(TokenScope.WRITE))])
    """
    async def check_scope(
        token_data: Optional[Token] = Depends(require_auth),
        auth_manager: AuthManager = Depends(get_auth_manager)
    ) -> Optional[Token]:
        if not auth_manager.enabled:
            return None

        if token_data is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )

        if required_scope not in token_data.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Scope '{required_scope.value}' required"
            )

        return token_data

    return check_scope
