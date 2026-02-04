"""Request logging middleware for analytics and debugging."""

import time
from typing import Callable, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logger import logger


class RequestLoggerMiddleware(BaseHTTPMiddleware):
    """
    Middleware that logs all API requests to the database.

    Captures:
    - Endpoint and method
    - Processing time
    - Status code
    - Model used (for NLP endpoints)
    - Token ID (if authenticated)
    - Client IP
    - Error messages (for failed requests)
    """

    # Endpoints to skip logging (health checks, static files, etc.)
    SKIP_PATHS = {"/service/health", "/docs", "/redoc", "/openapi.json", "/static"}

    # Endpoints that use models
    MODEL_ENDPOINTS = {"/api/tag/", "/api/batch/", "/api/stream/", "/api/models/"}

    def __init__(self, app, enabled: bool = True):
        super().__init__(app)
        self.enabled = enabled
        self._repo = None

    def _get_repo(self):
        """Lazy initialization of request log repository."""
        if self._repo is None:
            try:
                from app.core.database import RequestLogRepository
                self._repo = RequestLogRepository()
            except Exception as e:
                logger.warning(f"Could not initialize request logger: {e}")
                self.enabled = False
        return self._repo

    def _extract_model(self, path: str) -> Optional[str]:
        """Extract model name from path if present."""
        for endpoint in self.MODEL_ENDPOINTS:
            if path.startswith(endpoint):
                # Path format: /api/tag/{model} or /api/models/{model}
                parts = path.split("/")
                if len(parts) >= 4:
                    return parts[3]
        return None

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address, considering proxies."""
        # Check X-Forwarded-For header first (for proxied requests)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Take the first IP (client IP)
            return forwarded.split(",")[0].strip()

        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to direct client IP
        if request.client:
            return request.client.host

        return "unknown"

    def _get_token_id(self, request: Request) -> Optional[int]:
        """Get token ID from request state if authenticated."""
        # The auth middleware stores the token in request.state
        if hasattr(request.state, "token") and request.state.token:
            return request.state.token.id
        return None

    def _should_skip(self, path: str) -> bool:
        """Check if this path should be skipped."""
        if any(path.startswith(skip) for skip in self.SKIP_PATHS):
            return True
        return False

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log to database."""
        # Skip if disabled or path should be skipped
        if not self.enabled or self._should_skip(request.url.path):
            return await call_next(request)

        start_time = time.time()
        error_message = None
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response

        except Exception as e:
            error_message = str(e)
            raise

        finally:
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000

            # Extract info for logging
            path = request.url.path
            method = request.method
            model_code = self._extract_model(path)
            client_ip = self._get_client_ip(request)
            token_id = self._get_token_id(request)

            # Log to database (fire and forget)
            try:
                repo = self._get_repo()
                if repo:
                    repo.log(
                        endpoint=path,
                        method=method,
                        status_code=status_code,
                        model_code=model_code,
                        token_id=token_id,
                        processing_time_ms=round(processing_time_ms, 2),
                        error_message=error_message,
                        client_ip=client_ip
                    )
            except Exception as log_error:
                # Don't let logging errors affect the request
                logger.warning(f"Failed to log request: {log_error}")
