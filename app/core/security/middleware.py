"""Security middleware for headers and error handling."""

from typing import List, Optional, Callable
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.logger import logger


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Adds security headers to all responses.

    Headers added:
    - X-Content-Type-Options: nosniff
    - X-Frame-Options: DENY
    - X-XSS-Protection: 1; mode=block
    - Referrer-Policy: strict-origin-when-cross-origin
    - Cache-Control: no-store (for API responses)
    - Strict-Transport-Security (if HTTPS)
    """

    async def dispatch(self, request: Request, call_next: Callable):
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

        # HSTS for HTTPS connections
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        # Prevent caching of API responses (except static files)
        if not request.url.path.startswith("/static"):
            response.headers["Cache-Control"] = "no-store, max-age=0"

        return response


def setup_exception_handlers(app: FastAPI) -> None:
    """
    Setup secure exception handlers that don't leak internal details.
    """

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions without leaking internal details."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail if exc.status_code < 500 else "Internal server error",
                "status_code": exc.status_code
            },
            headers=exc.headers
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        """
        Handle unexpected exceptions securely.

        Logs full error server-side but returns generic message to client.
        """
        # Log full error details server-side
        logger.error(
            f"Unhandled exception on {request.method} {request.url.path}: "
            f"{type(exc).__name__}: {str(exc)}",
            exc_info=True
        )

        # Return generic error to client
        return JSONResponse(
            status_code=500,
            content={
                "error": "An internal error occurred",
                "status_code": 500
            }
        )


def validate_model_name(model: str, allowed_models: List[str]) -> str:
    """
    Validate model name against whitelist.

    Args:
        model: The model name from request
        allowed_models: List of allowed model names

    Returns:
        The validated model name

    Raises:
        HTTPException: If model name is not in whitelist
    """
    # Normalize and validate
    model_clean = model.strip().lower()

    if model_clean not in allowed_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{model}'. Allowed models: {', '.join(allowed_models)}"
        )

    return model_clean


def get_allowed_models() -> List[str]:
    """Get list of allowed model names from PieLanguage enum."""
    from app.schemas.nlp import PieLanguage
    return [lang.name for lang in PieLanguage]
