import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware

from app.routes.api import router as api_router
from app.routes.service import router as service_router
from app.routes.admin import router as admin_router
from app.core.settings import Settings
from app.core.logger import logger
from app.core.model_manager import model_manager
from app.core.environment import PIE_EXTENDED_DOWNLOADS
from app.core.security import AuthManager
from app.core.security.middleware import SecurityHeadersMiddleware, setup_exception_handlers
from app.core.database import get_db_manager, ModelRepository
from app.core.middleware import RequestLoggerMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for the Hellen FastAPI application.

    Handles:
    - Startup: Initializes model manager, auth manager, preloads configured models
    - Shutdown: Gracefully shuts down model manager and clears resources
    """
    from app.core.settings import settings

    # Initialize at startup
    logger.info("Starting up PyHellen API...")
    logger.info(f"Using PIE_EXTENDED_DOWNLOADS: {PIE_EXTENDED_DOWNLOADS}")

    # Initialize database (creates tables and default models if needed)
    db_engine = get_db_manager()  # get_db_manager is aliased to get_db_engine
    model_repo = ModelRepository()
    active_models = model_repo.get_active_codes()
    logger.info(f"Database initialized. Active models: {active_models}")

    # Initialize authentication
    try:
        auth_manager = AuthManager.setup(
            enabled=settings.auth_enabled,
            secret_key=settings.secret_key,
            db_path=settings.token_db_path,
            auto_create_admin=settings.auto_create_admin_token
        )
        app.state.auth_manager = auth_manager

        if settings.auth_enabled:
            logger.info("Authentication: ENABLED")
        else:
            logger.warning("Authentication: DISABLED - API is publicly accessible")

    except ValueError as e:
        logger.error(f"Authentication setup failed: {e}")
        raise

    # Store model manager in app state for access from route handlers
    app.state.model_manager = model_manager

    # Backwards compatibility
    app.state.taggers_ml = {}
    app.state.download_locks = {}
    app.state.is_downloading = {}

    # Preload configured models (verify against database)
    if settings.preload_models:
        logger.info(f"Preloading models: {settings.preload_models}")
        for model_name in settings.preload_models:
            if model_name not in active_models:
                logger.warning(f"Skipping preload of '{model_name}': model not found or inactive in database")
                continue
            try:
                await model_manager.get_or_load_model(model_name)
                logger.info(f"Preloaded model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to preload model '{model_name}': {e}")

    try:
        yield
    finally:
        # Cleanup at shutdown
        logger.info("Shutting down PyHellen API...")
        # Clear models from memory (don't shutdown executor to avoid issues with pending requests)
        model_manager.taggers.clear()
        model_manager.iterator_processors.clear()
        # Close HTTP client if open
        if model_manager._http_client and not model_manager._http_client.is_closed:
            await model_manager._http_client.aclose()
        # Log final metrics
        if model_manager._metrics:
            logger.info(f"Final metrics: {model_manager._metrics.total_requests} requests, {model_manager._metrics.total_errors} errors")


def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    settings = Settings()

    app = FastAPI(
        title=settings.title_app,
        description=settings.description,
        version=settings.version,
        openapi_url=settings.openapi_url,
        swagger_ui_parameters=settings.swagger_ui_parameters,
        lifespan=lifespan,
    )

    # ===================
    # Middleware Stack (applied in reverse order)
    # ===================

    # Request logging (applied first = runs last, after response)
    app.add_middleware(RequestLoggerMiddleware, enabled=settings.enable_metrics)

    # Security headers (applied last = runs first)
    app.add_middleware(SecurityHeadersMiddleware)

    # CORS Configuration - Fixed for security
    # Note: allow_credentials=True with allow_origins=["*"] is invalid per CORS spec
    cors_origins = settings.cors_origins
    allow_credentials = settings.cors_allow_credentials

    # Security check: cannot use credentials with wildcard origin
    if "*" in cors_origins and allow_credentials:
        logger.warning(
            "CORS: Cannot use allow_credentials=True with wildcard origin. "
            "Setting allow_credentials=False for security."
        )
        allow_credentials = False

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=allow_credentials,
        allow_methods=["GET", "POST", "DELETE"],  # Only methods we actually use
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
        expose_headers=["X-Request-ID"],
    )

    # Setup secure exception handlers
    setup_exception_handlers(app)

    # Static files
    static_dir = "app/statics"
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # ===================
    # Routers
    # ===================
    app.include_router(api_router, prefix="/api", tags=["NLP Processing"])
    app.include_router(admin_router, prefix="/admin", tags=["Administration"])
    app.include_router(service_router, prefix="/service", tags=["Service"])

    # Custom OpenAPI schema
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema

        openapi_schema = get_openapi(
            title=settings.title_app + " API",
            version=settings.version,
            summary=f"OpenAPI schema for {settings.title_app} API",
            description=settings.description,
            routes=app.routes,
        )

        openapi_schema["info"]["x-logo"] = {
            "url": "/static/logo.png"
        }

        # Add security scheme if auth is enabled
        if settings.auth_enabled:
            openapi_schema["components"] = openapi_schema.get("components", {})
            openapi_schema["components"]["securitySchemes"] = {
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "Token",
                    "description": "API token obtained from admin endpoints"
                },
                "ApiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key",
                    "description": "API key in X-API-Key header"
                }
            }
            openapi_schema["security"] = [
                {"BearerAuth": []},
                {"ApiKeyAuth": []}
            ]

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi

    return app


app = create_application()

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("  DEVELOPMENT SERVER ONLY - NOT FOR PRODUCTION USE")
    print("  For production, use:")
    print("    gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker")
    print("  Or run with Docker (see docs/docker.md)")
    print("=" * 60 + "\n")

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
