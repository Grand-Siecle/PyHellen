import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware

from app.routes.api import router as api_router
from app.routes.service import router as service_router
from app.core.settings import Settings
from app.core.logger import logger
from app.core.model_manager import model_manager
from app.core.environment import PIE_EXTENDED_DOWNLOADS


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for the Hellen FastAPI application.
    """
    # Initialize at startup
    logger.info("🚀 Application is starting up...")
    logger.info(f"Using PIE_EXTENDED_DOWNLOADS: {PIE_EXTENDED_DOWNLOADS}")

    # Store model manager in app state for access from route handlers
    app.state.model_manager = model_manager

    # Backwards compatibility
    app.state.taggers_ml = {}
    app.state.download_locks = {}
    app.state.is_downloading = {}

    try:
        yield
    finally:
        # Cleanup at shutdown
        logger.info("🛑 Application is shutting down...")
        # Clear models from memory
        model_manager.models.clear()


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

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static files
    app.mount("/static", StaticFiles(directory="app/statics"), name="static")

    # Routers
    app.include_router(api_router, prefix="/api", tags=["API Natural Language Processing"])
    app.include_router(service_router, prefix="/service", tags=["Service Information"])

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
        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi

    return app


app = create_application()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ssl_keyfile=os.environ.get("SSL_KEYFILE", None),
        ssl_certfile=os.environ.get("SSL_CERTFILE", None)
    )