import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.utils import get_openapi

from .routes.api import router as api_router
from .routes.service import router as service_router
from app.core.settings import Settings
from app.core.logger import logger

# settings
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.taggers_ml = {}
    app.state.download_locks = {}
    app.state.is_downloading = {}
    logger.info("🚀 Application is starting up...")
    try:
        yield
    finally:
        app.state.taggers_ml.clear()
        logger.info("🛑 Application is shutting down...")

settings = Settings()

# App initialization
app = FastAPI(
    title=settings.title_app,
    description=settings.description,
    version=settings.version,
    openapi_url=settings.openapi_url,
    swagger_ui_parameters=settings.swagger_ui_parameters,
    lifespan=lifespan,
)

# MOUNT Front
app.mount("/static", StaticFiles(directory="app/statics"), name="static")

# ROUTERS
app.include_router(api_router, prefix="/api", tags=["API Natural Language Processing"])
app.include_router(service_router, prefix="/service", tags=["Service Informations"])


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ssl_keyfile=os.environ.get("SSL_KEYFILE", None),
        ssl_certfile=os.environ.get("SSL_CERTFILE", None)
    )