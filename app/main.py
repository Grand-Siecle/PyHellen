import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.utils import get_openapi


from .routes.api import router as api_router
from .settings import Settings

settings = Settings()

app = FastAPI(
    title=settings.title,
    description=settings.description,
    version=settings.version,
    openapi_url=settings.openapi_url,
    swagger_ui_parameters=settings.swagger_ui_parameters,
)

#
app.mount("/static", StaticFiles(directory="app/statics"), name="static")

app.include_router(api_router, prefix="/api", tags=["API Natural Language Processing"])


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=settings.title + " API",
        version=settings.version,
        summary=f"OpenAPI schema for {settings.title} API",
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