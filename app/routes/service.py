from datetime import datetime
from fastapi import APIRouter, Request

from app.schemas.services import (HealthCheckResponse,
                                  StatusResponse, StatusSchema,
                                  CPUStatusSchema, GPUStatusSchema, ModelStatusSchema)
from app.core.utils import check_gpu_availability, get_device, get_n_workers
from app.core.settings import Settings
from app.core.database import get_db_manager, ModelRepository


router = APIRouter()

@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Simple health check endpoint that doesn't require authentication.

    Returns the current server status and time.
    """
    settings = Settings()

    return {
        "service_name": settings.title_app,
        "version": settings.version,
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "details": {"gpu_available": check_gpu_availability()[0]}
    }


@router.get("/api/status", response_model=StatusResponse)
async def status_models(request: Request):
    """
    Return the status of all taggers and GPU information.

    This endpoint checks if each language model is loaded and provides details about GPU availability.
    """
    device = get_device()
    status = {}

    # Check GPU status
    gpu_available, gpu_info = check_gpu_availability()
    if gpu_available:
        status["gpu"] = GPUStatusSchema(
            available=True,
            device=gpu_info,
            in_use=(device == "cuda")
        )
    else:
        status["cpu"] = CPUStatusSchema(workers=get_n_workers())

    # Get models from database
    model_repo = ModelRepository()
    db_models = model_repo.get_all(include_inactive=False)

    models = {}
    for model in db_models:
        models[model.code] = ModelStatusSchema(
            language=model.name,
            status=request.app.state.taggers_ml.get(model.code, "not loaded")
        )

    status["models"] = models

    return {"status": StatusSchema(**status)}