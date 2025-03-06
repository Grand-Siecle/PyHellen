from datetime import datetime
from fastapi import APIRouter

from app.models.services import HealthCheckResponse
from app.utils import check_gpu_availability
from app.settings import Settings


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