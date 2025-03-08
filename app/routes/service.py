from datetime import datetime
from fastapi import APIRouter

from app.models.services import HealthCheckResponse
from app.models.nlp import PieLanguage
from app.utils import check_gpu_availability, get_device, get_n_workers
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


@router.get("/api/status")
async def get_status():
    """
    Return the status of all taggers and GPU information.

    This endpoint checks if each language model is loaded and provides details about GPU availability.
    """

    #TODO maybe cache
    device = get_device()
    status = {}
    languages = [lang.value for lang in PieLanguage]

    # Check GPU status
    gpu_available, gpu_info = check_gpu_availability()
    if gpu_available:
        status["gpu"] = {
            "available": gpu_available,
            "device": gpu_info,
            "in_use": device == "cuda"
            }
    else:
        status["cpu"] = {
            "workers": get_n_workers() - 1,
        }

    # Check tagger status
    for language in languages:
        if language in taggers:
            status[language] = "loaded"
        else:
            # Try to initialize the tagger
            tagger = initialize_tagger(language)
            status[language] = "loaded" if tagger else "not loaded"

    return {"status": status}