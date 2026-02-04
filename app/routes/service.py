"""
Service endpoints for health checks, readiness, liveness, and metrics.

Kubernetes-compatible probes:
- /service/live  - Liveness probe (is the process alive?)
- /service/ready - Readiness probe (is the service ready for traffic?)
- /service/health - Combined health check
- /service/metrics - Prometheus metrics
"""

from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, Request, Response

from app.schemas.services import (
    HealthCheckResponse,
    LivenessResponse,
    ReadinessResponse,
    StatusResponse, StatusSchema,
    CPUStatusSchema, GPUStatusSchema, ModelStatusSchema
)
from app.core.utils import check_gpu_availability, get_device, get_n_workers
from app.core.settings import Settings
from app.core.database import ModelRepository


router = APIRouter()


# ===========================================
# Kubernetes Probes
# ===========================================

@router.get("/live", response_model=LivenessResponse)
async def liveness_probe():
    """
    Liveness probe - checks if the process is alive.

    Kubernetes uses this to know when to restart a container.
    Should be lightweight and always return quickly.

    Returns 200 if alive, container will be restarted if this fails.
    """
    return LivenessResponse(
        status="alive",
        timestamp=datetime.now()
    )


@router.get("/ready", response_model=ReadinessResponse)
async def readiness_probe(request: Request):
    """
    Readiness probe - checks if the service is ready to accept traffic.

    Kubernetes uses this to know when to send traffic to the pod.
    Checks critical dependencies (database, model manager, etc.)

    Returns 200 if ready, 503 if not ready.
    """
    checks: Dict[str, bool] = {}
    details: Dict[str, Any] = {}

    # Check 1: Model manager is initialized
    try:
        model_manager = getattr(request.app.state, 'model_manager', None)
        checks["model_manager"] = model_manager is not None
    except Exception as e:
        checks["model_manager"] = False
        details["model_manager_error"] = str(e)

    # Check 2: Database is accessible
    try:
        model_repo = ModelRepository()
        active_models = model_repo.get_active_codes()
        checks["database"] = True
        details["active_models_count"] = len(active_models)
    except Exception as e:
        checks["database"] = False
        details["database_error"] = str(e)

    # Check 3: Auth manager (if authentication is enabled)
    try:
        settings = Settings()
        if settings.auth_enabled:
            auth_manager = getattr(request.app.state, 'auth_manager', None)
            checks["auth_manager"] = auth_manager is not None
        else:
            checks["auth_manager"] = True  # Not required when auth disabled
    except Exception as e:
        checks["auth_manager"] = False
        details["auth_error"] = str(e)

    # Overall status
    is_ready = all(checks.values())

    response = ReadinessResponse(
        status="ready" if is_ready else "not_ready",
        timestamp=datetime.now(),
        checks=checks,
        details=details if details else None
    )

    if not is_ready:
        return Response(
            content=response.model_dump_json(),
            media_type="application/json",
            status_code=503
        )

    return response


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Simple health check endpoint that doesn't require authentication.

    Returns the current server status and time.
    Use /service/ready for Kubernetes readiness probes.
    Use /service/live for Kubernetes liveness probes.
    """
    settings = Settings()

    return HealthCheckResponse(
        service_name=settings.title_app,
        version=settings.version,
        status="healthy",
        timestamp=datetime.now(),
        details={"gpu_available": check_gpu_availability()[0]}
    )


# ===========================================
# Prometheus Metrics
# ===========================================

@router.get("/metrics")
async def prometheus_metrics(request: Request):
    """
    Prometheus metrics endpoint.

    Exposes metrics in Prometheus text format for scraping.
    """
    metrics_lines = []

    # Helper function to add metric
    def add_metric(name: str, value: float, help_text: str, metric_type: str = "gauge", labels: Dict[str, str] = None):
        metrics_lines.append(f"# HELP {name} {help_text}")
        metrics_lines.append(f"# TYPE {name} {metric_type}")
        if labels:
            label_str = ",".join([f'{k}="{v}"' for k, v in labels.items()])
            metrics_lines.append(f"{name}{{{label_str}}} {value}")
        else:
            metrics_lines.append(f"{name} {value}")

    # Service info
    settings = Settings()
    add_metric(
        "pyhellen_info",
        1,
        "PyHellen service information",
        labels={"version": settings.version, "service": settings.title_app}
    )

    # GPU availability
    gpu_available, gpu_info = check_gpu_availability()
    add_metric("pyhellen_gpu_available", 1 if gpu_available else 0, "Whether GPU is available")

    # Loaded models count
    try:
        model_manager = getattr(request.app.state, 'model_manager', None)
        if model_manager:
            loaded_count = len(model_manager.taggers)
            add_metric("pyhellen_models_loaded", loaded_count, "Number of models currently loaded")

            # Model manager metrics
            if hasattr(model_manager, '_metrics') and model_manager._metrics:
                metrics = model_manager._metrics
                add_metric(
                    "pyhellen_requests_total",
                    metrics.total_requests,
                    "Total number of requests processed",
                    metric_type="counter"
                )
                add_metric(
                    "pyhellen_errors_total",
                    metrics.total_errors,
                    "Total number of errors",
                    metric_type="counter"
                )
    except Exception:
        pass

    # Database models
    try:
        model_repo = ModelRepository()
        active_models = model_repo.get_active_codes()
        add_metric("pyhellen_models_available", len(active_models), "Number of models available")
    except Exception:
        pass

    # Request logging stats
    try:
        from app.core.database import RequestLogRepository
        request_repo = RequestLogRepository()
        stats = request_repo.get_statistics()
        if stats:
            add_metric(
                "pyhellen_requests_logged_total",
                stats.get("total_requests", 0),
                "Total logged requests",
                metric_type="counter"
            )
    except Exception:
        pass

    # CPU workers
    add_metric("pyhellen_cpu_workers", get_n_workers(), "Number of CPU workers available")

    return Response(
        content="\n".join(metrics_lines) + "\n",
        media_type="text/plain; charset=utf-8"
    )


# ===========================================
# Detailed Status
# ===========================================

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
