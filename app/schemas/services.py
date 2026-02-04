from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict, Any

from app.schemas.nlp import ModelStatusSchema


class HealthCheckResponse(BaseModel):
    """Basic health check response."""
    service_name: str
    version: Optional[str] = None
    status: Literal["healthy", "error", "maintenance"]
    timestamp: datetime
    details: Optional[dict] = None


class LivenessResponse(BaseModel):
    """Liveness probe response - is the process alive?"""
    status: Literal["alive", "dead"] = "alive"
    timestamp: datetime


class ReadinessResponse(BaseModel):
    """Readiness probe response - is the service ready to accept traffic?"""
    status: Literal["ready", "not_ready"]
    timestamp: datetime
    checks: Dict[str, bool] = Field(default_factory=dict)
    details: Optional[Dict[str, Any]] = None

class GPUStatusSchema(BaseModel):
    available: bool
    device: str
    in_use: bool

class CPUStatusSchema(BaseModel):
    workers: int = Field(gt=1)

class StatusSchema(BaseModel):
    gpu: GPUStatusSchema | None = None
    cpu: CPUStatusSchema | None = None
    models: dict[str, ModelStatusSchema] = Field(
        default=lambda: {
            "Model1": ModelStatusSchema(language="model1", status="loaded"),
            "model2": ModelStatusSchema(language="model2", status="not loaded"),
            "model3": ModelStatusSchema(language="model3", status="loading"),
        }
    )

class StatusResponse(BaseModel):
    status: StatusSchema
