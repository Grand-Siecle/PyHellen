from datetime import datetime
from pydantic import BaseModel
from typing import Optional, Literal

class HealthCheckResponse(BaseModel):
    service_name: str
    version: Optional[str] = None
    status: Literal["healthy", "error", "maintenance"]
    timestamp: datetime
    details: Optional[dict] = None
