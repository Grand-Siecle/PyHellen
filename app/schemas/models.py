"""Pydantic schemas for model management."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, computed_field


class ModelBase(BaseModel):
    """Base model schema."""
    code: str = Field(..., description="Unique model code (e.g., 'lasla', 'grc')")
    name: str = Field(..., description="Human-readable model name")
    description: Optional[str] = Field(None, description="Model description")


class ModelCreate(ModelBase):
    """Schema for creating a new model."""
    pie_module: str = Field(..., description="Pie Extended module name")
    batch_size: Optional[int] = Field(None, ge=1, description="Override batch size for this model")
    priority: int = Field(100, ge=0, description="Display priority (lower = first)")


class ModelUpdate(BaseModel):
    """Schema for updating a model."""
    name: Optional[str] = Field(None, description="Human-readable model name")
    description: Optional[str] = Field(None, description="Model description")
    pie_module: Optional[str] = Field(None, description="Pie Extended module name")
    batch_size: Optional[int] = Field(None, ge=1, description="Override batch size")
    priority: Optional[int] = Field(None, ge=0, description="Display priority")


class ModelInfo(ModelBase):
    """Schema for model information response."""
    id: int
    pie_module: str
    is_active: bool
    is_builtin: bool
    batch_size: Optional[int] = None
    priority: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ModelFileInfo(BaseModel):
    """Schema for model file information."""
    id: int
    filename: str
    url: str
    size_bytes: Optional[int] = None
    checksum: Optional[str] = None
    is_downloaded: bool
    downloaded_at: Optional[datetime] = None

    model_config = {"from_attributes": True}

    @computed_field
    @property
    def size_mb(self) -> Optional[float]:
        """File size in megabytes, computed from size_bytes."""
        if self.size_bytes is not None:
            return round(self.size_bytes / (1024 * 1024), 2)
        return None


class ModelDetailInfo(ModelInfo):
    """Schema for detailed model information including files and metrics."""
    files: List[ModelFileInfo] = Field(default_factory=list)
    total_size_mb: float = Field(0, description="Total size of downloaded files")
    status: str = Field("not loaded", description="Current model status")
    device: Optional[str] = Field(None, description="Device (cuda/cpu)")
    has_custom_processor: bool = Field(False, description="Has custom iterator/processor")
    metrics: Optional[dict] = Field(None, description="Model metrics")


class ModelListResponse(BaseModel):
    """Schema for listing models."""
    models: List[ModelInfo]
    total: int
    active: int


class ModelStatistics(BaseModel):
    """Schema for model statistics."""
    total: int
    active: int
    inactive: int
    builtin: int
    custom: int


class ModelFileCreate(BaseModel):
    """Schema for adding a model file."""
    filename: str = Field(..., description="File name")
    url: str = Field(..., description="Download URL")
    size_bytes: Optional[int] = Field(None, description="Expected file size")
    checksum: Optional[str] = Field(None, description="File checksum (MD5/SHA256)")
