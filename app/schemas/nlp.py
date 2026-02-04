from enum import Enum
from typing import Optional, Literal, List, Dict, Any
from pydantic import BaseModel, Field


class PieLanguage(str, Enum):
    """Supported languages for PIE Extended tagging."""
    lasla = "Classical Latin"
    grc = "Ancient Greek"
    fro = "Old French"
    freem = "Early Modern French"
    fr = "Classical French"
    dum = "Old Dutch"
    occ_cont = "Occitan Contemporain"

    @classmethod
    def get_description(cls, name: str) -> str:
        """Get the description for a language code."""
        try:
            return cls[name].value
        except KeyError:
            return "Unknown"


class SupportedLanguages(BaseModel):
    """Response model for supported languages."""
    languages: List[PieLanguage]
    count: int = Field(default=0, description="Number of supported languages")

    def __init__(self, **data):
        super().__init__(**data)
        object.__setattr__(self, 'count', len(self.languages))


class ModelStatusSchema(BaseModel):
    """Status information for a single model."""
    language: str = Field(..., description="Human-readable language name")
    status: Literal["loaded", "loading", "not loaded", "downloading"] = Field(
        ..., description="Current model status"
    )
    files: Optional[List[str]] = Field(None, description="List of model files")
    message: Optional[str] = Field(None, description="Additional status message")


class ModelFileInfo(BaseModel):
    """Information about a model file."""
    name: str = Field(..., description="File name")
    url: str = Field(..., description="Download URL")
    size_mb: Optional[float] = Field(None, description="File size in MB")
    downloaded: bool = Field(False, description="Whether file is downloaded")


class ModelDetailSchema(BaseModel):
    """Detailed information about a model."""
    name: str = Field(..., description="Model name/code")
    language: str = Field(..., description="Human-readable language name")
    status: str = Field(..., description="Current status")
    device: str = Field(..., description="Device (cuda/cpu)")
    batch_size: int = Field(..., description="Configured batch size")
    files: List[ModelFileInfo] = Field(default_factory=list, description="Model files")
    total_size_mb: float = Field(0, description="Total size of downloaded files")
    has_custom_processor: bool = Field(False, description="Has custom iterator/processor")


class TagToken(BaseModel):
    """A single tagged token."""
    form: str = Field(..., description="Original token form")
    lemma: Optional[str] = Field(None, description="Lemma")
    pos: Optional[str] = Field(None, description="Part of speech")
    morph: Optional[str] = Field(None, description="Morphological features")


class TagResult(BaseModel):
    """Result of tagging operation."""
    tokens: List[Dict[str, Any]] = Field(..., description="Tagged tokens")
    text_preview: Optional[str] = Field(None, description="Preview of input text")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model: str = Field(..., description="Model used")
    from_cache: bool = Field(False, description="Whether result was from cache")


class BatchTagResult(BaseModel):
    """Result of batch tagging operation."""
    results: List[List[Dict[str, Any]]] = Field(..., description="Results for each text")
    total_texts: int = Field(..., description="Number of texts processed")
    processing_time_ms: float = Field(..., description="Total processing time")
    model: str = Field(..., description="Model used")
    cache_hits: int = Field(0, description="Number of cache hits")


class CacheStats(BaseModel):
    """Cache statistics."""
    size: int = Field(..., description="Current number of entries")
    max_size: int = Field(..., description="Maximum cache size")
    ttl_seconds: int = Field(..., description="Time-to-live in seconds")
    hits: int = Field(..., description="Total cache hits")
    misses: int = Field(..., description="Total cache misses")
    hit_rate_percent: float = Field(..., description="Cache hit rate percentage")