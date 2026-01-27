import semver
from typing import Dict, Any, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # App metadata
    title_app: str = "Hellen"
    description: str = "REST API for accessing various NLP taggers languages schemas from Pie Extended"
    version: str = Field("0.0.1", description="Semantic versioning: MAJOR.MINOR.PATCH")
    openapi_url: str = "/openapi.json"
    swagger_ui_parameters: Dict[str, Any] = {"syntaxHighlight": {"theme": "obsidian"}}

    # Model management
    preload_models: List[str] = Field(
        default_factory=list,
        description="Models to preload at startup (e.g., ['lasla', 'grc'])"
    )

    # Download settings
    download_timeout_seconds: int = Field(
        default=300,
        ge=30,
        description="Timeout for model downloads in seconds"
    )
    download_max_retries: int = Field(
        default=3,
        ge=1,
        description="Maximum retry attempts for failed downloads"
    )

    # Processing settings
    max_concurrent_processing: int = Field(
        default=10,
        ge=1,
        description="Maximum concurrent text processing tasks"
    )
    batch_size: int = Field(
        default=256,
        ge=1,
        description="Batch size for model processing"
    )

    # Metrics
    enable_metrics: bool = Field(
        default=True,
        description="Enable metrics collection"
    )

    @field_validator("version")
    def validate_version(cls, v):
        try:
            semver.VersionInfo.parse(v)
        except ValueError:
            raise ValueError(f"Invalid semantic version: {v}")
        return v

    @field_validator("preload_models", mode="before")
    def parse_preload_models(cls, v):
        if isinstance(v, str):
            return [m.strip() for m in v.split(",") if m.strip()]
        return v


settings = Settings()
