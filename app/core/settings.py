import secrets
import semver
from typing import Dict, Any, List, Optional
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

    # ===================
    # Security Settings
    # ===================
    auth_enabled: bool = Field(
        default=False,
        description="Enable token-based authentication. When False, API is publicly accessible."
    )
    secret_key: str = Field(
        default="",
        description="Secret key for token hashing. REQUIRED if auth_enabled=True. "
                    "Generate with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
    )
    token_db_path: str = Field(
        default="tokens.db",
        description="Path to SQLite database for token storage"
    )
    auto_create_admin_token: bool = Field(
        default=True,
        description="Automatically create admin token on first run if no tokens exist"
    )

    # CORS Settings
    cors_origins: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins. Use ['*'] for development only!"
    )
    cors_allow_credentials: bool = Field(
        default=False,
        description="Allow credentials in CORS. Cannot be True if cors_origins contains '*'"
    )

    # Rate Limiting (optional)
    rate_limit_enabled: bool = Field(
        default=False,
        description="Enable rate limiting"
    )
    rate_limit_requests: int = Field(
        default=100,
        ge=1,
        description="Maximum requests per time window"
    )
    rate_limit_window_seconds: int = Field(
        default=60,
        ge=1,
        description="Rate limit time window in seconds"
    )

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

    @field_validator("secret_key", mode="before")
    @classmethod
    def generate_secret_if_empty(cls, v, info):
        """Generate a random secret key if not provided and auth is disabled."""
        if not v:
            # Will be validated later if auth is enabled
            return ""
        return v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string."""
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

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
