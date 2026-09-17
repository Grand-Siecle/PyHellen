import semver
from typing import Annotated, Dict, Any, List
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

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
        default=False, description="Enable token-based authentication. When False, API is publicly accessible."
    )
    secret_key: str = Field(
        default="",
        description="Secret key for token hashing. REQUIRED if auth_enabled=True. "
        'Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"',
    )
    token_db_path: str = Field(default="tokens.db", description="Path to SQLite database for token storage")
    auto_create_admin_token: bool = Field(
        default=True, description="Automatically create admin token on first run if no tokens exist"
    )

    # CORS Settings
    # NoDecode: values are comma-separated strings (see parse_cors_origins), not JSON
    cors_origins: Annotated[List[str], NoDecode] = Field(
        default=["*"], description="Allowed CORS origins. Use ['*'] for development only!"
    )
    cors_allow_credentials: bool = Field(
        default=False, description="Allow credentials in CORS. Cannot be True if cors_origins contains '*'"
    )

    # Rate Limiting (optional)
    rate_limit_enabled: bool = Field(default=False, description="Enable rate limiting")
    rate_limit_requests: int = Field(default=100, ge=1, description="Maximum requests per time window")
    rate_limit_window_seconds: int = Field(default=60, ge=1, description="Rate limit time window in seconds")

    # Model management
    preload_models: Annotated[List[str], NoDecode] = Field(
        default_factory=list, description="Models to preload at startup (e.g., ['lasla', 'grc'])"
    )

    # Download settings
    download_timeout_seconds: int = Field(default=300, ge=30, description="Timeout for model downloads in seconds")
    download_max_retries: int = Field(default=3, ge=1, description="Maximum retry attempts for failed downloads")

    # Processing settings
    max_concurrent_processing: int = Field(default=10, ge=1, description="Maximum concurrent text processing tasks")
    batch_size: int = Field(default=256, ge=1, description="Batch size for model processing")
    quantize_cpu: bool = Field(
        default=False,
        description="INT8-quantize models when running on CPU (~35% faster inference, slightly different "
        "annotations than float models). Ignored on CUDA, where quantized ops are unsupported.",
    )

    # Result cache
    cache_enabled: bool = Field(default=True, description="Cache tagging results (memory + SQLite)")
    cache_persist: bool = Field(default=True, description="Persist cached results to SQLite so they survive restarts")
    cache_ttl_seconds: int = Field(
        default=7 * 24 * 3600,
        ge=1,
        description="Lifetime of a cached result. Keys include model/library versions and inference settings, "
        "so a long TTL never serves outdated annotations.",
    )
    cache_memory_max_entries: int = Field(default=1000, ge=1, description="Maximum results kept in memory")
    cache_memory_max_bytes: int = Field(
        default=256 * 1024 * 1024, ge=1, description="Maximum JSON size of results kept in memory"
    )
    cache_db_max_entries: int = Field(default=20000, ge=1, description="Maximum results kept in SQLite")
    cache_db_max_bytes: int = Field(default=1024 * 1024 * 1024, ge=1, description="Maximum JSON size kept in SQLite")
    cache_max_entry_bytes: int = Field(
        default=1024 * 1024, ge=1, description="Results larger than this (JSON bytes) are not cached"
    )
    cache_cleanup_interval_seconds: int = Field(
        default=3600, ge=0, description="Interval of the background purge of expired entries (0 disables it)"
    )
    cache_store_text_preview: bool = Field(
        default=True, description="Store the first 100 characters of each text in SQLite for inspection"
    )

    # Metrics
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")

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
