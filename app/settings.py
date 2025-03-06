import semver
from typing import Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator

class Settings(BaseSettings):
    title_app: str = "Hellen"
    description: str = "REST API for accessing various NLP taggers languages models from Pie Extended"
    version: str = Field("0.0.1", description="Semantic versioning: MAJOR.MINOR.PATCH")
    openapi_url: str = "/openapi.json"
    swagger_ui_parameters: Dict[str, Any] = {"syntaxHighlight": {"theme": "obsidian"}}

    @field_validator("version")
    def validate_version(cls, v):
        try:
            semver.VersionInfo.parse(v)
        except ValueError:
            raise ValueError(f"Invalid semantic version: {v}")
        return v