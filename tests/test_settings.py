"""
Tests for application settings and configuration.
"""

import pytest
from pydantic import ValidationError


class TestSettingsValidation:
    """Test suite for Settings validation."""

    def test_default_settings(self):
        """Test that default settings are valid."""
        from app.core.settings import Settings

        settings = Settings()

        assert settings.title_app == "Hellen"
        assert settings.auth_enabled is False
        assert settings.batch_size == 256
        assert settings.enable_metrics is True

    def test_version_semantic_validation_valid(self):
        """Test that valid semantic versions are accepted."""
        from app.core.settings import Settings

        settings = Settings(version="1.2.3")
        assert settings.version == "1.2.3"

        settings = Settings(version="0.0.1")
        assert settings.version == "0.0.1"

        settings = Settings(version="10.20.30")
        assert settings.version == "10.20.30"

    def test_version_semantic_validation_invalid(self):
        """Test that invalid semantic versions are rejected."""
        from app.core.settings import Settings

        with pytest.raises(ValidationError):
            Settings(version="invalid")

        with pytest.raises(ValidationError):
            Settings(version="1.2")

        with pytest.raises(ValidationError):
            Settings(version="v1.2.3")

    def test_cors_origins_string_parsing(self):
        """Test that CORS origins can be parsed from comma-separated string."""
        from app.core.settings import Settings

        # Test the validator directly - it handles comma-separated strings
        result = Settings.parse_cors_origins("http://a.com,http://b.com")
        assert result == ["http://a.com", "http://b.com"]

        # Test with spaces
        result = Settings.parse_cors_origins("http://a.com , http://b.com")
        assert result == ["http://a.com", "http://b.com"]

    def test_cors_origins_list(self):
        """Test that CORS origins list is preserved."""
        from app.core.settings import Settings

        origins = ["http://localhost:3000", "https://example.com"]
        assert Settings.parse_cors_origins(origins) == origins

    def test_preload_models_string_parsing(self):
        """Test that preload_models can be parsed from comma-separated string."""
        from app.core.settings import Settings

        result = Settings.parse_preload_models("lasla,grc,fro")
        assert result == ["lasla", "grc", "fro"]

    def test_preload_models_list(self):
        """Test that preload_models list is preserved."""
        from app.core.settings import Settings

        models = ["lasla", "grc"]
        assert Settings.parse_preload_models(models) == models

    def test_download_timeout_minimum(self):
        """Test that download timeout has a minimum value."""
        from app.core.settings import Settings

        with pytest.raises(ValidationError):
            Settings(download_timeout_seconds=10)  # Below minimum of 30

    def test_download_max_retries_minimum(self):
        """Test that download max retries has a minimum value."""
        from app.core.settings import Settings

        with pytest.raises(ValidationError):
            Settings(download_max_retries=0)  # Below minimum of 1

    def test_rate_limit_requests_minimum(self):
        """Test that rate limit requests has a minimum value."""
        from app.core.settings import Settings

        with pytest.raises(ValidationError):
            Settings(rate_limit_requests=0)  # Below minimum of 1

    def test_rate_limit_window_minimum(self):
        """Test that rate limit window has a minimum value."""
        from app.core.settings import Settings

        with pytest.raises(ValidationError):
            Settings(rate_limit_window_seconds=0)  # Below minimum of 1

    def test_max_concurrent_processing_minimum(self):
        """Test that max concurrent processing has a minimum value."""
        from app.core.settings import Settings

        with pytest.raises(ValidationError):
            Settings(max_concurrent_processing=0)  # Below minimum of 1

    def test_batch_size_minimum(self):
        """Test that batch size has a minimum value."""
        from app.core.settings import Settings

        with pytest.raises(ValidationError):
            Settings(batch_size=0)  # Below minimum of 1

    def test_secret_key_empty_allowed_when_auth_disabled(self):
        """Test that empty secret key is allowed when auth is disabled."""
        from app.core.settings import Settings

        settings = Settings(auth_enabled=False, secret_key="")
        assert settings.secret_key == ""

    def test_settings_extra_ignored(self):
        """Test that extra fields in env are ignored."""
        from app.core.settings import Settings

        # The extra="ignore" config should prevent errors
        settings = Settings()
        assert settings is not None


class TestSettingsDefaults:
    """Test suite for Settings default values."""

    def test_default_auth_disabled(self):
        """Test auth is disabled by default."""
        from app.core.settings import Settings

        settings = Settings()
        assert settings.auth_enabled is False

    def test_default_rate_limit_disabled(self):
        """Test rate limiting is disabled by default."""
        from app.core.settings import Settings

        settings = Settings()
        assert settings.rate_limit_enabled is False

    def test_default_preload_models_empty(self):
        """Test no models are preloaded by default."""
        from app.core.settings import Settings

        settings = Settings()
        assert settings.preload_models == []

    def test_default_cors_origins_wildcard(self):
        """Test CORS allows all origins by default."""
        from app.core.settings import Settings

        settings = Settings()
        assert "*" in settings.cors_origins

    def test_default_metrics_enabled(self):
        """Test metrics are enabled by default."""
        from app.core.settings import Settings

        settings = Settings()
        assert settings.enable_metrics is True


class TestSettingsTypes:
    """Test suite for Settings field types."""

    def test_download_timeout_is_int(self):
        """Test download timeout is an integer."""
        from app.core.settings import Settings

        settings = Settings()
        assert isinstance(settings.download_timeout_seconds, int)

    def test_batch_size_is_int(self):
        """Test batch size is an integer."""
        from app.core.settings import Settings

        settings = Settings()
        assert isinstance(settings.batch_size, int)

    def test_cors_origins_is_list(self):
        """Test CORS origins is a list."""
        from app.core.settings import Settings

        settings = Settings()
        assert isinstance(settings.cors_origins, list)

    def test_preload_models_is_list(self):
        """Test preload models is a list."""
        from app.core.settings import Settings

        settings = Settings()
        assert isinstance(settings.preload_models, list)
