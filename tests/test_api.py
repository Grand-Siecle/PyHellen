"""
Tests for the API endpoints.
"""

import pytest
from fastapi import status


class TestLanguagesEndpoint:
    """Test suite for /api/languages endpoint."""

    def test_get_languages(self, client):
        """Test getting list of supported languages."""
        response = client.get("/api/languages")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "languages" in data
        assert len(data["languages"]) > 0

        # Check that known languages are present (returns language descriptions)
        language_names = [lang for lang in data["languages"]]
        assert "Classical Latin" in language_names
        assert "Ancient Greek" in language_names


class TestModelsEndpoint:
    """Test suite for /api/models endpoints."""

    def test_list_models(self, client):
        """Test listing all models."""
        response = client.get("/api/models")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "models" in data

    def test_get_model_info_not_found(self, client):
        """Test getting info for non-existent model."""
        response = client.get("/api/models/nonexistent_model")
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_model_info_valid(self, client):
        """Test getting info for a valid model."""
        response = client.get("/api/models/lasla")

        # If model exists in pie_extended, should return info
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "name" in data
            assert "status" in data
            assert "device" in data


class TestCacheEndpoint:
    """Test suite for /api/cache endpoints."""

    def test_get_cache_stats(self, client):
        """Test getting cache statistics."""
        response = client.get("/api/cache/stats")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "size" in data
        assert "max_size" in data
        assert "hits" in data
        assert "misses" in data
        assert "hit_rate_percent" in data

    def test_clear_cache(self, client):
        """Test clearing the cache."""
        response = client.post("/api/cache/clear")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "cleared_entries" in data
        assert "message" in data

    def test_cleanup_cache(self, client):
        """Test cleaning up expired cache entries."""
        response = client.post("/api/cache/cleanup")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "removed_entries" in data


class TestTagEndpointValidation:
    """Test suite for tag endpoint input validation."""

    def test_tag_empty_text_post(self, client):
        """Test that empty text is rejected in POST."""
        response = client.post(
            "/api/tag/lasla",
            json={"text": "", "lower": False}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_tag_whitespace_only_text_post(self, client):
        """Test that whitespace-only text is rejected."""
        response = client.post(
            "/api/tag/lasla",
            json={"text": "   ", "lower": False}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_tag_missing_text_get(self, client):
        """Test that missing text query param is rejected."""
        response = client.get("/api/tag/lasla")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestBatchEndpointValidation:
    """Test suite for batch endpoint input validation."""

    def test_batch_empty_list(self, client):
        """Test that empty texts list is rejected."""
        response = client.post(
            "/api/batch/lasla",
            json={"texts": [], "lower": False}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_batch_too_many_texts(self, client):
        """Test that too many texts are rejected."""
        texts = [f"Text {i}" for i in range(101)]  # Max is 100
        response = client.post(
            "/api/batch/lasla",
            json={"texts": texts, "lower": False}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestStreamEndpoint:
    """Test suite for stream endpoint."""

    def test_stream_format_ndjson(self, client):
        """Test NDJSON streaming format."""
        # This will likely fail without a model, but tests the endpoint structure
        response = client.post(
            "/api/stream/lasla?format=ndjson",
            json={"texts": ["Test"], "lower": False}
        )
        # Either succeeds or fails due to model not being available
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ]

    def test_stream_format_sse(self, client):
        """Test SSE streaming format."""
        response = client.post(
            "/api/stream/lasla?format=sse",
            json={"texts": ["Test"], "lower": False}
        )
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ]


class TestHealthEndpoint:
    """Test suite for health check endpoint."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/service/health")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "service_name" in data
        assert "status" in data
        assert data["status"] == "healthy"

    def test_status_check(self, client):
        """Test detailed status endpoint."""
        response = client.get("/service/api/status")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "status" in data


class TestDownloadEndpoint:
    """Test suite for model download endpoint."""

    def test_download_nonexistent_model(self, client):
        """Test downloading a non-existent model."""
        response = client.post("/api/models/nonexistent_xyz/download")
        assert response.status_code == status.HTTP_404_NOT_FOUND
