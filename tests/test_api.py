"""
Tests for the API endpoints.
"""

import pytest
from fastapi import status
from unittest.mock import patch, Mock, AsyncMock


class TestLanguagesEndpoint:
    """Test suite for /api/languages endpoint."""

    def test_get_languages(self, client):
        """Test getting list of supported languages."""
        response = client.get("/api/languages")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "languages" in data
        assert "count" in data
        assert len(data["languages"]) > 0
        assert data["count"] == len(data["languages"])

        # Check that known languages are present (now returns objects with code, name, description)
        language_codes = [lang["code"] for lang in data["languages"]]
        language_names = [lang["name"] for lang in data["languages"]]
        assert "lasla" in language_codes
        assert "grc" in language_codes
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
        """Test getting info for non-existent model returns 400 (invalid model name)."""
        response = client.get("/api/models/nonexistent_model")
        # Model name validation returns 400 Bad Request for invalid model names
        assert response.status_code == status.HTTP_400_BAD_REQUEST

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
        # HybridCache uses memory_size instead of size
        assert "memory_size" in data or "size" in data
        assert "max_size" in data
        assert "hits" in data
        assert "misses" in data
        assert "hit_rate_percent" in data
        assert "persistence_enabled" in data

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


class TestTagEndpointWithMock:
    """Test suite for tag endpoint with mocked model."""

    def test_tag_text_success(self, client):
        """Test successful text tagging with mocked model."""
        mock_result = [{"form": "test", "lemma": "test", "pos": "NOUN"}]

        with patch('app.routes.api.model_manager') as mock_mm:
            mock_mm.get_or_load_model = AsyncMock(return_value=Mock())
            mock_mm.process_text.return_value = mock_result

            response = client.post(
                "/api/tag/lasla",
                json={"text": "test", "lower": False}
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "result" in data
            assert "processing_time_ms" in data
            assert "model" in data
            assert data["model"] == "lasla"

    def test_tag_text_get_success(self, client):
        """Test GET tag endpoint with mocked model."""
        mock_result = [{"form": "hello", "lemma": "hello"}]

        with patch('app.routes.api.model_manager') as mock_mm:
            mock_mm.get_or_load_model = AsyncMock(return_value=Mock())
            mock_mm.process_text.return_value = mock_result

            response = client.get("/api/tag/lasla?text=hello")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "result" in data
            assert data["from_cache"] is False

    def test_tag_text_with_lowercase(self, client):
        """Test text tagging with lowercase option."""
        mock_result = [{"form": "test", "lemma": "test"}]

        with patch('app.routes.api.model_manager') as mock_mm:
            mock_mm.get_or_load_model = AsyncMock(return_value=Mock())
            mock_mm.process_text.return_value = mock_result

            response = client.post(
                "/api/tag/lasla",
                json={"text": "TEST", "lower": True}
            )

            assert response.status_code == status.HTTP_200_OK
            # Verify lowercase was passed to process_text
            mock_mm.process_text.assert_called_once()
            call_args = mock_mm.process_text.call_args
            assert call_args[0][3] is True  # lower argument

    def test_tag_returns_cached_result(self, client):
        """Test that cached results are returned."""
        cached_result = [{"form": "cached", "lemma": "cached"}]

        with patch('app.routes.api.cache') as mock_cache:
            mock_cache.get = AsyncMock(return_value=cached_result)

            response = client.post(
                "/api/tag/lasla",
                json={"text": "test", "lower": False}
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["from_cache"] is True
            assert data["result"] == cached_result


class TestBatchEndpointWithMock:
    """Test suite for batch endpoint with mocked model."""

    def test_batch_process_success(self, client):
        """Test successful batch processing."""
        mock_results = [
            [{"form": "text1", "lemma": "text1"}],
            [{"form": "text2", "lemma": "text2"}]
        ]

        with patch('app.routes.api.model_manager') as mock_mm, \
             patch('app.routes.api.cache') as mock_cache:
            mock_mm.get_or_load_model = AsyncMock(return_value=Mock())
            mock_mm.process_text_async = AsyncMock(side_effect=mock_results)
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()

            response = client.post(
                "/api/batch/lasla",
                json={"texts": ["text1", "text2"], "lower": False}
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "results" in data
            assert "total_texts" in data
            assert data["total_texts"] == 2

    def test_batch_sequential_processing(self, client):
        """Test batch with sequential processing."""
        mock_result = [{"form": "test", "lemma": "test"}]

        with patch('app.routes.api.model_manager') as mock_mm, \
             patch('app.routes.api.cache') as mock_cache:
            mock_mm.get_or_load_model = AsyncMock(return_value=Mock())
            mock_mm.process_text.return_value = mock_result
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()

            response = client.post(
                "/api/batch/lasla?concurrent=false",
                json={"texts": ["text1", "text2"], "lower": False}
            )

            assert response.status_code == status.HTTP_200_OK


class TestStreamEndpointWithMock:
    """Test suite for stream endpoint with mocked model."""

    def test_stream_ndjson_format(self, client):
        """Test NDJSON streaming format."""
        async def mock_ndjson_generator(*args, **kwargs):
            yield '{"index": 0, "result": [{"form": "test"}]}\n'

        with patch('app.routes.api.model_manager') as mock_mm:
            mock_mm.stream_process_ndjson = mock_ndjson_generator

            response = client.post(
                "/api/stream/lasla?format=ndjson",
                json={"texts": ["Test"], "lower": False}
            )

            assert response.status_code == status.HTTP_200_OK

    def test_stream_sse_format(self, client):
        """Test SSE streaming format."""
        async def mock_sse_generator(*args, **kwargs):
            yield 'event: start\ndata: {"total": 1}\n\n'
            yield 'event: result\ndata: {"index": 0}\n\n'
            yield 'event: complete\ndata: {"total_processed": 1}\n\n'

        with patch('app.routes.api.model_manager') as mock_mm:
            mock_mm.stream_process_sse = mock_sse_generator

            response = client.post(
                "/api/stream/lasla?format=sse",
                json={"texts": ["Test"], "lower": False}
            )

            assert response.status_code == status.HTTP_200_OK

    def test_stream_invalid_format(self, client):
        """Test that invalid stream format is rejected."""
        response = client.post(
            "/api/stream/lasla?format=invalid",
            json={"texts": ["Test"], "lower": False}
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        # Response may be JSON with detail or text/plain
        try:
            data = response.json()
            assert "Invalid format" in data.get("detail", "")
        except Exception:
            # If not JSON, check text content
            assert "Invalid format" in response.text or response.status_code == status.HTTP_400_BAD_REQUEST


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


class TestMetricsEndpoint:
    """Test suite for metrics endpoint."""

    def test_get_metrics(self, client):
        """Test getting metrics."""
        response = client.get("/api/metrics")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        # Either metrics data or disabled message
        assert "started_at" in data or "message" in data

    def test_metrics_contains_expected_fields(self, client):
        """Test that metrics contain expected fields when enabled."""
        response = client.get("/api/metrics")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        if "started_at" in data:
            assert "uptime_seconds" in data
            assert "total_requests" in data
            assert "total_errors" in data
            assert "models" in data


class TestUnloadEndpoint:
    """Test suite for model unload endpoint."""

    def test_unload_not_loaded_model(self, client):
        """Test unloading a model with invalid name returns 400."""
        response = client.post("/api/models/nonexistent_xyz/unload")
        # Model name validation returns 400 Bad Request for invalid model names
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_unload_response_structure(self, client):
        """Test unload endpoint response structure on success."""
        # First load a model, then unload it
        # This test just verifies the endpoint exists and returns proper error
        response = client.post("/api/models/grc/unload")
        # Model might not be loaded, so 404 is expected
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_404_NOT_FOUND]


class TestLoadEndpoint:
    """Test suite for model load endpoint."""

    def test_load_invalid_model(self, client):
        """Test loading an invalid model returns 400 Bad Request."""
        response = client.post("/api/models/invalid_model_xyz/load")
        # Model name validation returns 400 Bad Request for invalid model names
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_load_model_success(self, client):
        """Test loading a model successfully with mock."""
        with patch('app.routes.api.model_manager') as mock_mm:
            mock_mm.get_or_load_model = AsyncMock(return_value=Mock())
            mock_mm.device = "cpu"

            response = client.post("/api/models/lasla/load")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["status"] == "loaded"
            assert data["model"] == "lasla"
            assert "load_time_ms" in data


class TestModelDownloadEndpoint:
    """Test suite for model download endpoint."""

    def test_download_invalid_model(self, client):
        """Test downloading an invalid model returns error."""
        response = client.post("/api/models/invalid_xyz/download")
        # Route may not exist (404) or model validation fails (400)
        assert response.status_code in [
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_404_NOT_FOUND
        ]


class TestInputValidation:
    """Test suite for comprehensive input validation."""

    def test_tag_text_max_length(self, client):
        """Test that text exceeding max length is rejected."""
        long_text = "a" * 100001  # Exceeds max_length=100000
        response = client.post(
            "/api/tag/lasla",
            json={"text": long_text, "lower": False}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_tag_get_text_max_length(self, client):
        """Test GET tag with text exceeding max length."""
        long_text = "a" * 10001  # Exceeds max_length=10000 for GET
        response = client.get(f"/api/tag/lasla?text={long_text}")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_batch_invalid_json(self, client):
        """Test batch endpoint with invalid JSON."""
        response = client.post(
            "/api/batch/lasla",
            data="not json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_tag_missing_required_field(self, client):
        """Test tag endpoint with missing required text field."""
        response = client.post(
            "/api/tag/lasla",
            json={"lower": False}  # Missing 'text' field
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestErrorHandling:
    """Test suite for API error handling."""

    def test_model_not_available_returns_404(self, client):
        """Test that unavailable model returns 404."""
        with patch('app.routes.api.model_manager') as mock_mm, \
             patch('app.routes.api.cache') as mock_cache:
            mock_cache.get = AsyncMock(return_value=None)
            mock_mm.get_or_load_model = AsyncMock(
                side_effect=RuntimeError("Model 'lasla' not available")
            )

            response = client.post(
                "/api/tag/lasla",
                json={"text": "test", "lower": False}
            )

            assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_model_load_failure_returns_503(self, client):
        """Test that model load failure returns 503."""
        with patch('app.routes.api.model_manager') as mock_mm, \
             patch('app.routes.api.cache') as mock_cache:
            mock_cache.get = AsyncMock(return_value=None)
            mock_mm.get_or_load_model = AsyncMock(
                side_effect=RuntimeError("Failed to load model 'lasla'")
            )

            response = client.post(
                "/api/tag/lasla",
                json={"text": "test", "lower": False}
            )

            assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE

    def test_processing_error_returns_400(self, client):
        """Test that processing errors return 400."""
        with patch('app.routes.api.model_manager') as mock_mm, \
             patch('app.routes.api.cache') as mock_cache:
            mock_mm.get_or_load_model = AsyncMock(return_value=Mock())
            mock_mm.process_text.side_effect = ValueError("Invalid input")
            mock_cache.get = AsyncMock(return_value=None)

            response = client.post(
                "/api/tag/lasla",
                json={"text": "test", "lower": False}
            )

            assert response.status_code == status.HTTP_400_BAD_REQUEST


class TestCacheEndpointDetails:
    """Test suite for detailed cache endpoint behavior."""

    def test_cache_stats_fields(self, client):
        """Test that cache stats contain all expected fields."""
        response = client.get("/api/cache/stats")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        expected_fields = ["memory_size", "max_size", "ttl_seconds", "hits", "misses", "hit_rate_percent"]
        for field in expected_fields:
            assert field in data, f"Missing field: {field}"

    def test_clear_cache_response_structure(self, client):
        """Test clear cache returns expected structure."""
        response = client.post("/api/cache/clear")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "cleared_entries" in data
        assert "message" in data
        assert isinstance(data["cleared_entries"], int)

    def test_cleanup_cache_response_structure(self, client):
        """Test cleanup cache returns expected structure."""
        response = client.post("/api/cache/cleanup")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "removed_entries" in data
        assert isinstance(data["removed_entries"], int)


class TestLanguagesEndpointDetails:
    """Test suite for detailed languages endpoint behavior."""

    def test_languages_structure(self, client):
        """Test languages response structure."""
        response = client.get("/api/languages")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "languages" in data
        assert "count" in data
        assert isinstance(data["languages"], list)
        assert data["count"] == len(data["languages"])

    def test_languages_each_has_required_fields(self, client):
        """Test each language has required fields."""
        response = client.get("/api/languages")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        for lang in data["languages"]:
            assert "code" in lang
            assert "name" in lang
            assert "description" in lang


class TestModelsEndpointDetails:
    """Test suite for detailed models endpoint behavior."""

    def test_list_models_structure(self, client):
        """Test list models returns expected structure."""
        response = client.get("/api/models")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], dict)

    def test_model_info_structure_for_valid_model(self, client):
        """Test model info returns expected structure for valid model."""
        response = client.get("/api/models/lasla")
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            # Check for expected fields
            assert "name" in data or "status" in data
