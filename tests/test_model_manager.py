"""
Tests for the ModelManager class.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from app.core.model_manager import ModelManager, ModelMetrics, GlobalMetrics
from app.schemas.nlp import PieLanguage


class TestModelManagerInit:
    """Test suite for ModelManager initialization."""

    def test_initialization(self):
        """Test ModelManager initializes correctly."""
        manager = ModelManager()

        assert manager.taggers == {}
        assert manager.iterator_processors == {}
        assert manager.download_locks == {}
        assert manager.is_downloading == {}
        assert manager._executor is not None
        assert manager._processing_semaphore is not None

    def test_device_property(self, mock_model_manager):
        """Test device property returns cpu or cuda."""
        device = mock_model_manager.device
        assert device in ["cpu", "cuda"]

    def test_batch_size_default(self, mock_model_manager):
        """Test default batch size."""
        assert mock_model_manager.batch_size == 256

    def test_batch_size_setter(self, mock_model_manager):
        """Test batch size setter with validation."""
        mock_model_manager.batch_size = 128
        assert mock_model_manager.batch_size == 128

        # Test minimum value enforcement
        mock_model_manager.batch_size = -10
        assert mock_model_manager.batch_size == 1


class TestModelStatus:
    """Test suite for model status methods."""

    def test_get_model_status_not_loaded(self, mock_model_manager):
        """Test status for a model that isn't loaded."""
        with patch.object(mock_model_manager, '_is_model_available', return_value=True):
            status = mock_model_manager.get_model_status("lasla")
            assert status == "not loaded"

    def test_get_model_status_loaded(self, mock_model_manager):
        """Test status for a loaded model."""
        mock_model_manager.taggers["lasla"] = Mock()
        status = mock_model_manager.get_model_status("lasla")
        assert status == "loaded"

    def test_get_model_status_downloading(self, mock_model_manager):
        """Test status for a model being downloaded."""
        mock_model_manager.is_downloading["lasla"] = True
        status = mock_model_manager.get_model_status("lasla")
        assert status == "downloading"

    def test_get_model_status_not_available(self, mock_model_manager):
        """Test status for an unavailable model."""
        with patch.object(mock_model_manager, '_is_model_available', return_value=False):
            status = mock_model_manager.get_model_status("fake_model")
            assert status == "not available"

    def test_get_all_models_status(self, mock_model_manager):
        """Test getting status for all models."""
        with patch.object(mock_model_manager, '_is_model_available', return_value=True):
            with patch.object(mock_model_manager, '_get_model_files', return_value=["file1.pt"]):
                statuses = mock_model_manager.get_all_models_status()

                # Should have status for each PieLanguage
                assert len(statuses) == len(PieLanguage)

                for lang in PieLanguage:
                    assert lang.name in statuses
                    assert statuses[lang.name].language == lang.value


class TestModelInfo:
    """Test suite for model info method."""

    def test_get_model_info_nonexistent(self, mock_model_manager):
        """Test getting info for non-existent model."""
        with patch('app.core.model_manager.get_model', return_value=None):
            info = mock_model_manager.get_model_info("nonexistent")
            assert info is None


class TestProcessText:
    """Test suite for text processing methods."""

    def test_process_text_with_lower(self, mock_model_manager):
        """Test processing text with lowercase option."""
        mock_tagger = Mock()
        mock_tagger.tag.return_value = [{"form": "test", "lemma": "test"}]

        result = mock_model_manager.process_text(
            "model",
            mock_tagger,
            "TEST TEXT",
            lower=True
        )

        # Should be called with lowercased text
        mock_tagger.tag.assert_called_with("test text")

    def test_process_text_without_lower(self, mock_model_manager):
        """Test processing text without lowercase option."""
        mock_tagger = Mock()
        mock_tagger.tag.return_value = [{"form": "TEST", "lemma": "test"}]

        result = mock_model_manager.process_text(
            "model",
            mock_tagger,
            "TEST TEXT",
            lower=False
        )

        mock_tagger.tag.assert_called_with("TEST TEXT")

    def test_process_text_with_custom_processor(self, mock_model_manager):
        """Test processing with custom iterator/processor."""
        mock_tagger = Mock()
        mock_iterator = Mock()
        mock_processor = Mock()
        mock_tagger.tag_str.return_value = [{"form": "test"}]

        mock_model_manager.iterator_processors["model"] = lambda: (mock_iterator, mock_processor)

        result = mock_model_manager.process_text(
            "model",
            mock_tagger,
            "test text",
            lower=False
        )

        mock_tagger.tag_str.assert_called_once()


class TestBatchProcessing:
    """Test suite for batch processing methods."""

    @pytest.mark.asyncio
    async def test_batch_process_concurrent_empty_list(self, mock_model_manager):
        """Test batch processing with empty list."""
        with patch.object(mock_model_manager, 'get_or_load_model', new_callable=AsyncMock) as mock_load:
            mock_tagger = Mock()
            mock_load.return_value = mock_tagger

            results = await mock_model_manager.batch_process_concurrent("model", [], False)
            assert results == []


class TestStreamProcessing:
    """Test suite for stream processing methods."""

    @pytest.mark.asyncio
    async def test_stream_process_yields_results(self, mock_model_manager):
        """Test that stream_process yields results for each text."""
        mock_tagger = Mock()
        mock_tagger.tag.return_value = [{"form": "test"}]

        with patch.object(mock_model_manager, 'get_or_load_model', new_callable=AsyncMock) as mock_load:
            mock_load.return_value = mock_tagger

            results = []
            async for result in mock_model_manager.stream_process("model", ["text1", "text2"], False):
                results.append(result)

            assert len(results) == 2

    @pytest.mark.asyncio
    async def test_stream_process_ndjson_format(self, mock_model_manager):
        """Test NDJSON streaming format."""
        import json

        mock_tagger = Mock()
        mock_tagger.tag.return_value = [{"form": "test"}]

        with patch.object(mock_model_manager, 'get_or_load_model', new_callable=AsyncMock) as mock_load:
            mock_load.return_value = mock_tagger

            with patch.object(mock_model_manager, 'process_text_async', new_callable=AsyncMock) as mock_process:
                mock_process.return_value = [{"form": "test"}]

                results = []
                async for result in mock_model_manager.stream_process_ndjson("model", ["text1"], False):
                    results.append(result)

                assert len(results) == 1

                # Parse NDJSON line
                parsed = json.loads(results[0].strip())
                assert "index" in parsed
                assert "result" in parsed
                assert "processing_time_ms" in parsed

    @pytest.mark.asyncio
    async def test_stream_process_sse_format(self, mock_model_manager):
        """Test SSE streaming format."""
        mock_tagger = Mock()
        mock_tagger.tag.return_value = [{"form": "test"}]

        with patch.object(mock_model_manager, 'get_or_load_model', new_callable=AsyncMock) as mock_load:
            mock_load.return_value = mock_tagger

            with patch.object(mock_model_manager, 'process_text_async', new_callable=AsyncMock) as mock_process:
                mock_process.return_value = [{"form": "test"}]

                results = []
                async for result in mock_model_manager.stream_process_sse("model", ["text1"], False):
                    results.append(result)

                # Should have start, result, and complete events
                assert len(results) == 3
                assert "event: start" in results[0]
                assert "event: result" in results[1]
                assert "event: complete" in results[2]


class TestModelMetrics:
    """Test suite for ModelMetrics dataclass."""

    def test_model_metrics_defaults(self):
        """Test ModelMetrics initializes with default values."""
        metrics = ModelMetrics()
        assert metrics.load_count == 0
        assert metrics.process_count == 0
        assert metrics.error_count == 0
        assert metrics.last_loaded_at is None

    def test_model_metrics_avg_load_time(self):
        """Test average load time calculation."""
        metrics = ModelMetrics()
        metrics.load_count = 2
        metrics.load_time_total_ms = 1000.0
        assert metrics.avg_load_time_ms == 500.0

    def test_model_metrics_avg_load_time_zero(self):
        """Test average load time when no loads."""
        metrics = ModelMetrics()
        assert metrics.avg_load_time_ms == 0.0

    def test_model_metrics_avg_process_time(self):
        """Test average process time calculation."""
        metrics = ModelMetrics()
        metrics.process_count = 4
        metrics.process_time_total_ms = 200.0
        assert metrics.avg_process_time_ms == 50.0

    def test_model_metrics_to_dict(self):
        """Test ModelMetrics serialization to dict."""
        metrics = ModelMetrics()
        metrics.load_count = 1
        metrics.process_count = 10
        metrics.last_loaded_at = datetime(2024, 1, 1, 12, 0, 0)

        data = metrics.to_dict()
        assert "load_count" in data
        assert "avg_load_time_ms" in data
        assert "process_count" in data
        assert "last_loaded_at" in data
        assert data["load_count"] == 1
        assert data["process_count"] == 10


class TestGlobalMetrics:
    """Test suite for GlobalMetrics dataclass."""

    def test_global_metrics_defaults(self):
        """Test GlobalMetrics initializes with default values."""
        metrics = GlobalMetrics()
        assert metrics.total_requests == 0
        assert metrics.total_errors == 0
        assert metrics.models == {}
        assert metrics.started_at is not None

    def test_global_metrics_get_model_metrics(self):
        """Test getting model metrics creates entry if not exists."""
        metrics = GlobalMetrics()
        model_metrics = metrics.get_model_metrics("lasla")

        assert "lasla" in metrics.models
        assert isinstance(model_metrics, ModelMetrics)

    def test_global_metrics_get_model_metrics_existing(self):
        """Test getting existing model metrics returns same instance."""
        metrics = GlobalMetrics()
        metrics.models["lasla"] = ModelMetrics()
        metrics.models["lasla"].load_count = 5

        model_metrics = metrics.get_model_metrics("lasla")
        assert model_metrics.load_count == 5

    def test_global_metrics_to_dict(self):
        """Test GlobalMetrics serialization to dict."""
        metrics = GlobalMetrics()
        metrics.total_requests = 100
        metrics.total_errors = 5
        metrics.get_model_metrics("lasla")

        data = metrics.to_dict()
        assert "started_at" in data
        assert "uptime_seconds" in data
        assert "total_requests" in data
        assert "total_errors" in data
        assert "requests_per_minute" in data
        assert "models" in data
        assert "lasla" in data["models"]


class TestModelManagerMetrics:
    """Test suite for ModelManager metrics integration."""

    def test_metrics_property_enabled(self):
        """Test metrics property when enabled."""
        manager = ModelManager()
        metrics = manager.metrics

        assert metrics is not None
        assert "started_at" in metrics
        assert "total_requests" in metrics

    def test_metrics_property_disabled(self):
        """Test metrics property when disabled."""
        with patch('app.core.model_manager.settings') as mock_settings:
            mock_settings.enable_metrics = False
            mock_settings.max_concurrent_processing = 10
            mock_settings.batch_size = 256

            manager = ModelManager()
            assert manager.metrics is None

    def test_process_text_updates_metrics(self):
        """Test that process_text updates metrics."""
        manager = ModelManager()
        mock_tagger = Mock()
        mock_tagger.tag.return_value = [{"form": "test"}]

        initial_count = manager._metrics.total_requests

        manager.process_text("test_model", mock_tagger, "test text", False)

        assert manager._metrics.total_requests == initial_count + 1
        assert "test_model" in manager._metrics.models
        assert manager._metrics.models["test_model"].process_count == 1


class TestModelManagerUnload:
    """Test suite for model unload functionality."""

    @pytest.mark.asyncio
    async def test_unload_model_success(self):
        """Test successfully unloading a model."""
        manager = ModelManager()
        manager.taggers["lasla"] = Mock()
        manager.iterator_processors["lasla"] = Mock()

        result = await manager.unload_model("lasla")

        assert result is True
        assert "lasla" not in manager.taggers
        assert "lasla" not in manager.iterator_processors

    @pytest.mark.asyncio
    async def test_unload_model_not_loaded(self):
        """Test unloading a model that isn't loaded."""
        manager = ModelManager()

        result = await manager.unload_model("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_unload_model_only_tagger(self):
        """Test unloading model with only tagger (no processor)."""
        manager = ModelManager()
        manager.taggers["lasla"] = Mock()

        result = await manager.unload_model("lasla")

        assert result is True
        assert "lasla" not in manager.taggers


class TestModelManagerHttpClient:
    """Test suite for HTTP client management."""

    @pytest.mark.asyncio
    async def test_get_http_client_creates_client(self):
        """Test that _get_http_client creates a client."""
        manager = ModelManager()
        assert manager._http_client is None

        client = await manager._get_http_client()

        assert client is not None
        assert manager._http_client is not None

        # Cleanup
        await client.aclose()

    @pytest.mark.asyncio
    async def test_get_http_client_reuses_client(self):
        """Test that _get_http_client reuses existing client."""
        manager = ModelManager()
        client1 = await manager._get_http_client()
        client2 = await manager._get_http_client()

        assert client1 is client2

        # Cleanup
        await client1.aclose()
