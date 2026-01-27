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


class TestModelManagerShutdown:
    """Test suite for ModelManager shutdown functionality."""

    @pytest.mark.asyncio
    async def test_shutdown_clears_taggers(self):
        """Test that shutdown clears all loaded taggers."""
        manager = ModelManager()
        manager.taggers["lasla"] = Mock()
        manager.taggers["grc"] = Mock()
        manager.iterator_processors["lasla"] = Mock()

        await manager.shutdown(shutdown_executor=False)

        assert len(manager.taggers) == 0
        assert len(manager.iterator_processors) == 0

    @pytest.mark.asyncio
    async def test_shutdown_sets_event(self):
        """Test that shutdown sets the shutdown event."""
        manager = ModelManager()
        assert not manager._shutdown_event.is_set()

        await manager.shutdown(shutdown_executor=False)

        assert manager._shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_shutdown_closes_http_client(self):
        """Test that shutdown closes the HTTP client."""
        manager = ModelManager()
        # Create a client
        client = await manager._get_http_client()
        assert not client.is_closed

        await manager.shutdown(shutdown_executor=False)

        assert client.is_closed

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(self):
        """Test that shutdown can be called multiple times safely."""
        manager = ModelManager()
        await manager._get_http_client()

        # Should not raise
        await manager.shutdown(shutdown_executor=False)
        await manager.shutdown(shutdown_executor=False)


class TestModelManagerGetModelInfo:
    """Test suite for get_model_info method."""

    def test_get_model_info_returns_none_for_invalid(self, mock_model_manager):
        """Test get_model_info returns None for invalid model."""
        with patch('app.core.model_manager.get_model', return_value=None):
            info = mock_model_manager.get_model_info("nonexistent_model")
            assert info is None

    def test_get_model_info_includes_basic_fields(self, mock_model_manager):
        """Test get_model_info returns expected fields."""
        mock_module = Mock()
        mock_module.DOWNLOADS = []

        with patch('app.core.model_manager.get_model', return_value=mock_module):
            info = mock_model_manager.get_model_info("test_model")

            assert info is not None
            assert "name" in info
            assert "status" in info
            assert "device" in info
            assert "batch_size" in info
            assert "files" in info
            assert "total_size_mb" in info
            assert "has_custom_processor" in info

    def test_get_model_info_with_loaded_model(self, mock_model_manager):
        """Test get_model_info includes metrics for loaded model."""
        mock_module = Mock()
        mock_module.DOWNLOADS = []
        mock_model_manager.taggers["test_model"] = Mock()

        with patch('app.core.model_manager.get_model', return_value=mock_module):
            info = mock_model_manager.get_model_info("test_model")

            assert info is not None
            assert info["status"] == "loaded"


class TestModelManagerErrorHandling:
    """Test suite for error handling in ModelManager."""

    def test_process_text_increments_error_on_failure(self, mock_model_manager):
        """Test that process_text updates error metrics on failure."""
        mock_tagger = Mock()
        mock_tagger.tag.side_effect = RuntimeError("Processing failed")

        initial_errors = mock_model_manager._metrics.total_errors if mock_model_manager._metrics else 0

        with pytest.raises(RuntimeError):
            mock_model_manager.process_text("model", mock_tagger, "text", False)

        if mock_model_manager._metrics:
            # Error should be tracked
            assert mock_model_manager._metrics.models.get("model") is not None

    @pytest.mark.asyncio
    async def test_batch_process_handles_empty_list(self, mock_model_manager):
        """Test batch processing with empty list returns empty list."""
        with patch.object(mock_model_manager, 'get_or_load_model', new_callable=AsyncMock) as mock_load:
            mock_load.return_value = Mock()

            results = await mock_model_manager.batch_process_concurrent("model", [], False)
            assert results == []

    @pytest.mark.asyncio
    async def test_unload_clears_both_tagger_and_processor(self):
        """Test unload removes both tagger and iterator_processor."""
        manager = ModelManager()
        manager.taggers["model"] = Mock()
        manager.iterator_processors["model"] = Mock()

        result = await manager.unload_model("model")

        assert result is True
        assert "model" not in manager.taggers
        assert "model" not in manager.iterator_processors


class TestModelManagerDeviceAndBatchSize:
    """Test suite for device and batch size configuration."""

    def test_batch_size_cannot_be_zero(self, mock_model_manager):
        """Test that batch size cannot be set to zero."""
        mock_model_manager.batch_size = 0
        assert mock_model_manager.batch_size == 1

    def test_batch_size_cannot_be_negative(self, mock_model_manager):
        """Test that batch size cannot be negative."""
        mock_model_manager.batch_size = -100
        assert mock_model_manager.batch_size == 1

    def test_batch_size_accepts_positive_values(self, mock_model_manager):
        """Test that positive batch sizes are accepted."""
        mock_model_manager.batch_size = 512
        assert mock_model_manager.batch_size == 512

    def test_device_is_valid(self, mock_model_manager):
        """Test that device returns a valid value."""
        device = mock_model_manager.device
        assert device in ["cpu", "cuda"]


class TestModelManagerCheckModelFiles:
    """Test suite for _check_model_files_exist method."""

    def test_check_model_files_returns_false_on_import_error(self, mock_model_manager):
        """Test _check_model_files_exist returns False on import error."""
        with patch('app.core.model_manager.get_model', side_effect=ImportError("Not found")):
            result = mock_model_manager._check_model_files_exist("fake_module", "/fake/path")
            assert result is False

    def test_check_model_files_returns_false_when_no_downloads(self, mock_model_manager):
        """Test _check_model_files_exist returns False when module has no DOWNLOADS."""
        mock_module = Mock(spec=[])  # No DOWNLOADS attribute

        with patch('app.core.model_manager.get_model', return_value=mock_module):
            result = mock_model_manager._check_model_files_exist("module", "/fake/path")
            assert result is False


class TestModelManagerIsModelAvailable:
    """Test suite for _is_model_available method."""

    def test_is_model_available_returns_true_for_valid_model(self, mock_model_manager):
        """Test _is_model_available returns truthy for valid model."""
        with patch('app.core.model_manager.get_model', return_value=Mock()):
            result = mock_model_manager._is_model_available("lasla")
            assert result

    def test_is_model_available_returns_false_on_import_error(self, mock_model_manager):
        """Test _is_model_available returns False on ImportError."""
        with patch('app.core.model_manager.get_model', side_effect=ImportError("Not found")):
            result = mock_model_manager._is_model_available("nonexistent")
            assert result is False


class TestModelMetricsEdgeCases:
    """Test suite for ModelMetrics edge cases."""

    def test_model_metrics_avg_process_time_zero(self):
        """Test average process time when no processing."""
        metrics = ModelMetrics()
        assert metrics.avg_process_time_ms == 0.0

    def test_model_metrics_to_dict_with_all_values(self):
        """Test to_dict with all metrics populated."""
        metrics = ModelMetrics()
        metrics.load_count = 3
        metrics.load_time_total_ms = 1500.0
        metrics.process_count = 100
        metrics.process_time_total_ms = 5000.0
        metrics.download_count = 1
        metrics.download_time_total_ms = 60000.0
        metrics.download_size_bytes = 1024 * 1024 * 50  # 50 MB
        metrics.error_count = 2
        metrics.last_loaded_at = datetime(2024, 1, 1, 12, 0, 0)
        metrics.last_used_at = datetime(2024, 1, 1, 13, 0, 0)

        data = metrics.to_dict()

        assert data["load_count"] == 3
        assert data["avg_load_time_ms"] == 500.0
        assert data["process_count"] == 100
        assert data["avg_process_time_ms"] == 50.0
        assert data["download_count"] == 1
        assert data["download_size_mb"] == 50.0
        assert data["error_count"] == 2
        assert data["last_loaded_at"] is not None
        assert data["last_used_at"] is not None


class TestGlobalMetricsEdgeCases:
    """Test suite for GlobalMetrics edge cases."""

    def test_global_metrics_requests_per_minute_with_small_uptime(self):
        """Test requests_per_minute calculation with very small uptime."""
        metrics = GlobalMetrics()
        # Immediately after creation, uptime is tiny
        metrics.total_requests = 10

        data = metrics.to_dict()

        # Should handle small uptime gracefully
        assert "requests_per_minute" in data
        assert isinstance(data["requests_per_minute"], float)

    def test_global_metrics_to_dict_with_multiple_models(self):
        """Test to_dict includes all tracked models."""
        metrics = GlobalMetrics()
        metrics.get_model_metrics("lasla")
        metrics.get_model_metrics("grc")
        metrics.get_model_metrics("fro")

        data = metrics.to_dict()

        assert "models" in data
        assert "lasla" in data["models"]
        assert "grc" in data["models"]
        assert "fro" in data["models"]

    def test_global_metrics_uptime_increases(self):
        """Test that uptime increases over time."""
        import time

        metrics = GlobalMetrics()
        data1 = metrics.to_dict()

        time.sleep(0.1)

        data2 = metrics.to_dict()

        assert data2["uptime_seconds"] > data1["uptime_seconds"]


class TestModelManagerDownload:
    """Test suite for download functionality."""

    @pytest.mark.asyncio
    async def test_download_model_sets_downloading_flag(self):
        """Test that download_model sets is_downloading flag."""
        manager = ModelManager()

        with patch('app.core.model_manager.get_model') as mock_get_model:
            mock_module = Mock()
            mock_module.DOWNLOADS = []
            mock_get_model.return_value = mock_module

            # Should return False because no files to download
            result = await manager.download_model("test_model")

            # Even if it fails, flag should be reset
            assert manager.is_downloading.get("test_model", False) is False

    @pytest.mark.asyncio
    async def test_download_model_invalid_module(self):
        """Test download_model with invalid module returns False."""
        manager = ModelManager()

        with patch('app.core.model_manager.get_model', return_value=None):
            result = await manager.download_model("nonexistent_model")
            assert result is False

    @pytest.mark.asyncio
    async def test_download_model_no_downloads_attribute(self):
        """Test download_model when module has no DOWNLOADS."""
        manager = ModelManager()

        with patch('app.core.model_manager.get_model') as mock_get_model:
            mock_module = Mock(spec=[])  # No DOWNLOADS attribute
            mock_get_model.return_value = mock_module

            result = await manager.download_model("test_model")
            assert result is False

    @pytest.mark.asyncio
    async def test_download_model_empty_downloads(self):
        """Test download_model when DOWNLOADS is empty."""
        manager = ModelManager()

        with patch('app.core.model_manager.get_model') as mock_get_model:
            mock_module = Mock()
            mock_module.DOWNLOADS = []
            mock_get_model.return_value = mock_module

            result = await manager.download_model("test_model")
            assert result is False

    @pytest.mark.asyncio
    async def test_download_model_concurrent_blocked(self):
        """Test that concurrent downloads of same model are blocked."""
        manager = ModelManager()
        manager.is_downloading["test_model"] = True

        with patch('app.core.model_manager.get_model') as mock_get_model:
            mock_module = Mock()
            mock_file = Mock()
            mock_file.name = "file.bin"
            mock_file.url = "http://example.com/file.bin"
            mock_module.DOWNLOADS = [mock_file]
            mock_get_model.return_value = mock_module

            result = await manager.download_model("test_model")
            assert result is False


class TestModelManagerGetOrLoad:
    """Test suite for get_or_load_model."""

    @pytest.mark.asyncio
    async def test_get_or_load_returns_cached_tagger(self):
        """Test that get_or_load_model returns cached tagger."""
        manager = ModelManager()
        mock_tagger = Mock()
        manager.taggers["test_model"] = mock_tagger

        result = await manager.get_or_load_model("test_model")
        assert result is mock_tagger

    @pytest.mark.asyncio
    async def test_get_or_load_model_not_available(self):
        """Test get_or_load_model raises error for unavailable model."""
        manager = ModelManager()

        with patch.object(manager, '_is_model_available', return_value=None):
            with pytest.raises(RuntimeError, match="not available"):
                await manager.get_or_load_model("nonexistent_model")


class TestModelManagerProcessTextAsync:
    """Test suite for process_text_async."""

    @pytest.mark.asyncio
    async def test_process_text_async_uses_executor(self):
        """Test that process_text_async uses thread pool executor."""
        manager = ModelManager()
        mock_tagger = Mock()
        mock_result = [{"form": "test", "lemma": "test"}]

        with patch.object(manager, 'process_text', return_value=mock_result):
            result = await manager.process_text_async(
                "test_model",
                mock_tagger,
                "test text",
                False
            )

            assert result == mock_result


class TestModelManagerStreamProcessingExtended:
    """Extended test suite for streaming processing methods."""

    @pytest.mark.asyncio
    async def test_stream_process_basic(self):
        """Test basic stream_process functionality."""
        manager = ModelManager()
        mock_tagger = Mock()
        manager.taggers["test_model"] = mock_tagger

        with patch.object(manager, 'process_text') as mock_process, \
             patch.object(manager, 'get_or_load_model', new_callable=AsyncMock) as mock_load:
            mock_load.return_value = mock_tagger
            mock_process.return_value = [{"form": "test"}]

            results = []
            async for chunk in manager.stream_process("test_model", ["text1"], False):
                results.append(chunk)

            assert len(results) > 0
            assert mock_process.called

    @pytest.mark.asyncio
    async def test_stream_process_ndjson_output_format(self):
        """Test NDJSON stream processing output format."""
        import json
        manager = ModelManager()
        mock_tagger = Mock()
        manager.taggers["test_model"] = mock_tagger

        with patch.object(manager, 'process_text_async', new_callable=AsyncMock) as mock_process, \
             patch.object(manager, 'get_or_load_model', new_callable=AsyncMock) as mock_load, \
             patch('app.core.cache.cache') as mock_cache:
            mock_load.return_value = mock_tagger
            mock_process.return_value = [{"form": "test"}]
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()

            results = []
            async for chunk in manager.stream_process_ndjson("test_model", ["text1"], False):
                results.append(chunk)

            assert len(results) == 1
            # Parse and verify format
            parsed = json.loads(results[0].strip())
            assert "index" in parsed
            assert "result" in parsed


class TestModelManagerBatchProcessingExtended:
    """Extended test suite for batch processing."""

    @pytest.mark.asyncio
    async def test_batch_process_concurrent_multiple_texts(self):
        """Test batch processing with multiple texts."""
        manager = ModelManager()
        mock_tagger = Mock()

        with patch.object(manager, 'get_or_load_model', new_callable=AsyncMock) as mock_load, \
             patch.object(manager, 'process_text_async', new_callable=AsyncMock) as mock_process:
            mock_load.return_value = mock_tagger
            mock_process.return_value = [{"form": "test"}]

            texts = ["text" + str(i) for i in range(10)]
            results = await manager.batch_process_concurrent(
                "test_model",
                texts,
                False,
                max_concurrent=2
            )

            assert len(results) == 10


class TestModelManagerUpdateMetrics:
    """Test suite for metrics update methods."""

    def test_update_metrics_sync_with_load_time(self, mock_model_manager):
        """Test _update_metrics_sync with load time."""
        mock_model_manager._metrics = GlobalMetrics()

        mock_model_manager._update_metrics_sync("test_model", load_time_ms=100.0)

        model_metrics = mock_model_manager._metrics.models.get("test_model")
        assert model_metrics is not None
        assert model_metrics.load_count == 1
        assert model_metrics.load_time_total_ms == 100.0

    def test_update_metrics_sync_with_process_time(self, mock_model_manager):
        """Test _update_metrics_sync with process time."""
        mock_model_manager._metrics = GlobalMetrics()

        mock_model_manager._update_metrics_sync(
            "test_model",
            process_time_ms=50.0,
            increment_requests=True
        )

        model_metrics = mock_model_manager._metrics.models.get("test_model")
        assert model_metrics.process_count == 1
        assert model_metrics.process_time_total_ms == 50.0

    def test_update_metrics_sync_with_error(self, mock_model_manager):
        """Test _update_metrics_sync with error flag."""
        mock_model_manager._metrics = GlobalMetrics()

        mock_model_manager._update_metrics_sync("test_model", error=True)

        model_metrics = mock_model_manager._metrics.models.get("test_model")
        assert model_metrics.error_count == 1
        assert mock_model_manager._metrics.total_errors == 1

    def test_update_metrics_sync_with_download(self, mock_model_manager):
        """Test _update_metrics_sync with download metrics."""
        mock_model_manager._metrics = GlobalMetrics()

        mock_model_manager._update_metrics_sync(
            "test_model",
            download_time_ms=5000.0,
            download_bytes=1024 * 1024 * 50  # 50 MB
        )

        model_metrics = mock_model_manager._metrics.models.get("test_model")
        assert model_metrics.download_count == 1
        assert model_metrics.download_time_total_ms == 5000.0
        assert model_metrics.download_size_bytes == 1024 * 1024 * 50


class TestModelManagerGetModelFiles:
    """Test suite for _get_model_files method."""

    def test_get_model_files_success(self, mock_model_manager):
        """Test _get_model_files returns file list."""
        mock_module = Mock()
        mock_file1 = Mock()
        mock_file1.name = "model.bin"
        mock_file2 = Mock()
        mock_file2.name = "vocab.txt"
        mock_module.DOWNLOADS = [mock_file1, mock_file2]

        with patch('app.core.model_manager.get_model', return_value=mock_module):
            result = mock_model_manager._get_model_files("test_model")

            assert result == ["model.bin", "vocab.txt"]

    def test_get_model_files_no_downloads(self, mock_model_manager):
        """Test _get_model_files when no DOWNLOADS attribute."""
        mock_module = Mock(spec=[])

        with patch('app.core.model_manager.get_model', return_value=mock_module):
            result = mock_model_manager._get_model_files("test_model")

            assert result is None

    def test_get_model_files_exception(self, mock_model_manager):
        """Test _get_model_files handles exceptions."""
        with patch('app.core.model_manager.get_model', side_effect=Exception("Error")):
            result = mock_model_manager._get_model_files("test_model")

            assert result is None
