"""
Tests for the ModelManager class.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from app.core.model_manager import ModelManager
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
