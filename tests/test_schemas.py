"""
Tests for Pydantic schemas and validation.
"""

import pytest
from pydantic import ValidationError

from app.schemas.nlp import (
    PieLanguage,
    SupportedLanguages,
    ModelStatusSchema,
    ModelFileInfo,
    ModelDetailSchema,
    TagToken,
    TagResult,
    BatchTagResult,
    CacheStats,
)


class TestPieLanguage:
    """Test suite for PieLanguage enum."""

    def test_all_languages_defined(self):
        """Test that all expected languages are defined."""
        expected = ["lasla", "grc", "fro", "freem", "fr", "dum", "occ_cont"]
        actual = [lang.name for lang in PieLanguage]
        assert sorted(actual) == sorted(expected)

    def test_language_values(self):
        """Test that language values are human-readable names."""
        assert PieLanguage.lasla.value == "Classical Latin"
        assert PieLanguage.grc.value == "Ancient Greek"
        assert PieLanguage.fro.value == "Old French"

    def test_get_description_valid(self):
        """Test get_description returns correct value for valid language."""
        assert PieLanguage.get_description("lasla") == "Classical Latin"
        assert PieLanguage.get_description("grc") == "Ancient Greek"

    def test_get_description_invalid(self):
        """Test get_description returns 'Unknown' for invalid language."""
        assert PieLanguage.get_description("nonexistent") == "Unknown"


class TestSupportedLanguages:
    """Test suite for SupportedLanguages schema."""

    def test_count_auto_calculated(self):
        """Test that count is automatically calculated from languages list."""
        langs = SupportedLanguages(languages=[PieLanguage.lasla, PieLanguage.grc])
        assert langs.count == 2

    def test_empty_languages(self):
        """Test with empty languages list."""
        langs = SupportedLanguages(languages=[])
        assert langs.count == 0

    def test_all_languages(self):
        """Test with all available languages."""
        langs = SupportedLanguages(languages=list(PieLanguage))
        assert langs.count == len(PieLanguage)


class TestModelStatusSchema:
    """Test suite for ModelStatusSchema."""

    def test_valid_status_loaded(self):
        """Test creating schema with loaded status."""
        status = ModelStatusSchema(language="Classical Latin", status="loaded")
        assert status.language == "Classical Latin"
        assert status.status == "loaded"
        assert status.files is None
        assert status.message is None

    def test_valid_status_not_loaded(self):
        """Test creating schema with not loaded status."""
        status = ModelStatusSchema(
            language="Classical Latin",
            status="not loaded",
            files=["model.pt", "vocab.txt"]
        )
        assert status.status == "not loaded"
        assert status.files == ["model.pt", "vocab.txt"]

    def test_valid_status_downloading(self):
        """Test creating schema with downloading status."""
        status = ModelStatusSchema(
            language="Classical Latin",
            status="downloading",
            message="50% complete"
        )
        assert status.status == "downloading"
        assert status.message == "50% complete"

    def test_invalid_status_rejected(self):
        """Test that invalid status values are rejected."""
        with pytest.raises(ValidationError):
            ModelStatusSchema(language="Test", status="invalid_status")


class TestModelFileInfo:
    """Test suite for ModelFileInfo schema."""

    def test_minimal_creation(self):
        """Test creating with minimal required fields."""
        info = ModelFileInfo(name="model.pt", url="https://example.com/model.pt")
        assert info.name == "model.pt"
        assert info.url == "https://example.com/model.pt"
        assert info.size_mb is None
        assert info.downloaded is False

    def test_full_creation(self):
        """Test creating with all fields."""
        info = ModelFileInfo(
            name="model.pt",
            url="https://example.com/model.pt",
            size_mb=150.5,
            downloaded=True
        )
        assert info.size_mb == 150.5
        assert info.downloaded is True


class TestModelDetailSchema:
    """Test suite for ModelDetailSchema."""

    def test_creation_with_defaults(self):
        """Test creating with default values."""
        detail = ModelDetailSchema(
            name="lasla",
            language="Classical Latin",
            status="loaded",
            device="cpu",
            batch_size=256
        )
        assert detail.files == []
        assert detail.total_size_mb == 0
        assert detail.has_custom_processor is False

    def test_creation_with_files(self):
        """Test creating with file information."""
        files = [
            ModelFileInfo(name="model.pt", url="https://example.com/model.pt", downloaded=True)
        ]
        detail = ModelDetailSchema(
            name="lasla",
            language="Classical Latin",
            status="loaded",
            device="cuda",
            batch_size=512,
            files=files,
            total_size_mb=150.5,
            has_custom_processor=True
        )
        assert len(detail.files) == 1
        assert detail.total_size_mb == 150.5
        assert detail.has_custom_processor is True


class TestTagToken:
    """Test suite for TagToken schema."""

    def test_minimal_creation(self):
        """Test creating with only required form field."""
        token = TagToken(form="word")
        assert token.form == "word"
        assert token.lemma is None
        assert token.pos is None
        assert token.morph is None

    def test_full_creation(self):
        """Test creating with all fields."""
        token = TagToken(
            form="running",
            lemma="run",
            pos="VERB",
            morph="VerbForm=Ger"
        )
        assert token.form == "running"
        assert token.lemma == "run"
        assert token.pos == "VERB"
        assert token.morph == "VerbForm=Ger"


class TestTagResult:
    """Test suite for TagResult schema."""

    def test_creation(self):
        """Test creating tag result."""
        result = TagResult(
            tokens=[{"form": "test", "lemma": "test"}],
            processing_time_ms=10.5,
            model="lasla"
        )
        assert result.tokens == [{"form": "test", "lemma": "test"}]
        assert result.processing_time_ms == 10.5
        assert result.model == "lasla"
        assert result.from_cache is False

    def test_with_cache_flag(self):
        """Test creating tag result with cache flag."""
        result = TagResult(
            tokens=[],
            processing_time_ms=0.5,
            model="lasla",
            from_cache=True
        )
        assert result.from_cache is True


class TestBatchTagResult:
    """Test suite for BatchTagResult schema."""

    def test_creation(self):
        """Test creating batch tag result."""
        result = BatchTagResult(
            results=[[{"form": "a"}], [{"form": "b"}]],
            total_texts=2,
            processing_time_ms=25.0,
            model="lasla"
        )
        assert len(result.results) == 2
        assert result.total_texts == 2
        assert result.cache_hits == 0

    def test_with_cache_hits(self):
        """Test creating batch result with cache hits."""
        result = BatchTagResult(
            results=[[{"form": "a"}]],
            total_texts=1,
            processing_time_ms=5.0,
            model="lasla",
            cache_hits=1
        )
        assert result.cache_hits == 1


class TestCacheStats:
    """Test suite for CacheStats schema."""

    def test_creation(self):
        """Test creating cache stats."""
        stats = CacheStats(
            size=100,
            max_size=1000,
            ttl_seconds=3600,
            hits=500,
            misses=100,
            hit_rate_percent=83.33
        )
        assert stats.size == 100
        assert stats.max_size == 1000
        assert stats.ttl_seconds == 3600
        assert stats.hits == 500
        assert stats.misses == 100
        assert stats.hit_rate_percent == 83.33

    def test_requires_all_fields(self):
        """Test that all fields are required."""
        with pytest.raises(ValidationError):
            CacheStats(size=100)  # Missing required fields
