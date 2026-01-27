"""Tests for the database module with SQLModel."""

import pytest
import tempfile
import os
from datetime import datetime, timedelta

# We need to set up environment before importing app modules
os.environ.setdefault("TOKEN_DB_PATH", ":memory:")


class TestDatabaseModule:
    """Tests for the database module initialization and repositories."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database file."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    @pytest.fixture
    def db_engine(self, temp_db_path):
        """Create a database engine with temporary database."""
        from app.core.database import engine as engine_module
        from app.core.database.engine import DatabaseEngine
        # Reset singletons for testing
        DatabaseEngine._instance = None
        engine_module._db_engine = None
        db = DatabaseEngine(temp_db_path)
        yield db
        DatabaseEngine._instance = None
        engine_module._db_engine = None

    @pytest.fixture
    def model_repo(self, db_engine):
        """Create a model repository."""
        from app.core.database.repositories.model_repo import ModelRepository
        return ModelRepository()

    @pytest.fixture
    def cache_repo(self, db_engine):
        """Create a cache repository."""
        from app.core.database.repositories.cache_repo import CacheRepository
        return CacheRepository(max_size=100, ttl_seconds=3600)

    @pytest.fixture
    def metrics_repo(self, db_engine):
        """Create a metrics repository."""
        from app.core.database.repositories.metrics_repo import MetricsRepository
        return MetricsRepository()

    @pytest.fixture
    def audit_repo(self, db_engine):
        """Create an audit repository."""
        from app.core.database.repositories.audit_repo import AuditRepository
        return AuditRepository()

    @pytest.fixture
    def request_log_repo(self, db_engine):
        """Create a request log repository."""
        from app.core.database.repositories.request_log_repo import RequestLogRepository
        return RequestLogRepository()

    # ==================
    # Model Repository Tests
    # ==================

    def test_model_repo_get_all_builtin(self, model_repo):
        """Test that builtin models are created on initialization."""
        models = model_repo.get_all()
        assert len(models) == 7  # 7 builtin models

        # Check that all are builtin and active
        for model in models:
            assert model.is_builtin is True
            assert model.is_active is True

    def test_model_repo_get_by_code(self, model_repo):
        """Test getting a model by code."""
        model = model_repo.get_by_code("lasla")
        assert model is not None
        assert model.code == "lasla"
        assert model.name == "Classical Latin"

    def test_model_repo_is_valid_model(self, model_repo):
        """Test model validation."""
        assert model_repo.is_valid_model("lasla") is True
        assert model_repo.is_valid_model("invalid") is False

    def test_model_repo_create_custom(self, model_repo):
        """Test creating a custom model."""
        model = model_repo.create(
            code="custom_test",
            name="Custom Test Model",
            pie_module="custom_test",
            description="A test model"
        )
        assert model is not None
        assert model.code == "custom_test"
        assert model.is_builtin is False
        assert model.is_active is True

    def test_model_repo_deactivate(self, model_repo):
        """Test deactivating a model."""
        success = model_repo.deactivate("lasla")
        assert success is True

        model = model_repo.get_by_code("lasla")
        assert model.is_active is False

        # Should not appear in active list
        assert "lasla" not in model_repo.get_active_codes()

    def test_model_repo_delete_builtin_fails(self, model_repo):
        """Test that deleting a builtin model fails."""
        success = model_repo.delete("lasla")
        assert success is False

    def test_model_repo_delete_custom_succeeds(self, model_repo):
        """Test that deleting a custom model succeeds."""
        # Create custom model first
        model_repo.create(
            code="to_delete",
            name="To Delete",
            pie_module="to_delete"
        )

        success = model_repo.delete("to_delete")
        assert success is True

        model = model_repo.get_by_code("to_delete")
        assert model is None

    # ==================
    # Cache Repository Tests
    # ==================

    def test_cache_repo_set_and_get(self, cache_repo):
        """Test setting and getting cache values."""
        cache_repo.set("lasla", "test text", False, {"result": "data"})

        result = cache_repo.get("lasla", "test text", False)
        assert result == {"result": "data"}

    def test_cache_repo_miss(self, cache_repo):
        """Test cache miss."""
        result = cache_repo.get("lasla", "nonexistent", False)
        assert result is None

    def test_cache_repo_clear(self, cache_repo):
        """Test clearing cache."""
        cache_repo.set("lasla", "text1", False, {"a": 1})
        cache_repo.set("lasla", "text2", False, {"b": 2})

        count = cache_repo.clear()
        assert count == 2

        stats = cache_repo.get_statistics()
        assert stats["size"] == 0

    # ==================
    # Metrics Repository Tests
    # ==================

    def test_metrics_repo_update_load(self, metrics_repo):
        """Test updating load metrics."""
        success = metrics_repo.update_load_metrics("lasla", 100.0)
        assert success is True

        metrics = metrics_repo.get_by_model("lasla")
        assert metrics.load_count == 1
        assert metrics.load_time_total_ms == 100.0

    def test_metrics_repo_update_process(self, metrics_repo):
        """Test updating process metrics."""
        metrics_repo.update_process_metrics("lasla", 50.0)
        metrics_repo.update_process_metrics("lasla", 30.0)

        metrics = metrics_repo.get_by_model("lasla")
        assert metrics.process_count == 2
        assert metrics.process_time_total_ms == 80.0

    def test_metrics_repo_global_stats(self, metrics_repo):
        """Test global statistics."""
        metrics_repo.update_load_metrics("lasla", 100.0)
        metrics_repo.update_process_metrics("lasla", 50.0)
        metrics_repo.update_process_metrics("grc", 30.0)

        stats = metrics_repo.get_global_statistics()
        assert stats["total_loads"] == 1
        assert stats["total_processed"] == 2

    # ==================
    # Audit Repository Tests
    # ==================

    def test_audit_repo_log(self, audit_repo):
        """Test logging audit events."""
        entry_id = audit_repo.log(
            action="test.action",
            target_type="test",
            target_id="123",
            details={"key": "value"}
        )
        assert entry_id > 0

        entries = audit_repo.get_recent(limit=1)
        assert len(entries) == 1
        assert entries[0].action == "test.action"

    def test_audit_repo_get_by_action(self, audit_repo):
        """Test getting audit entries by action."""
        audit_repo.log(action="auth.success")
        audit_repo.log(action="auth.failed")
        audit_repo.log(action="auth.success")

        entries = audit_repo.get_by_action("auth.success")
        assert len(entries) == 2

    # ==================
    # Request Log Repository Tests
    # ==================

    def test_request_log_repo_log(self, request_log_repo):
        """Test logging requests."""
        entry_id = request_log_repo.log(
            endpoint="/api/tag/lasla",
            method="POST",
            status_code=200,
            model_code="lasla",
            processing_time_ms=50.0
        )
        assert entry_id > 0

    def test_request_log_repo_statistics(self, request_log_repo):
        """Test request statistics."""
        # Log some requests
        request_log_repo.log("/api/tag/lasla", "POST", 200, model_code="lasla", processing_time_ms=50.0)
        request_log_repo.log("/api/tag/lasla", "POST", 200, model_code="lasla", processing_time_ms=30.0, from_cache=True)
        request_log_repo.log("/api/tag/grc", "POST", 500, model_code="grc", error_message="Test error")

        stats = request_log_repo.get_statistics(hours=1)
        assert stats["total_requests"] == 3
        assert stats["successful_requests"] == 2
        assert stats["cache_hits"] == 1
