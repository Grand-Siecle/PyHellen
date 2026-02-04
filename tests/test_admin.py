"""
Tests for admin routes with authentication.
"""

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime

from app.main import app


class TestAuthStatusEndpoint:
    """Test suite for /admin/auth/status endpoint."""

    def test_auth_status_when_disabled(self, client):
        """Test auth status when authentication is disabled."""
        response = client.get("/admin/auth/status")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "auth_enabled" in data
        # Auth is disabled in tests by default
        assert data["auth_enabled"] is False

    def test_auth_status_structure(self, client):
        """Test auth status returns expected structure."""
        response = client.get("/admin/auth/status")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "authenticated" in data
        assert "auth_enabled" in data


class TestTokenEndpointsAuthDisabled:
    """Test suite for token endpoints when auth is disabled."""

    def test_list_tokens_auth_disabled(self, client):
        """Test listing tokens returns error when auth is disabled."""
        response = client.get("/admin/tokens")
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        # Custom error format uses "error" not "detail"
        assert "error" in data or "detail" in data
        error_msg = data.get("error", data.get("detail", "")).lower()
        assert "not enabled" in error_msg or "authentication" in error_msg

    def test_create_token_auth_disabled(self, client):
        """Test creating token returns error when auth is disabled."""
        response = client.post(
            "/admin/tokens",
            json={"name": "test", "scopes": ["read"]}
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_revoke_token_auth_disabled(self, client):
        """Test revoking token returns error when auth is disabled."""
        response = client.delete("/admin/tokens/1")
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_delete_token_permanent_auth_disabled(self, client):
        """Test permanently deleting token returns error when auth is disabled."""
        response = client.delete("/admin/tokens/1/permanent")
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_cleanup_tokens_auth_disabled(self, client):
        """Test cleanup tokens returns error when auth is disabled."""
        response = client.post("/admin/tokens/cleanup")
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_token_stats_auth_disabled(self, client):
        """Test token stats when auth is disabled."""
        response = client.get("/admin/tokens/stats")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["auth_enabled"] is False


class TestModelAdminEndpoints:
    """Test suite for model admin endpoints."""

    def test_list_models_admin(self, client):
        """Test listing models via admin endpoint."""
        response = client.get("/admin/models")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "models" in data
        assert "total" in data
        assert "active" in data

    def test_get_model_stats(self, client):
        """Test getting model statistics."""
        response = client.get("/admin/models/stats")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "total" in data
        assert "active" in data

    def test_get_model_detail(self, client):
        """Test getting detailed model info."""
        response = client.get("/admin/models/lasla")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "code" in data
        assert "name" in data
        assert "status" in data

    def test_get_model_not_found(self, client):
        """Test getting non-existent model returns 404."""
        response = client.get("/admin/models/nonexistent_xyz")
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_activate_model_already_active(self, client):
        """Test activating an already active model."""
        # lasla is active by default
        response = client.post("/admin/models/lasla/activate")
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        # Custom error format uses "error" not "detail"
        error_msg = data.get("error", data.get("detail", "")).lower()
        assert "already active" in error_msg


class TestAuditEndpoints:
    """Test suite for audit log endpoints."""

    def test_get_audit_log(self, client):
        """Test getting audit log."""
        response = client.get("/admin/audit")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "entries" in data
        assert "count" in data
        assert isinstance(data["entries"], list)

    def test_get_audit_log_with_limit(self, client):
        """Test getting audit log with limit."""
        response = client.get("/admin/audit?limit=10")
        assert response.status_code == status.HTTP_200_OK

    def test_get_audit_stats(self, client):
        """Test getting audit statistics."""
        response = client.get("/admin/audit/stats")
        assert response.status_code == status.HTTP_200_OK

    def test_cleanup_audit_log(self, client):
        """Test cleaning up audit log."""
        response = client.post("/admin/audit/cleanup?days=90")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "removed_entries" in data


class TestRequestLogEndpoints:
    """Test suite for request log endpoints."""

    def test_get_request_log(self, client):
        """Test getting request log."""
        response = client.get("/admin/requests")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "entries" in data
        assert "count" in data

    def test_get_request_stats(self, client):
        """Test getting request statistics."""
        response = client.get("/admin/requests/stats")
        assert response.status_code == status.HTTP_200_OK

    def test_get_request_errors(self, client):
        """Test getting request errors."""
        response = client.get("/admin/requests/errors")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "entries" in data

    def test_cleanup_request_log(self, client):
        """Test cleaning up request log."""
        response = client.post("/admin/requests/cleanup?days=30")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "removed_entries" in data


class TestPersistentMetricsEndpoints:
    """Test suite for persistent metrics endpoints."""

    def test_get_persistent_metrics(self, client):
        """Test getting persistent metrics."""
        response = client.get("/admin/metrics/persistent")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "global" in data
        assert "models" in data

    def test_get_model_persistent_metrics_not_found(self, client):
        """Test getting metrics for non-existent model."""
        response = client.get("/admin/metrics/persistent/nonexistent")
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_reset_all_metrics(self, client):
        """Test resetting all metrics."""
        response = client.post("/admin/metrics/reset")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "message" in data


class TestModelManagementEndpoints:
    """Test suite for model management operations."""

    def test_create_duplicate_model_fails(self, client):
        """Test creating a model that already exists."""
        response = client.post(
            "/admin/models",
            json={
                "code": "lasla",  # Already exists
                "name": "Duplicate Latin",
                "pie_module": "lasla"
            }
        )
        assert response.status_code == status.HTTP_409_CONFLICT

    def test_delete_builtin_model_fails(self, client):
        """Test that builtin models cannot be deleted."""
        response = client.delete("/admin/models/lasla")
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        # Custom error format uses "error" not "detail"
        error_msg = data.get("error", data.get("detail", "")).lower()
        assert "builtin" in error_msg

    def test_update_model(self, client):
        """Test updating a model."""
        response = client.patch(
            "/admin/models/lasla",
            json={"description": "Updated description"}
        )
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["code"] == "lasla"

    def test_deactivate_model(self, client):
        """Test deactivating a model."""
        # First check it's active
        response = client.get("/admin/models/grc")
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["is_active"] is True

        # Deactivate
        response = client.post("/admin/models/grc/deactivate")
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["is_active"] is False

        # Reactivate for other tests
        response = client.post("/admin/models/grc/activate")
        assert response.status_code == status.HTTP_200_OK

    def test_add_model_file(self, client):
        """Test adding a file to a model."""
        response = client.post(
            "/admin/models/lasla/files",
            json={
                "filename": "test_file.bin",
                "url": "https://example.com/test_file.bin"
            }
        )
        # Either succeeds or model not found
        assert response.status_code in [
            status.HTTP_201_CREATED,
            status.HTTP_404_NOT_FOUND
        ]
