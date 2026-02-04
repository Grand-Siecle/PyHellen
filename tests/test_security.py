"""Tests for the security module."""

import pytest
import tempfile
import os
from datetime import datetime, timedelta

from app.core.security.database import TokenDatabase
from app.core.security.models import TokenScope, TokenCreate
from app.core.security.auth import AuthManager


class TestTokenDatabase:
    """Tests for TokenDatabase class."""

    @pytest.fixture
    def db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        db = TokenDatabase(db_path)
        yield db

        # Cleanup
        os.unlink(db_path)

    @pytest.fixture
    def secret_key(self):
        """Test secret key."""
        return "test_secret_key_for_testing_only"

    def test_create_token(self, db, secret_key):
        """Test token creation."""
        token_obj, plain_token = db.create_token(
            name="Test Token",
            scopes=[TokenScope.READ],
            secret_key=secret_key
        )

        assert token_obj.id == 1
        assert token_obj.name == "Test Token"
        assert token_obj.scopes == [TokenScope.READ]
        assert token_obj.is_active is True
        assert token_obj.expires_at is None
        assert plain_token.startswith("pyhellen_")

    def test_create_token_with_expiry(self, db, secret_key):
        """Test token creation with expiry."""
        token_obj, _ = db.create_token(
            name="Expiring Token",
            scopes=[TokenScope.READ],
            secret_key=secret_key,
            expires_days=30
        )

        assert token_obj.expires_at is not None
        assert token_obj.expires_at > datetime.utcnow()

    def test_validate_token(self, db, secret_key):
        """Test token validation."""
        token_obj, plain_token = db.create_token(
            name="Validate Test",
            scopes=[TokenScope.READ, TokenScope.WRITE],
            secret_key=secret_key
        )

        # Valid token
        validated = db.validate_token(plain_token, secret_key)
        assert validated is not None
        assert validated.id == token_obj.id
        assert validated.name == "Validate Test"
        assert TokenScope.READ in validated.scopes
        assert TokenScope.WRITE in validated.scopes

    def test_validate_invalid_token(self, db, secret_key):
        """Test validation of invalid token."""
        result = db.validate_token("invalid_token", secret_key)
        assert result is None

    def test_validate_with_wrong_secret(self, db, secret_key):
        """Test validation with wrong secret key."""
        _, plain_token = db.create_token(
            name="Wrong Secret Test",
            scopes=[TokenScope.READ],
            secret_key=secret_key
        )

        result = db.validate_token(plain_token, "wrong_secret")
        assert result is None

    def test_revoke_token(self, db, secret_key):
        """Test token revocation."""
        token_obj, plain_token = db.create_token(
            name="Revoke Test",
            scopes=[TokenScope.READ],
            secret_key=secret_key
        )

        # Revoke the token
        success = db.revoke_token(token_obj.id)
        assert success is True

        # Token should no longer validate
        result = db.validate_token(plain_token, secret_key)
        assert result is None

    def test_delete_token(self, db, secret_key):
        """Test token deletion."""
        token_obj, _ = db.create_token(
            name="Delete Test",
            scopes=[TokenScope.READ],
            secret_key=secret_key
        )

        # Delete the token
        success = db.delete_token(token_obj.id)
        assert success is True

        # Token should not be in list
        tokens = db.list_tokens()
        assert len(tokens) == 0

    def test_list_tokens(self, db, secret_key):
        """Test listing tokens."""
        db.create_token("Token 1", [TokenScope.READ], secret_key)
        db.create_token("Token 2", [TokenScope.WRITE], secret_key)
        db.create_token("Token 3", [TokenScope.ADMIN], secret_key)

        tokens = db.list_tokens()
        assert len(tokens) == 3

    def test_token_stats(self, db, secret_key):
        """Test token statistics."""
        db.create_token("Active 1", [TokenScope.READ], secret_key)
        token_obj, _ = db.create_token("Active 2", [TokenScope.READ], secret_key)

        # Revoke one token
        db.revoke_token(token_obj.id)

        stats = db.get_token_count()
        assert stats["total"] == 2
        assert stats["active"] == 1
        assert stats["inactive"] == 1

    def test_last_used_updated(self, db, secret_key):
        """Test that last_used_at is updated on validation."""
        token_obj, plain_token = db.create_token(
            name="Last Used Test",
            scopes=[TokenScope.READ],
            secret_key=secret_key
        )

        # Initially no last_used
        tokens = db.list_tokens()
        assert tokens[0].last_used_at is None

        # Validate token
        db.validate_token(plain_token, secret_key)

        # Check last_used is now set
        tokens = db.list_tokens()
        assert tokens[0].last_used_at is not None


class TestAuthManager:
    """Tests for AuthManager class."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_auth_disabled(self, temp_db_path):
        """Test auth manager when disabled."""
        manager = AuthManager(
            enabled=False,
            secret_key="",
            db_path=temp_db_path
        )
        manager.initialize()

        assert manager.enabled is False
        assert manager.db is None

    def test_auth_enabled_requires_secret(self, temp_db_path):
        """Test that auth enabled requires secret key."""
        manager = AuthManager(
            enabled=True,
            secret_key="",
            db_path=temp_db_path
        )

        with pytest.raises(ValueError, match="SECRET_KEY is required"):
            manager.initialize()

    def test_auth_enabled_with_secret(self, temp_db_path):
        """Test auth enabled with proper secret."""
        manager = AuthManager(
            enabled=True,
            secret_key="test_secret_key",
            db_path=temp_db_path,
            auto_create_admin=False
        )
        manager.initialize()

        assert manager.enabled is True
        assert manager.db is not None

    def test_auto_create_admin_token(self, temp_db_path):
        """Test automatic admin token creation."""
        manager = AuthManager(
            enabled=True,
            secret_key="test_secret_key",
            db_path=temp_db_path,
            auto_create_admin=True
        )
        manager.initialize()

        # Should have created one admin token
        stats = manager.db.get_token_count()
        assert stats["total"] == 1

        # Token should have admin scope
        tokens = manager.db.list_tokens()
        assert TokenScope.ADMIN in tokens[0].scopes

    def test_create_and_validate_token(self, temp_db_path):
        """Test token creation and validation through manager."""
        manager = AuthManager(
            enabled=True,
            secret_key="test_secret_key",
            db_path=temp_db_path,
            auto_create_admin=False
        )
        manager.initialize()

        # Create token
        token_obj, plain_token = manager.create_token(
            name="Test",
            scopes=[TokenScope.READ]
        )

        # Validate token
        validated = manager.validate_token(plain_token)
        assert validated is not None
        assert validated.id == token_obj.id


class TestTokenScopes:
    """Tests for token scopes."""

    def test_scope_values(self):
        """Test scope enum values."""
        assert TokenScope.READ.value == "read"
        assert TokenScope.WRITE.value == "write"
        assert TokenScope.ADMIN.value == "admin"

    def test_scope_from_string(self):
        """Test creating scope from string."""
        assert TokenScope("read") == TokenScope.READ
        assert TokenScope("write") == TokenScope.WRITE
        assert TokenScope("admin") == TokenScope.ADMIN


class TestTokenModels:
    """Tests for token Pydantic models."""

    def test_token_create_defaults(self):
        """Test TokenCreate model defaults."""
        token = TokenCreate(name="Test")
        assert token.name == "Test"
        assert token.scopes == [TokenScope.READ]
        assert token.expires_days is None

    def test_token_create_with_scopes(self):
        """Test TokenCreate with custom scopes."""
        token = TokenCreate(
            name="Admin Token",
            scopes=[TokenScope.READ, TokenScope.WRITE, TokenScope.ADMIN]
        )
        assert len(token.scopes) == 3

    def test_token_create_validation(self):
        """Test TokenCreate validation."""
        # Name too short
        with pytest.raises(ValueError):
            TokenCreate(name="")

        # Invalid expires_days
        with pytest.raises(ValueError):
            TokenCreate(name="Test", expires_days=0)
