"""
Tests for TokenRepository - advanced token operations.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from app.core.database.repositories.token_repo import TokenRepository, TokenScope


class TestTokenRepositoryBasics:
    """Test suite for basic TokenRepository operations."""

    @pytest.fixture
    def token_repo(self):
        """Create a TokenRepository with mocked database."""
        repo = TokenRepository()
        return repo

    @pytest.fixture
    def secret_key(self):
        return "test_secret_key_12345"

    def test_generate_token_format(self):
        """Test that generated tokens have the correct format."""
        token = TokenRepository._generate_token()
        assert token.startswith("pyhellen_")
        assert len(token) > 20

    def test_generate_token_unique(self):
        """Test that generated tokens are unique."""
        tokens = [TokenRepository._generate_token() for _ in range(100)]
        assert len(set(tokens)) == 100

    def test_hash_token_deterministic(self, secret_key):
        """Test that hashing is deterministic."""
        token = "test_token"
        hash1 = TokenRepository._hash_token(token, secret_key)
        hash2 = TokenRepository._hash_token(token, secret_key)
        assert hash1 == hash2

    def test_hash_token_different_with_different_secret(self):
        """Test that different secrets produce different hashes."""
        token = "test_token"
        hash1 = TokenRepository._hash_token(token, "secret1")
        hash2 = TokenRepository._hash_token(token, "secret2")
        assert hash1 != hash2

    def test_hash_token_different_for_different_tokens(self, secret_key):
        """Test that different tokens produce different hashes."""
        hash1 = TokenRepository._hash_token("token1", secret_key)
        hash2 = TokenRepository._hash_token("token2", secret_key)
        assert hash1 != hash2


class TestTokenRepositoryCreate:
    """Test suite for token creation."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        session = MagicMock()
        return session

    def test_create_token_returns_tuple(self):
        """Test that create returns a tuple of (Token, plain_token)."""
        repo = TokenRepository()

        with patch.object(repo, '_get_session') as mock_get_session, \
             patch.object(repo, '_close_session'):
            mock_session = MagicMock()
            mock_get_session.return_value = mock_session

            # Mock the token object returned after commit
            mock_token = Mock()
            mock_token.id = 1
            mock_token.name = "test_token"
            mock_token.token_hash = "a" * 64

            def refresh_side_effect(token):
                token.id = 1

            mock_session.refresh.side_effect = refresh_side_effect

            result = repo.create(
                name="test_token",
                scopes=[TokenScope.READ],
                secret_key="secret"
            )

            assert isinstance(result, tuple)
            assert len(result) == 2
            # Second element is the plain token
            assert result[1].startswith("pyhellen_")

    def test_create_token_with_expiry(self):
        """Test creating a token with expiration."""
        repo = TokenRepository()

        with patch.object(repo, '_get_session') as mock_get_session, \
             patch.object(repo, '_close_session'):
            mock_session = MagicMock()
            mock_get_session.return_value = mock_session

            result = repo.create(
                name="expiring_token",
                scopes=[TokenScope.READ],
                secret_key="secret",
                expires_days=30
            )

            # Verify session.add was called with a token that has expires_at set
            add_call = mock_session.add.call_args
            token_arg = add_call[0][0]
            assert token_arg.expires_at is not None

    def test_create_token_without_expiry(self):
        """Test creating a token without expiration."""
        repo = TokenRepository()

        with patch.object(repo, '_get_session') as mock_get_session, \
             patch.object(repo, '_close_session'):
            mock_session = MagicMock()
            mock_get_session.return_value = mock_session

            result = repo.create(
                name="permanent_token",
                scopes=[TokenScope.ADMIN],
                secret_key="secret",
                expires_days=None
            )

            add_call = mock_session.add.call_args
            token_arg = add_call[0][0]
            assert token_arg.expires_at is None


class TestTokenRepositoryValidate:
    """Test suite for token validation."""

    def test_validate_returns_none_for_invalid_token(self):
        """Test that validation returns None for invalid token."""
        repo = TokenRepository()

        with patch.object(repo, '_get_session') as mock_get_session, \
             patch.object(repo, '_close_session'):
            mock_session = MagicMock()
            mock_get_session.return_value = mock_session
            mock_session.exec.return_value.first.return_value = None

            result = repo.validate("invalid_token", "secret")
            assert result is None

    def test_validate_returns_none_for_expired_token(self):
        """Test that validation returns None for expired token."""
        repo = TokenRepository()

        with patch.object(repo, '_get_session') as mock_get_session, \
             patch.object(repo, '_close_session'):
            mock_session = MagicMock()
            mock_get_session.return_value = mock_session

            # Create an expired token
            expired_token = Mock()
            expired_token.expires_at = datetime.utcnow() - timedelta(days=1)
            expired_token.name = "expired"
            mock_session.exec.return_value.first.return_value = expired_token

            result = repo.validate("some_token", "secret")
            assert result is None

    def test_validate_updates_last_used(self):
        """Test that validation updates last_used_at."""
        repo = TokenRepository()

        with patch.object(repo, '_get_session') as mock_get_session, \
             patch.object(repo, '_close_session'):
            mock_session = MagicMock()
            mock_get_session.return_value = mock_session

            valid_token = Mock()
            valid_token.expires_at = datetime.utcnow() + timedelta(days=30)
            valid_token.last_used_at = None
            valid_token.token_hash = "a" * 64
            mock_session.exec.return_value.first.return_value = valid_token

            repo.validate("some_token", "secret")

            # Verify last_used_at was updated
            assert valid_token.last_used_at is not None
            mock_session.add.assert_called()
            mock_session.commit.assert_called()


class TestTokenRepositoryList:
    """Test suite for listing tokens."""

    def test_list_all_returns_list(self):
        """Test that list_all returns a list."""
        repo = TokenRepository()

        with patch.object(repo, '_get_session') as mock_get_session, \
             patch.object(repo, '_close_session'):
            mock_session = MagicMock()
            mock_get_session.return_value = mock_session

            mock_token = Mock()
            mock_token.token_hash = "a" * 64
            mock_session.exec.return_value.all.return_value = [mock_token]

            result = repo.list_all()

            assert isinstance(result, list)
            assert len(result) == 1

    def test_list_all_truncates_hash(self):
        """Test that list_all truncates token hashes."""
        repo = TokenRepository()

        with patch.object(repo, '_get_session') as mock_get_session, \
             patch.object(repo, '_close_session'):
            mock_session = MagicMock()
            mock_get_session.return_value = mock_session

            mock_token = Mock()
            mock_token.token_hash = "a" * 64
            mock_session.exec.return_value.all.return_value = [mock_token]

            result = repo.list_all()

            # Hash should be truncated
            assert result[0].token_hash.endswith("...")


class TestTokenRepositoryRevoke:
    """Test suite for token revocation."""

    def test_revoke_returns_false_for_nonexistent(self):
        """Test that revoke returns False for nonexistent token."""
        repo = TokenRepository()

        with patch.object(repo, '_get_session') as mock_get_session, \
             patch.object(repo, '_close_session'):
            mock_session = MagicMock()
            mock_get_session.return_value = mock_session
            mock_session.get.return_value = None

            result = repo.revoke(999)
            assert result is False

    def test_revoke_sets_inactive(self):
        """Test that revoke sets token to inactive."""
        repo = TokenRepository()

        with patch.object(repo, '_get_session') as mock_get_session, \
             patch.object(repo, '_close_session'):
            mock_session = MagicMock()
            mock_get_session.return_value = mock_session

            mock_token = Mock()
            mock_token.is_active = True
            mock_session.get.return_value = mock_token

            result = repo.revoke(1)

            assert result is True
            assert mock_token.is_active is False
            mock_session.commit.assert_called()


class TestTokenRepositoryDelete:
    """Test suite for token deletion."""

    def test_delete_returns_false_for_nonexistent(self):
        """Test that delete returns False for nonexistent token."""
        repo = TokenRepository()

        with patch.object(repo, '_get_session') as mock_get_session, \
             patch.object(repo, '_close_session'):
            mock_session = MagicMock()
            mock_get_session.return_value = mock_session
            mock_session.get.return_value = None

            result = repo.delete(999)
            assert result is False

    def test_delete_removes_token(self):
        """Test that delete removes the token."""
        repo = TokenRepository()

        with patch.object(repo, '_get_session') as mock_get_session, \
             patch.object(repo, '_close_session'):
            mock_session = MagicMock()
            mock_get_session.return_value = mock_session

            mock_token = Mock()
            mock_session.get.return_value = mock_token

            result = repo.delete(1)

            assert result is True
            mock_session.delete.assert_called_with(mock_token)
            mock_session.commit.assert_called()


class TestTokenRepositoryCleanup:
    """Test suite for expired token cleanup."""

    def test_cleanup_expired_returns_count(self):
        """Test that cleanup_expired returns count of removed tokens."""
        repo = TokenRepository()

        with patch.object(repo, '_get_session') as mock_get_session, \
             patch.object(repo, '_close_session'):
            mock_session = MagicMock()
            mock_get_session.return_value = mock_session

            expired_tokens = [Mock(), Mock(), Mock()]
            mock_session.exec.return_value.all.return_value = expired_tokens

            result = repo.cleanup_expired()

            assert result == 3
            assert mock_session.delete.call_count == 3

    def test_cleanup_expired_no_expired_tokens(self):
        """Test cleanup when no tokens are expired."""
        repo = TokenRepository()

        with patch.object(repo, '_get_session') as mock_get_session, \
             patch.object(repo, '_close_session'):
            mock_session = MagicMock()
            mock_get_session.return_value = mock_session
            mock_session.exec.return_value.all.return_value = []

            result = repo.cleanup_expired()

            assert result == 0
            mock_session.delete.assert_not_called()


class TestTokenRepositoryStatistics:
    """Test suite for token statistics."""

    def test_get_statistics_returns_dict(self):
        """Test that get_statistics returns expected dict structure."""
        repo = TokenRepository()

        with patch.object(repo, '_get_session') as mock_get_session, \
             patch.object(repo, '_close_session'):
            mock_session = MagicMock()
            mock_get_session.return_value = mock_session

            # Mock the count queries
            mock_session.exec.return_value.one.side_effect = [10, 8, 2]

            result = repo.get_statistics()

            assert "total" in result
            assert "active" in result
            assert "inactive" in result
            assert "expired" in result
            assert result["total"] == 10
            assert result["active"] == 8
            assert result["inactive"] == 2
            assert result["expired"] == 2


class TestTokenRepositoryHasTokens:
    """Test suite for has_any_tokens check."""

    def test_has_any_tokens_true(self):
        """Test has_any_tokens returns True when tokens exist."""
        repo = TokenRepository()

        with patch.object(repo, '_get_session') as mock_get_session, \
             patch.object(repo, '_close_session'):
            mock_session = MagicMock()
            mock_get_session.return_value = mock_session
            mock_session.exec.return_value.first.return_value = 1

            result = repo.has_any_tokens()
            assert result is True

    def test_has_any_tokens_false(self):
        """Test has_any_tokens returns False when no tokens exist."""
        repo = TokenRepository()

        with patch.object(repo, '_get_session') as mock_get_session, \
             patch.object(repo, '_close_session'):
            mock_session = MagicMock()
            mock_get_session.return_value = mock_session
            mock_session.exec.return_value.first.return_value = None

            result = repo.has_any_tokens()
            assert result is False


class TestTokenScope:
    """Test suite for TokenScope enum."""

    def test_scope_values(self):
        """Test that all expected scopes exist."""
        assert TokenScope.READ.value == "read"
        assert TokenScope.WRITE.value == "write"
        assert TokenScope.ADMIN.value == "admin"

    def test_scope_is_string_enum(self):
        """Test that TokenScope is a string enum."""
        assert isinstance(TokenScope.READ, str)
        assert TokenScope.READ == "read"
