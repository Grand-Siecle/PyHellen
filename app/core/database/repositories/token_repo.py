"""Repository for token management using SQLModel."""

import hashlib
import json
import secrets
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Tuple

from sqlmodel import Session, select, func

from app.core.database.models import Token
from app.core.database.repositories.base import BaseRepository
from app.core.logger import logger


class TokenScope(str, Enum):
    """Available token scopes."""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class TokenRepository(BaseRepository):
    """
    Repository for managing API tokens.

    Tokens are stored as SHA-256 hashes - the plain token is only
    returned once at creation time and cannot be recovered.
    """

    @staticmethod
    def _generate_token() -> str:
        """Generate a cryptographically secure token."""
        return f"pyhellen_{secrets.token_urlsafe(32)}"

    @staticmethod
    def _hash_token(token: str, secret_key: str) -> str:
        """Hash a token using SHA-256 with secret key as salt."""
        salted = f"{secret_key}:{token}"
        return hashlib.sha256(salted.encode()).hexdigest()

    def _token_to_response(self, token: Token, truncate_hash: bool = True) -> Token:
        """Prepare token for response (truncate hash for security)."""
        if truncate_hash:
            token.token_hash = token.token_hash[:16] + "..."
        return token

    def create(
        self,
        name: str,
        scopes: List[TokenScope],
        secret_key: str,
        expires_days: Optional[int] = None
    ) -> Tuple[Token, str]:
        """
        Create a new token.

        Returns:
            Tuple of (Token object, plain token string)
            The plain token is only available at creation time!
        """
        plain_token = self._generate_token()
        token_hash = self._hash_token(plain_token, secret_key)

        now = datetime.utcnow()
        expires_at = None
        if expires_days:
            expires_at = now + timedelta(days=expires_days)

        scopes_json = json.dumps([s.value for s in scopes])

        session = self._get_session()
        try:
            token = Token(
                name=name,
                token_hash=token_hash,
                scopes=scopes_json,
                created_at=now,
                expires_at=expires_at,
                is_active=True,
            )
            session.add(token)
            session.commit()
            session.refresh(token)

            logger.info(f"Created token '{name}' (id={token.id}) with scopes: {[s.value for s in scopes]}")

            # Truncate hash for return
            token.token_hash = token_hash[:16] + "..."
            return token, plain_token
        finally:
            self._close_session(session)

    def validate(self, plain_token: str, secret_key: str) -> Optional[Token]:
        """
        Validate a token and return its data if valid.

        Updates last_used_at on successful validation.
        Returns None if token is invalid, expired, or inactive.
        """
        token_hash = self._hash_token(plain_token, secret_key)

        session = self._get_session()
        try:
            token = session.exec(
                select(Token).where(
                    Token.token_hash == token_hash,
                    Token.is_active == True
                )
            ).first()

            if not token:
                return None

            # Check expiration
            if token.expires_at and token.expires_at < datetime.utcnow():
                logger.warning(f"Token '{token.name}' has expired")
                return None

            # Update last_used_at
            token.last_used_at = datetime.utcnow()
            session.add(token)
            session.commit()
            session.refresh(token)

            return self._token_to_response(token)
        finally:
            self._close_session(session)

    def list_all(self) -> List[Token]:
        """List all tokens (without sensitive hash data)."""
        session = self._get_session()
        try:
            tokens = list(session.exec(
                select(Token).order_by(Token.created_at.desc())
            ).all())

            return [self._token_to_response(t) for t in tokens]
        finally:
            self._close_session(session)

    def get_by_id(self, token_id: int) -> Optional[Token]:
        """Get a token by ID."""
        session = self._get_session()
        try:
            token = session.get(Token, token_id)
            return self._token_to_response(token) if token else None
        finally:
            self._close_session(session)

    def revoke(self, token_id: int) -> bool:
        """Revoke a token by ID."""
        session = self._get_session()
        try:
            token = session.get(Token, token_id)
            if not token:
                return False

            token.is_active = False
            session.add(token)
            session.commit()
            logger.info(f"Revoked token id={token_id}")
            return True
        finally:
            self._close_session(session)

    def delete(self, token_id: int) -> bool:
        """Permanently delete a token."""
        session = self._get_session()
        try:
            token = session.get(Token, token_id)
            if not token:
                return False

            session.delete(token)
            session.commit()
            logger.info(f"Deleted token id={token_id}")
            return True
        finally:
            self._close_session(session)

    def cleanup_expired(self) -> int:
        """Remove expired tokens. Returns count of removed tokens."""
        now = datetime.utcnow()

        session = self._get_session()
        try:
            expired = list(session.exec(
                select(Token).where(
                    Token.expires_at.is_not(None),
                    Token.expires_at < now
                )
            ).all())

            count = len(expired)
            for token in expired:
                session.delete(token)

            session.commit()
            if count > 0:
                logger.info(f"Cleaned up {count} expired tokens")
            return count
        finally:
            self._close_session(session)

    def get_statistics(self) -> dict:
        """Get token statistics."""
        session = self._get_session()
        try:
            total = session.exec(select(func.count(Token.id))).one()
            active = session.exec(
                select(func.count(Token.id)).where(Token.is_active == True)
            ).one()
            expired = session.exec(
                select(func.count(Token.id)).where(
                    Token.expires_at.is_not(None),
                    Token.expires_at < datetime.utcnow()
                )
            ).one()

            return {
                "total": total,
                "active": active,
                "inactive": total - active,
                "expired": expired
            }
        finally:
            self._close_session(session)

    def has_any_tokens(self) -> bool:
        """Check if any tokens exist in the database."""
        session = self._get_session()
        try:
            result = session.exec(select(Token.id).limit(1)).first()
            return result is not None
        finally:
            self._close_session(session)
