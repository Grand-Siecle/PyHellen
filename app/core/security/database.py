"""SQLite database manager for token storage."""

import sqlite3
import secrets
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Tuple
from contextlib import contextmanager

from app.core.security.models import Token, TokenScope
from app.core.logger import logger


class TokenDatabase:
    """
    SQLite-based token storage with secure hashing.

    Tokens are stored as SHA-256 hashes - the plain token is only
    returned once at creation time and cannot be recovered.
    """

    def __init__(self, db_path: str = "tokens.db"):
        self.db_path = Path(db_path)
        self._init_database()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_database(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tokens (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    token_hash TEXT NOT NULL UNIQUE,
                    scopes TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    last_used_at TEXT,
                    is_active INTEGER DEFAULT 1
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_token_hash ON tokens(token_hash)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_is_active ON tokens(is_active)
            """)
            logger.info(f"Token database initialized at {self.db_path}")

    @staticmethod
    def _generate_token() -> str:
        """Generate a cryptographically secure token."""
        return f"pyhellen_{secrets.token_urlsafe(32)}"

    @staticmethod
    def _hash_token(token: str, secret_key: str) -> str:
        """Hash a token using SHA-256 with secret key as salt."""
        salted = f"{secret_key}:{token}"
        return hashlib.sha256(salted.encode()).hexdigest()

    def create_token(
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

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO tokens (name, token_hash, scopes, created_at, expires_at, is_active)
                VALUES (?, ?, ?, ?, ?, 1)
                """,
                (
                    name,
                    token_hash,
                    scopes_json,
                    now.isoformat(),
                    expires_at.isoformat() if expires_at else None
                )
            )
            token_id = cursor.lastrowid

        token = Token(
            id=token_id,
            name=name,
            token_hash=token_hash[:16] + "...",  # Truncated for display
            scopes=scopes,
            created_at=now,
            expires_at=expires_at,
            last_used_at=None,
            is_active=True
        )

        logger.info(f"Created token '{name}' (id={token_id}) with scopes: {[s.value for s in scopes]}")
        return token, plain_token

    def validate_token(self, plain_token: str, secret_key: str) -> Optional[Token]:
        """
        Validate a token and return its data if valid.

        Updates last_used_at on successful validation.
        Returns None if token is invalid, expired, or inactive.
        """
        token_hash = self._hash_token(plain_token, secret_key)

        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT id, name, token_hash, scopes, created_at, expires_at, last_used_at, is_active
                FROM tokens
                WHERE token_hash = ? AND is_active = 1
                """,
                (token_hash,)
            ).fetchone()

            if not row:
                return None

            # Check expiration
            expires_at = None
            if row["expires_at"]:
                expires_at = datetime.fromisoformat(row["expires_at"])
                if expires_at < datetime.utcnow():
                    logger.warning(f"Token '{row['name']}' has expired")
                    return None

            # Update last_used_at
            now = datetime.utcnow()
            conn.execute(
                "UPDATE tokens SET last_used_at = ? WHERE id = ?",
                (now.isoformat(), row["id"])
            )

            scopes = [TokenScope(s) for s in json.loads(row["scopes"])]

            return Token(
                id=row["id"],
                name=row["name"],
                token_hash=row["token_hash"][:16] + "...",
                scopes=scopes,
                created_at=datetime.fromisoformat(row["created_at"]),
                expires_at=expires_at,
                last_used_at=now,
                is_active=bool(row["is_active"])
            )

    def list_tokens(self) -> List[Token]:
        """List all tokens (without sensitive hash data)."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, name, token_hash, scopes, created_at, expires_at, last_used_at, is_active
                FROM tokens
                ORDER BY created_at DESC
                """
            ).fetchall()

            tokens = []
            for row in rows:
                expires_at = datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None
                last_used = datetime.fromisoformat(row["last_used_at"]) if row["last_used_at"] else None
                scopes = [TokenScope(s) for s in json.loads(row["scopes"])]

                tokens.append(Token(
                    id=row["id"],
                    name=row["name"],
                    token_hash=row["token_hash"][:16] + "...",
                    scopes=scopes,
                    created_at=datetime.fromisoformat(row["created_at"]),
                    expires_at=expires_at,
                    last_used_at=last_used,
                    is_active=bool(row["is_active"])
                ))

            return tokens

    def revoke_token(self, token_id: int) -> bool:
        """Revoke a token by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "UPDATE tokens SET is_active = 0 WHERE id = ?",
                (token_id,)
            )
            if cursor.rowcount > 0:
                logger.info(f"Revoked token id={token_id}")
                return True
            return False

    def delete_token(self, token_id: int) -> bool:
        """Permanently delete a token."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM tokens WHERE id = ?",
                (token_id,)
            )
            if cursor.rowcount > 0:
                logger.info(f"Deleted token id={token_id}")
                return True
            return False

    def cleanup_expired(self) -> int:
        """Remove expired tokens. Returns count of removed tokens."""
        now = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM tokens WHERE expires_at IS NOT NULL AND expires_at < ?",
                (now,)
            )
            count = cursor.rowcount
            if count > 0:
                logger.info(f"Cleaned up {count} expired tokens")
            return count

    def get_token_count(self) -> dict:
        """Get token statistics."""
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM tokens").fetchone()[0]
            active = conn.execute("SELECT COUNT(*) FROM tokens WHERE is_active = 1").fetchone()[0]
            expired = conn.execute(
                "SELECT COUNT(*) FROM tokens WHERE expires_at IS NOT NULL AND expires_at < ?",
                (datetime.utcnow().isoformat(),)
            ).fetchone()[0]

            return {
                "total": total,
                "active": active,
                "inactive": total - active,
                "expired": expired
            }
