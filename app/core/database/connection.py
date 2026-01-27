"""
Unified SQLite database connection manager.
Provides thread-safe connection handling and schema migrations.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Generator
import threading

from app.core.logger import logger


class DatabaseManager:
    """
    Singleton database manager for SQLite.

    Handles connection pooling, schema migrations, and provides
    a unified interface for all repositories.
    """

    _instance: Optional["DatabaseManager"] = None
    _lock = threading.Lock()

    def __new__(cls, db_path: Optional[str] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, db_path: Optional[str] = None):
        if self._initialized:
            return

        from app.core.settings import settings
        self.db_path = Path(db_path or settings.token_db_path)
        self._local = threading.local()
        self._initialized = True
        self._schema_version = 1

        # Initialize database schema
        self._init_database()
        logger.info(f"DatabaseManager initialized with database: {self.db_path}")

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager for database connections.

        Provides automatic commit on success and rollback on failure.
        Uses thread-local connections for thread safety.
        """
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")

        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_database(self):
        """Initialize database schema with all tables."""
        with self.get_connection() as conn:
            # Schema version tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL,
                    description TEXT
                )
            """)

            # Check current version
            current_version = conn.execute(
                "SELECT MAX(version) FROM schema_migrations"
            ).fetchone()[0] or 0

            if current_version < self._schema_version:
                self._apply_migrations(conn, current_version)

    def _apply_migrations(self, conn: sqlite3.Connection, from_version: int):
        """Apply database migrations from current version."""
        migrations = self._get_migrations()

        for version, (description, sql_statements) in migrations.items():
            if version > from_version:
                logger.info(f"Applying migration v{version}: {description}")
                for sql in sql_statements:
                    conn.execute(sql)
                conn.execute(
                    "INSERT INTO schema_migrations (version, applied_at, description) VALUES (?, datetime('now'), ?)",
                    (version, description)
                )

        logger.info(f"Database migrated to version {self._schema_version}")

    def _get_migrations(self) -> dict:
        """Return all database migrations."""
        return {
            1: ("Initial schema", [
                # Tokens table (existing, preserved)
                """
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
                """,
                "CREATE INDEX IF NOT EXISTS idx_token_hash ON tokens(token_hash)",
                "CREATE INDEX IF NOT EXISTS idx_tokens_active ON tokens(is_active)",

                # Models table (replaces hardcoded PieLanguage enum)
                """
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    description TEXT,
                    pie_module TEXT NOT NULL,
                    is_active INTEGER DEFAULT 1,
                    is_builtin INTEGER DEFAULT 0,
                    batch_size INTEGER,
                    priority INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """,
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_models_code ON models(code)",
                "CREATE INDEX IF NOT EXISTS idx_models_active ON models(is_active)",

                # Model files metadata
                """
                CREATE TABLE IF NOT EXISTS model_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER NOT NULL,
                    filename TEXT NOT NULL,
                    url TEXT NOT NULL,
                    size_bytes INTEGER,
                    checksum TEXT,
                    is_downloaded INTEGER DEFAULT 0,
                    downloaded_at TEXT,
                    FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE,
                    UNIQUE(model_id, filename)
                )
                """,
                "CREATE INDEX IF NOT EXISTS idx_model_files_model ON model_files(model_id)",

                # Model metrics
                """
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER NOT NULL UNIQUE,
                    load_count INTEGER DEFAULT 0,
                    load_time_total_ms REAL DEFAULT 0,
                    last_loaded_at TEXT,
                    process_count INTEGER DEFAULT 0,
                    process_time_total_ms REAL DEFAULT 0,
                    last_used_at TEXT,
                    download_count INTEGER DEFAULT 0,
                    download_time_total_ms REAL DEFAULT 0,
                    download_size_bytes INTEGER DEFAULT 0,
                    error_count INTEGER DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE
                )
                """,

                # Cache entries
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key TEXT NOT NULL UNIQUE,
                    model_id INTEGER NOT NULL,
                    text_hash TEXT NOT NULL,
                    text_preview TEXT,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    hit_count INTEGER DEFAULT 0,
                    last_hit_at TEXT,
                    size_bytes INTEGER DEFAULT 0,
                    FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE
                )
                """,
                "CREATE INDEX IF NOT EXISTS idx_cache_key ON cache_entries(cache_key)",
                "CREATE INDEX IF NOT EXISTS idx_cache_model ON cache_entries(model_id)",
                "CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_entries(expires_at)",

                # Request log
                """
                CREATE TABLE IF NOT EXISTS request_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_id INTEGER,
                    endpoint TEXT NOT NULL,
                    method TEXT NOT NULL,
                    token_id INTEGER,
                    text_length INTEGER,
                    batch_size INTEGER,
                    processing_time_ms REAL,
                    from_cache INTEGER DEFAULT 0,
                    status_code INTEGER NOT NULL,
                    error_message TEXT,
                    client_ip TEXT,
                    FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE SET NULL,
                    FOREIGN KEY (token_id) REFERENCES tokens(id) ON DELETE SET NULL
                )
                """,
                "CREATE INDEX IF NOT EXISTS idx_request_timestamp ON request_log(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_request_model ON request_log(model_id)",

                # Audit log
                """
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    action TEXT NOT NULL,
                    actor_token_id INTEGER,
                    target_type TEXT,
                    target_id TEXT,
                    details_json TEXT,
                    client_ip TEXT,
                    success INTEGER DEFAULT 1,
                    FOREIGN KEY (actor_token_id) REFERENCES tokens(id) ON DELETE SET NULL
                )
                """,
                "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action)",

                # App state (key-value store)
                """
                CREATE TABLE IF NOT EXISTS app_state (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """,

                # Insert default models (builtin)
                """
                INSERT OR IGNORE INTO models (code, name, description, pie_module, is_active, is_builtin, priority, created_at, updated_at)
                VALUES
                    ('lasla', 'Classical Latin', 'Tagger for Classical Latin texts', 'lasla', 1, 1, 1, datetime('now'), datetime('now')),
                    ('grc', 'Ancient Greek', 'Tagger for Ancient Greek texts', 'grc', 1, 1, 2, datetime('now'), datetime('now')),
                    ('fro', 'Old French', 'Tagger for Old French texts', 'fro', 1, 1, 3, datetime('now'), datetime('now')),
                    ('freem', 'Early Modern French', 'Tagger for Early Modern French texts', 'freem', 1, 1, 4, datetime('now'), datetime('now')),
                    ('fr', 'Classical French', 'Tagger for Classical French texts', 'fr', 1, 1, 5, datetime('now'), datetime('now')),
                    ('dum', 'Old Dutch', 'Tagger for Old Dutch texts', 'dum', 1, 1, 6, datetime('now'), datetime('now')),
                    ('occ_cont', 'Occitan Contemporain', 'Tagger for Contemporary Occitan texts', 'occ_cont', 1, 1, 7, datetime('now'), datetime('now'))
                """,
            ])
        }

    def get_schema_version(self) -> int:
        """Get current schema version."""
        with self.get_connection() as conn:
            result = conn.execute(
                "SELECT MAX(version) FROM schema_migrations"
            ).fetchone()[0]
            return result or 0


# Global instance getter
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get or create the global DatabaseManager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
