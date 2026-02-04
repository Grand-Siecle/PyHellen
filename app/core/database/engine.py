"""
SQLModel engine and session management.

Provides thread-safe database sessions and connection management.
"""

from contextlib import contextmanager
from typing import Generator, Optional
import threading

from sqlmodel import SQLModel, Session, create_engine, select
from sqlalchemy import event
from sqlalchemy.engine import Engine

from app.core.logger import logger


# Enable foreign keys for SQLite
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


class DatabaseEngine:
    """
    Singleton database engine manager for SQLModel.

    Handles connection pooling, schema creation, and provides
    session management for repositories.
    """

    _instance: Optional["DatabaseEngine"] = None
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
        self.db_path = db_path or settings.token_db_path

        # Create SQLite URL
        if self.db_path == ":memory:":
            # Use shared cache for in-memory database so all connections see same data
            db_url = "sqlite:///file::memory:?cache=shared&uri=true"
            connect_args = {"check_same_thread": False}
        else:
            db_url = f"sqlite:///{self.db_path}"
            connect_args = {"check_same_thread": False, "timeout": 30}

        self._engine = create_engine(
            db_url,
            echo=False,
            connect_args=connect_args,
        )

        self._initialized = True
        self._init_database()
        logger.info(f"DatabaseEngine initialized with database: {self.db_path}")

    def _init_database(self):
        """Create all tables and apply migrations."""
        from app.core.database.models import (
            Model, ModelFile, ModelMetrics, CacheEntry,
            Token, RequestLog, AuditLog, AppState, SchemaMigration
        )

        # Create all tables
        SQLModel.metadata.create_all(self._engine)

        # Insert default models if not exist
        self._seed_default_models()

    def _seed_default_models(self):
        """Insert default builtin models."""
        from app.core.database.models import Model

        default_models = [
            {"code": "lasla", "name": "Classical Latin", "description": "Tagger for Classical Latin texts", "pie_module": "lasla", "priority": 1},
            {"code": "grc", "name": "Ancient Greek", "description": "Tagger for Ancient Greek texts", "pie_module": "grc", "priority": 2},
            {"code": "fro", "name": "Old French", "description": "Tagger for Old French texts", "pie_module": "fro", "priority": 3},
            {"code": "freem", "name": "Early Modern French", "description": "Tagger for Early Modern French texts", "pie_module": "freem", "priority": 4},
            {"code": "fr", "name": "Classical French", "description": "Tagger for Classical French texts", "pie_module": "fr", "priority": 5},
            {"code": "dum", "name": "Old Dutch", "description": "Tagger for Old Dutch texts", "pie_module": "dum", "priority": 6},
            {"code": "occ_cont", "name": "Occitan Contemporain", "description": "Tagger for Contemporary Occitan texts", "pie_module": "occ_cont", "priority": 7},
        ]

        with self.get_session() as session:
            for model_data in default_models:
                existing = session.exec(
                    select(Model).where(Model.code == model_data["code"])
                ).first()

                if not existing:
                    model = Model(
                        **model_data,
                        is_active=True,
                        is_builtin=True,
                    )
                    session.add(model)
            session.commit()

    @property
    def engine(self):
        """Get the SQLAlchemy engine."""
        return self._engine

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Context manager for database sessions.

        Provides automatic commit on success and rollback on failure.
        """
        session = Session(self._engine)
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_new_session(self) -> Session:
        """Get a new session (caller must manage lifecycle)."""
        return Session(self._engine)


# Global instance getter
_db_engine: Optional[DatabaseEngine] = None


def get_db_engine() -> DatabaseEngine:
    """Get or create the global DatabaseEngine instance."""
    global _db_engine
    if _db_engine is None:
        _db_engine = DatabaseEngine()
    return _db_engine


def get_session() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI routes.

    Usage:
        @app.get("/items")
        def get_items(session: Session = Depends(get_session)):
            ...
    """
    engine = get_db_engine()
    with engine.get_session() as session:
        yield session
