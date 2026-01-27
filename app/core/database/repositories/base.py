"""Base repository class with SQLModel session management."""

from typing import Optional, TYPE_CHECKING

from sqlmodel import Session

if TYPE_CHECKING:
    from app.core.database.engine import DatabaseEngine


class BaseRepository:
    """
    Base class for all repositories.

    Provides common functionality and session management.
    """

    def __init__(self, session: Optional[Session] = None):
        """
        Initialize repository.

        Args:
            session: Optional session to use. If None, will create sessions as needed.
        """
        self._session = session
        self._engine = None

    def _get_engine(self):
        """Lazy load database engine."""
        if self._engine is None:
            from app.core.database.engine import get_db_engine
            self._engine = get_db_engine()
        return self._engine

    def _get_session(self) -> Session:
        """Get a session for operations."""
        if self._session:
            return self._session
        return self._get_engine().get_new_session()

    def _close_session(self, session: Session):
        """Close session if we created it."""
        if session != self._session:
            session.close()
