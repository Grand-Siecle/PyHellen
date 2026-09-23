"""Repository for persistent cache management using SQLModel."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, col, delete, func, select, update

from app.core.database.models import Model, CacheEntry
from app.core.database.repositories.base import BaseRepository
from app.core.logger import logger
from app.core.timeutils import ensure_utc, utcnow

# Rows per statement, well below SQLite's bound-parameter limit
_CHUNK_SIZE = 50

# Limits are checked after this fraction of max_size/max_bytes has been written since the last check:
# measuring the table reads every row (COUNT + SUM over large JSON rows), too costly on every write.
_LIMIT_CHECK_FRACTION = 0.01

_UPSERT_COLUMNS = (
    "model_id",
    "text_hash",
    "text_preview",
    "result_json",
    "created_at",
    "expires_at",
    "hit_count",
    "last_hit_at",
    "size_bytes",
)


@dataclass
class CacheRecord:
    """A serialized result ready to be persisted. Keys are computed by the caller (HybridCache)."""

    key: str
    result_json: str
    expires_at: float  # Unix timestamp
    text_hash: str = ""
    text_preview: Optional[str] = None


def _to_datetime(timestamp: float) -> datetime:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def _to_timestamp(value: datetime) -> float:
    return ensure_utc(value).timestamp()


def _chunks(items: Sequence, size: int = _CHUNK_SIZE):
    for start in range(0, len(items), size):
        yield items[start : start + size]


class CacheRepository(BaseRepository):
    """
    SQLite storage for cached tagging results.

    Works on pre-computed keys and JSON payloads so that all hashing and serialization stays in HybridCache.
    Methods are synchronous; HybridCache calls them from a worker thread.
    """

    def __init__(
        self,
        session: Optional[Session] = None,
        max_size: int = 1000,
        ttl_seconds: float = 3600,
        max_bytes: Optional[int] = None,
    ):
        super().__init__(session)
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.max_bytes = max_bytes
        self._model_ids: Dict[str, int] = {}
        # Start "due" so the first write enforces limits on a pre-existing table
        self._entries_since_check = max_size
        self._bytes_since_check = max_bytes or 0

    def get_many(self, keys: Sequence[str]) -> Dict[str, Tuple[str, float]]:
        """
        Return {key: (result_json, expires_at timestamp)} for the non-expired keys found.

        Hit counters of found entries are updated in the same transaction.
        """
        if not keys:
            return {}
        now = utcnow()
        session = self._get_session()
        try:
            found: Dict[str, Tuple[str, float]] = {}
            for chunk in _chunks(list(keys)):
                rows = session.exec(
                    select(CacheEntry.cache_key, CacheEntry.result_json, CacheEntry.expires_at).where(
                        col(CacheEntry.cache_key).in_(chunk), CacheEntry.expires_at > now
                    )
                ).all()
                for key, result_json, expires_at in rows:
                    found[key] = (result_json, _to_timestamp(expires_at))

            if found:
                for chunk in _chunks(list(found)):
                    session.exec(
                        update(CacheEntry)
                        .where(col(CacheEntry.cache_key).in_(chunk))
                        .values(hit_count=CacheEntry.hit_count + 1, last_hit_at=now)
                    )
                session.commit()
            return found
        finally:
            self._close_session(session)

    def set_many(self, model_code: str, records: Sequence[CacheRecord]) -> int:
        """
        Insert or replace records, then evict entries if the table exceeds its limits.

        Returns the number of stored records (0 if the model is unknown).
        """
        if not records:
            return 0
        try:
            return self._set_many(model_code, records)
        except IntegrityError:
            # The model was deleted and re-created with a new id since it was memoized
            self._model_ids.pop(model_code, None)
            return self._set_many(model_code, records)

    def _set_many(self, model_code: str, records: Sequence[CacheRecord]) -> int:
        session = self._get_session()
        try:
            model_id = self._get_model_id(session, model_code)
            if model_id is None:
                logger.warning(f"Cannot cache: model '{model_code}' not found")
                return 0

            now = utcnow()
            rows = [
                {
                    "cache_key": record.key,
                    "model_id": model_id,
                    "text_hash": record.text_hash,
                    "text_preview": record.text_preview,
                    "result_json": record.result_json,
                    "created_at": now,
                    "expires_at": _to_datetime(record.expires_at),
                    "hit_count": 0,
                    "last_hit_at": None,
                    "size_bytes": len(record.result_json),
                }
                for record in records
            ]
            for chunk in _chunks(rows):
                statement = sqlite_insert(CacheEntry).values(chunk)
                statement = statement.on_conflict_do_update(
                    index_elements=[CacheEntry.cache_key],
                    set_={column: statement.excluded[column] for column in _UPSERT_COLUMNS},
                )
                session.exec(statement)

            self._entries_since_check += len(rows)
            self._bytes_since_check += sum(row["size_bytes"] for row in rows)
            if self._limit_check_due():
                self._evict(session, now)
                self._entries_since_check = 0
                self._bytes_since_check = 0
            session.commit()
            return len(rows)
        except IntegrityError:
            session.rollback()
            raise
        finally:
            self._close_session(session)

    def _get_model_id(self, session: Session, model_code: str) -> Optional[int]:
        if model_code not in self._model_ids:
            model_id = session.exec(select(Model.id).where(Model.code == model_code)).first()
            if model_id is None:
                return None
            self._model_ids[model_code] = model_id
        return self._model_ids[model_code]

    def _limit_check_due(self) -> bool:
        entries_step = max(1, int(self.max_size * _LIMIT_CHECK_FRACTION))
        if self._entries_since_check >= entries_step:
            return True
        return self.max_bytes is not None and self._bytes_since_check >= self.max_bytes * _LIMIT_CHECK_FRACTION

    def _evict(self, session: Session, now: datetime) -> None:
        """Keep the table within max_size/max_bytes: drop expired entries first, then the least recently active."""
        count, total_bytes = self._size(session)
        if not self._over_limits(count, total_bytes):
            return

        session.exec(delete(CacheEntry).where(CacheEntry.expires_at <= now))
        count, total_bytes = self._size(session)
        if not self._over_limits(count, total_bytes):
            return

        excess_entries = count - self.max_size
        excess_bytes = total_bytes - self.max_bytes if self.max_bytes is not None else 0
        victims: List[int] = []
        oldest_first = session.exec(
            select(CacheEntry.id, CacheEntry.size_bytes).order_by(
                func.coalesce(CacheEntry.last_hit_at, CacheEntry.created_at).asc(), CacheEntry.id.asc()
            )
        )
        for entry_id, size_bytes in oldest_first:
            if excess_entries <= 0 and excess_bytes <= 0:
                break
            victims.append(entry_id)
            excess_entries -= 1
            excess_bytes -= size_bytes
        for chunk in _chunks(victims, 500):
            session.exec(delete(CacheEntry).where(col(CacheEntry.id).in_(chunk)))

    def _size(self, session: Session) -> Tuple[int, int]:
        return session.exec(select(func.count(CacheEntry.id), func.coalesce(func.sum(CacheEntry.size_bytes), 0))).one()

    def _over_limits(self, count: int, total_bytes: int) -> bool:
        return count > self.max_size or (self.max_bytes is not None and total_bytes > self.max_bytes)

    def delete(self, key: str) -> bool:
        """Delete a specific cache entry."""
        session = self._get_session()
        try:
            result = session.exec(delete(CacheEntry).where(CacheEntry.cache_key == key))
            session.commit()
            return result.rowcount > 0
        finally:
            self._close_session(session)

    def clear(self) -> int:
        """Clear all cache entries. Returns number of cleared entries."""
        session = self._get_session()
        try:
            count = session.exec(delete(CacheEntry)).rowcount
            session.commit()
            return count
        finally:
            self._close_session(session)

    def clear_by_model(self, model_code: str) -> int:
        """Clear all cache entries for a specific model."""
        session = self._get_session()
        try:
            model_id = session.exec(select(Model.id).where(Model.code == model_code)).first()
            if model_id is None:
                return 0
            count = session.exec(delete(CacheEntry).where(CacheEntry.model_id == model_id)).rowcount
            session.commit()
            return count
        finally:
            self._close_session(session)

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns number of removed entries."""
        session = self._get_session()
        try:
            count = session.exec(delete(CacheEntry).where(CacheEntry.expires_at <= utcnow())).rowcount
            session.commit()
            return count
        finally:
            self._close_session(session)

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        session = self._get_session()
        try:
            total, total_size = self._size(session)
            total_hits = session.exec(select(func.coalesce(func.sum(CacheEntry.hit_count), 0))).one()

            # Per-model stats
            model_stats = session.exec(
                select(Model.code, func.count(CacheEntry.id), func.sum(CacheEntry.hit_count))
                .join(CacheEntry)
                .group_by(Model.code)
            ).all()

            return {
                "size": total,
                "max_size": self.max_size,
                "max_bytes": self.max_bytes,
                "ttl_seconds": self.ttl_seconds,
                "total_hits": total_hits,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "models": {code: {"entries": count, "hits": hits or 0} for code, count, hits in model_stats},
            }
        finally:
            self._close_session(session)

    def get_entries(self, limit: int = 100, offset: int = 0) -> List[CacheEntry]:
        """Get cache entries with pagination."""
        session = self._get_session()
        try:
            return list(
                session.exec(
                    select(CacheEntry)
                    .order_by(col(CacheEntry.last_hit_at).desc().nullslast(), CacheEntry.created_at.desc())
                    .offset(offset)
                    .limit(limit)
                ).all()
            )
        finally:
            self._close_session(session)
