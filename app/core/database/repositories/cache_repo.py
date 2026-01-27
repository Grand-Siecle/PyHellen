"""Repository for persistent cache management using SQLModel."""

import hashlib
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlmodel import Session, select, func, col

from app.core.database.models import Model, CacheEntry
from app.core.database.repositories.base import BaseRepository
from app.core.logger import logger


class CacheRepository(BaseRepository):
    """
    Repository for persistent LRU cache with TTL.

    Replaces the in-memory cache with SQLite-backed storage,
    providing persistence across restarts.
    """

    def __init__(self, session: Optional[Session] = None, max_size: int = 1000, ttl_seconds: int = 3600):
        super().__init__(session)
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def _generate_key(model: str, text: str, lower: bool) -> str:
        """Generate a unique cache key from the input parameters."""
        content = f"{model}:{text}:{lower}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    @staticmethod
    def _generate_text_hash(text: str) -> str:
        """Generate a hash of the text for storage."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def get(self, model_code: str, text: str, lower: bool) -> Optional[Any]:
        """
        Get a value from the cache.

        Returns cached value if found and not expired, None otherwise.
        Updates hit count on successful retrieval.
        """
        cache_key = self._generate_key(model_code, text, lower)
        now = datetime.utcnow()

        session = self._get_session()
        try:
            entry = session.exec(
                select(CacheEntry).where(CacheEntry.cache_key == cache_key)
            ).first()

            if not entry:
                return None

            # Check expiration
            if now > entry.expires_at:
                session.delete(entry)
                session.commit()
                return None

            # Update hit count and last_hit_at
            entry.hit_count += 1
            entry.last_hit_at = now
            session.add(entry)
            session.commit()

            return json.loads(entry.result_json)
        finally:
            self._close_session(session)

    def set(self, model_code: str, text: str, lower: bool, value: Any) -> bool:
        """
        Set a value in the cache.

        Implements LRU eviction when max_size is reached.
        """
        cache_key = self._generate_key(model_code, text, lower)
        text_hash = self._generate_text_hash(text)
        text_preview = text[:100] if len(text) > 100 else text
        result_json = json.dumps(value)
        size_bytes = len(result_json.encode())
        now = datetime.utcnow()
        expires_at = now + timedelta(seconds=self.ttl_seconds)

        session = self._get_session()
        try:
            # Get model
            model = session.exec(
                select(Model).where(Model.code == model_code)
            ).first()

            if not model:
                logger.warning(f"Cannot cache: model '{model_code}' not found")
                return False

            # Check current cache size and evict if necessary
            count = session.exec(select(func.count(CacheEntry.id))).one()

            if count >= self.max_size:
                # Delete oldest entries (LRU based on last_hit_at, then created_at)
                entries_to_delete = count - self.max_size + 1
                old_entries = session.exec(
                    select(CacheEntry)
                    .order_by(col(CacheEntry.last_hit_at).asc().nullsfirst(), CacheEntry.created_at.asc())
                    .limit(entries_to_delete)
                ).all()

                for old_entry in old_entries:
                    session.delete(old_entry)

            # Check if entry already exists
            existing = session.exec(
                select(CacheEntry).where(CacheEntry.cache_key == cache_key)
            ).first()

            if existing:
                existing.result_json = result_json
                existing.expires_at = expires_at
                existing.size_bytes = size_bytes
                existing.hit_count = 0
                existing.last_hit_at = None
                session.add(existing)
            else:
                entry = CacheEntry(
                    cache_key=cache_key,
                    model_id=model.id,
                    text_hash=text_hash,
                    text_preview=text_preview,
                    result_json=result_json,
                    created_at=now,
                    expires_at=expires_at,
                    size_bytes=size_bytes,
                )
                session.add(entry)

            session.commit()
            return True
        finally:
            self._close_session(session)

    def delete(self, model_code: str, text: str, lower: bool) -> bool:
        """Delete a specific cache entry."""
        cache_key = self._generate_key(model_code, text, lower)

        session = self._get_session()
        try:
            entry = session.exec(
                select(CacheEntry).where(CacheEntry.cache_key == cache_key)
            ).first()

            if entry:
                session.delete(entry)
                session.commit()
                return True
            return False
        finally:
            self._close_session(session)

    def clear(self) -> int:
        """Clear all cache entries. Returns number of cleared entries."""
        session = self._get_session()
        try:
            entries = list(session.exec(select(CacheEntry)).all())
            count = len(entries)

            for entry in entries:
                session.delete(entry)

            session.commit()
            logger.info(f"Cache cleared: {count} entries removed")
            return count
        finally:
            self._close_session(session)

    def clear_by_model(self, model_code: str) -> int:
        """Clear all cache entries for a specific model."""
        session = self._get_session()
        try:
            model = session.exec(
                select(Model).where(Model.code == model_code)
            ).first()

            if not model:
                return 0

            entries = list(session.exec(
                select(CacheEntry).where(CacheEntry.model_id == model.id)
            ).all())

            count = len(entries)
            for entry in entries:
                session.delete(entry)

            session.commit()
            if count > 0:
                logger.info(f"Cleared {count} cache entries for model '{model_code}'")
            return count
        finally:
            self._close_session(session)

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns number of removed entries."""
        now = datetime.utcnow()

        session = self._get_session()
        try:
            expired = list(session.exec(
                select(CacheEntry).where(CacheEntry.expires_at < now)
            ).all())

            count = len(expired)
            for entry in expired:
                session.delete(entry)

            session.commit()
            if count > 0:
                logger.info(f"Cache cleanup: {count} expired entries removed")
            return count
        finally:
            self._close_session(session)

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        session = self._get_session()
        try:
            total = session.exec(select(func.count(CacheEntry.id))).one()
            total_hits = session.exec(
                select(func.coalesce(func.sum(CacheEntry.hit_count), 0))
            ).one()
            total_size = session.exec(
                select(func.coalesce(func.sum(CacheEntry.size_bytes), 0))
            ).one()

            # Per-model stats
            model_stats = session.exec(
                select(Model.code, func.count(CacheEntry.id), func.sum(CacheEntry.hit_count))
                .join(CacheEntry)
                .group_by(Model.code)
            ).all()

            return {
                "size": total,
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "total_hits": total_hits,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "models": {
                    code: {"entries": count, "hits": hits or 0}
                    for code, count, hits in model_stats
                }
            }
        finally:
            self._close_session(session)

    def get_entries(self, limit: int = 100, offset: int = 0) -> List[CacheEntry]:
        """Get cache entries with pagination."""
        session = self._get_session()
        try:
            return list(session.exec(
                select(CacheEntry)
                .order_by(col(CacheEntry.last_hit_at).desc().nullslast(), CacheEntry.created_at.desc())
                .offset(offset)
                .limit(limit)
            ).all())
        finally:
            self._close_session(session)
