"""Repository for request logging using SQLModel."""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlmodel import Session, select, func, col

from app.core.database.models import Model, RequestLog
from app.core.database.repositories.base import BaseRepository
from app.core.logger import logger


class RequestLogRepository(BaseRepository):
    """
    Repository for request logging and analytics.

    Provides detailed request logging for analytics,
    performance monitoring, and debugging.
    """

    def log(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        model_code: Optional[str] = None,
        token_id: Optional[int] = None,
        text_length: Optional[int] = None,
        batch_size: Optional[int] = None,
        processing_time_ms: Optional[float] = None,
        from_cache: bool = False,
        error_message: Optional[str] = None,
        client_ip: Optional[str] = None
    ) -> int:
        """
        Log a request.

        Returns the ID of the created log entry.
        """
        model_id = None

        session = self._get_session()
        try:
            # Get model_id if model_code provided
            if model_code:
                model = session.exec(
                    select(Model).where(Model.code == model_code)
                ).first()
                if model:
                    model_id = model.id

            entry = RequestLog(
                model_id=model_id,
                endpoint=endpoint,
                method=method,
                token_id=token_id,
                text_length=text_length,
                batch_size=batch_size,
                processing_time_ms=processing_time_ms,
                from_cache=from_cache,
                status_code=status_code,
                error_message=error_message,
                client_ip=client_ip,
            )
            session.add(entry)
            session.commit()
            session.refresh(entry)
            return entry.id
        finally:
            self._close_session(session)

    def get_recent(self, limit: int = 100, offset: int = 0) -> List[RequestLog]:
        """Get recent request logs."""
        session = self._get_session()
        try:
            return list(session.exec(
                select(RequestLog)
                .order_by(col(RequestLog.timestamp).desc())
                .offset(offset)
                .limit(limit)
            ).all())
        finally:
            self._close_session(session)

    def get_by_model(self, model_code: str, limit: int = 100) -> List[RequestLog]:
        """Get request logs for a specific model."""
        session = self._get_session()
        try:
            model = session.exec(
                select(Model).where(Model.code == model_code)
            ).first()

            if not model:
                return []

            return list(session.exec(
                select(RequestLog)
                .where(RequestLog.model_id == model.id)
                .order_by(col(RequestLog.timestamp).desc())
                .limit(limit)
            ).all())
        finally:
            self._close_session(session)

    def get_by_token(self, token_id: int, limit: int = 100) -> List[RequestLog]:
        """Get request logs for a specific token."""
        session = self._get_session()
        try:
            return list(session.exec(
                select(RequestLog)
                .where(RequestLog.token_id == token_id)
                .order_by(col(RequestLog.timestamp).desc())
                .limit(limit)
            ).all())
        finally:
            self._close_session(session)

    def get_errors(self, limit: int = 100) -> List[RequestLog]:
        """Get error request logs (status >= 400)."""
        session = self._get_session()
        try:
            return list(session.exec(
                select(RequestLog)
                .where(RequestLog.status_code >= 400)
                .order_by(col(RequestLog.timestamp).desc())
                .limit(limit)
            ).all())
        finally:
            self._close_session(session)

    def get_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get request statistics for the last N hours."""
        since = datetime.utcnow() - timedelta(hours=hours)

        session = self._get_session()
        try:
            # Total requests
            total = session.exec(
                select(func.count(RequestLog.id)).where(RequestLog.timestamp >= since)
            ).one()

            # Successful requests (status < 400)
            successful = session.exec(
                select(func.count(RequestLog.id)).where(
                    RequestLog.timestamp >= since,
                    RequestLog.status_code < 400
                )
            ).one()

            # Cache hits
            cache_hits = session.exec(
                select(func.count(RequestLog.id)).where(
                    RequestLog.timestamp >= since,
                    RequestLog.from_cache == True
                )
            ).one()

            # Average processing time
            avg_time = session.exec(
                select(func.avg(RequestLog.processing_time_ms)).where(
                    RequestLog.timestamp >= since,
                    RequestLog.processing_time_ms.is_not(None)
                )
            ).one()

            # Requests by model
            model_stats = session.exec(
                select(
                    Model.code,
                    func.count(RequestLog.id),
                    func.avg(RequestLog.processing_time_ms)
                )
                .join(Model)
                .where(RequestLog.timestamp >= since)
                .group_by(Model.code)
                .order_by(func.count(RequestLog.id).desc())
            ).all()

            # Requests by endpoint
            endpoint_stats = session.exec(
                select(
                    RequestLog.endpoint,
                    RequestLog.method,
                    func.count(RequestLog.id)
                )
                .where(RequestLog.timestamp >= since)
                .group_by(RequestLog.endpoint, RequestLog.method)
                .order_by(func.count(RequestLog.id).desc())
            ).all()

            # Error breakdown
            error_stats = session.exec(
                select(RequestLog.status_code, func.count(RequestLog.id))
                .where(RequestLog.timestamp >= since, RequestLog.status_code >= 400)
                .group_by(RequestLog.status_code)
                .order_by(func.count(RequestLog.id).desc())
            ).all()

            return {
                "period_hours": hours,
                "total_requests": total,
                "successful_requests": successful,
                "error_requests": total - successful,
                "success_rate": round(successful / total * 100, 2) if total > 0 else 100,
                "cache_hits": cache_hits,
                "cache_hit_rate": round(cache_hits / total * 100, 2) if total > 0 else 0,
                "avg_processing_time_ms": round(avg_time, 2) if avg_time else 0,
                "requests_per_hour": round(total / hours, 2) if hours > 0 else 0,
                "models": {
                    code: {
                        "count": count,
                        "avg_time_ms": round(avg_t, 2) if avg_t else 0
                    }
                    for code, count, avg_t in model_stats
                },
                "endpoints": {
                    f"{method} {endpoint}": count
                    for endpoint, method, count in endpoint_stats
                },
                "errors": {
                    str(status): count
                    for status, count in error_stats
                },
            }
        finally:
            self._close_session(session)

    def get_model_statistics(self, model_code: str, hours: int = 24) -> Dict[str, Any]:
        """Get statistics for a specific model."""
        since = datetime.utcnow() - timedelta(hours=hours)

        session = self._get_session()
        try:
            model = session.exec(
                select(Model).where(Model.code == model_code)
            ).first()

            if not model:
                return {}

            stats = session.exec(
                select(
                    func.count(RequestLog.id),
                    func.sum(func.cast(RequestLog.status_code < 400, int)),
                    func.sum(func.cast(RequestLog.from_cache, int)),
                    func.avg(RequestLog.processing_time_ms),
                    func.min(RequestLog.processing_time_ms),
                    func.max(RequestLog.processing_time_ms),
                    func.avg(RequestLog.text_length),
                )
                .where(RequestLog.model_id == model.id, RequestLog.timestamp >= since)
            ).first()

            total = stats[0] or 0
            successful = stats[1] or 0
            cache_hits = stats[2] or 0
            avg_time = stats[3]
            min_time = stats[4]
            max_time = stats[5]
            avg_text_length = stats[6]

            return {
                "model": model_code,
                "period_hours": hours,
                "total_requests": total,
                "successful_requests": successful,
                "cache_hits": cache_hits,
                "cache_hit_rate": round(cache_hits / total * 100, 2) if total > 0 else 0,
                "avg_processing_time_ms": round(avg_time, 2) if avg_time else 0,
                "min_processing_time_ms": round(min_time, 2) if min_time else 0,
                "max_processing_time_ms": round(max_time, 2) if max_time else 0,
                "avg_text_length": round(avg_text_length, 0) if avg_text_length else 0,
            }
        finally:
            self._close_session(session)

    def cleanup(self, days: int = 30) -> int:
        """
        Remove request logs older than N days.

        Returns number of deleted entries.
        """
        cutoff = datetime.utcnow() - timedelta(days=days)

        session = self._get_session()
        try:
            old_entries = list(session.exec(
                select(RequestLog).where(RequestLog.timestamp < cutoff)
            ).all())

            count = len(old_entries)
            for entry in old_entries:
                session.delete(entry)

            session.commit()
            if count > 0:
                logger.info(f"Cleaned up {count} request log entries older than {days} days")
            return count
        finally:
            self._close_session(session)

    def count(self) -> int:
        """Get total number of request log entries."""
        session = self._get_session()
        try:
            return session.exec(select(func.count(RequestLog.id))).one()
        finally:
            self._close_session(session)
