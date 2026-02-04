"""Repository for audit trail logging using SQLModel."""

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlmodel import Session, select, func, col

from app.core.database.models import AuditLog
from app.core.database.repositories.base import BaseRepository
from app.core.logger import logger


class AuditAction:
    """Standard audit action constants."""
    # Token actions
    TOKEN_CREATED = "token.created"
    TOKEN_REVOKED = "token.revoked"
    TOKEN_DELETED = "token.deleted"
    TOKEN_VALIDATED = "token.validated"
    TOKEN_EXPIRED_CLEANUP = "token.expired_cleanup"

    # Authentication actions
    AUTH_SUCCESS = "auth.success"
    AUTH_FAILED = "auth.failed"
    AUTH_INVALID_TOKEN = "auth.invalid_token"
    AUTH_EXPIRED_TOKEN = "auth.expired_token"
    AUTH_INSUFFICIENT_SCOPE = "auth.insufficient_scope"

    # Model actions
    MODEL_CREATED = "model.created"
    MODEL_UPDATED = "model.updated"
    MODEL_DELETED = "model.deleted"
    MODEL_ACTIVATED = "model.activated"
    MODEL_DEACTIVATED = "model.deactivated"
    MODEL_LOADED = "model.loaded"
    MODEL_UNLOADED = "model.unloaded"
    MODEL_DOWNLOADED = "model.downloaded"

    # Cache actions
    CACHE_CLEARED = "cache.cleared"
    CACHE_CLEANUP = "cache.cleanup"

    # Admin actions
    ADMIN_LOGIN = "admin.login"
    ADMIN_ACTION = "admin.action"


class AuditRepository(BaseRepository):
    """
    Repository for audit trail logging.

    Provides comprehensive logging of security-relevant events
    for compliance and debugging purposes.
    """

    def log(
        self,
        action: str,
        actor_token_id: Optional[int] = None,
        target_type: Optional[str] = None,
        target_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        client_ip: Optional[str] = None,
        success: bool = True
    ) -> int:
        """
        Log an audit event.

        Returns the ID of the created audit entry.
        """
        details_json = json.dumps(details) if details else None

        session = self._get_session()
        try:
            entry = AuditLog(
                action=action,
                actor_token_id=actor_token_id,
                target_type=target_type,
                target_id=target_id,
                details_json=details_json,
                client_ip=client_ip,
                success=success,
            )
            session.add(entry)
            session.commit()
            session.refresh(entry)
            return entry.id
        finally:
            self._close_session(session)

    def get_recent(self, limit: int = 100, offset: int = 0) -> List[AuditLog]:
        """Get recent audit entries."""
        session = self._get_session()
        try:
            return list(session.exec(
                select(AuditLog)
                .order_by(col(AuditLog.timestamp).desc())
                .offset(offset)
                .limit(limit)
            ).all())
        finally:
            self._close_session(session)

    def get_by_action(self, action: str, limit: int = 100) -> List[AuditLog]:
        """Get audit entries by action type."""
        session = self._get_session()
        try:
            return list(session.exec(
                select(AuditLog)
                .where(AuditLog.action == action)
                .order_by(col(AuditLog.timestamp).desc())
                .limit(limit)
            ).all())
        finally:
            self._close_session(session)

    def get_by_actor(self, token_id: int, limit: int = 100) -> List[AuditLog]:
        """Get audit entries by actor token ID."""
        session = self._get_session()
        try:
            return list(session.exec(
                select(AuditLog)
                .where(AuditLog.actor_token_id == token_id)
                .order_by(col(AuditLog.timestamp).desc())
                .limit(limit)
            ).all())
        finally:
            self._close_session(session)

    def get_by_target(
        self,
        target_type: str,
        target_id: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditLog]:
        """Get audit entries by target type and optionally ID."""
        session = self._get_session()
        try:
            query = select(AuditLog).where(AuditLog.target_type == target_type)
            if target_id:
                query = query.where(AuditLog.target_id == target_id)
            query = query.order_by(col(AuditLog.timestamp).desc()).limit(limit)

            return list(session.exec(query).all())
        finally:
            self._close_session(session)

    def get_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        limit: int = 1000
    ) -> List[AuditLog]:
        """Get audit entries within a date range."""
        session = self._get_session()
        try:
            return list(session.exec(
                select(AuditLog)
                .where(AuditLog.timestamp >= start_date, AuditLog.timestamp <= end_date)
                .order_by(col(AuditLog.timestamp).desc())
                .limit(limit)
            ).all())
        finally:
            self._close_session(session)

    def get_failed_actions(self, limit: int = 100) -> List[AuditLog]:
        """Get failed audit entries."""
        session = self._get_session()
        try:
            return list(session.exec(
                select(AuditLog)
                .where(AuditLog.success == False)
                .order_by(col(AuditLog.timestamp).desc())
                .limit(limit)
            ).all())
        finally:
            self._close_session(session)

    def get_auth_failures(self, hours: int = 24, limit: int = 100) -> List[AuditLog]:
        """Get authentication failures in the last N hours."""
        since = datetime.utcnow() - timedelta(hours=hours)

        session = self._get_session()
        try:
            return list(session.exec(
                select(AuditLog)
                .where(
                    AuditLog.action.like("auth.%"),
                    AuditLog.success == False,
                    AuditLog.timestamp >= since
                )
                .order_by(col(AuditLog.timestamp).desc())
                .limit(limit)
            ).all())
        finally:
            self._close_session(session)

    def get_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get audit statistics for the last N hours."""
        since = datetime.utcnow() - timedelta(hours=hours)

        session = self._get_session()
        try:
            total = session.exec(
                select(func.count(AuditLog.id)).where(AuditLog.timestamp >= since)
            ).one()

            failed = session.exec(
                select(func.count(AuditLog.id)).where(
                    AuditLog.timestamp >= since,
                    AuditLog.success == False
                )
            ).one()

            # Actions breakdown
            action_counts = session.exec(
                select(AuditLog.action, func.count(AuditLog.id))
                .where(AuditLog.timestamp >= since)
                .group_by(AuditLog.action)
                .order_by(func.count(AuditLog.id).desc())
            ).all()

            # Auth failures by IP
            ip_failures = session.exec(
                select(AuditLog.client_ip, func.count(AuditLog.id))
                .where(
                    AuditLog.timestamp >= since,
                    AuditLog.action.like("auth.%"),
                    AuditLog.success == False,
                    AuditLog.client_ip.is_not(None)
                )
                .group_by(AuditLog.client_ip)
                .order_by(func.count(AuditLog.id).desc())
                .limit(10)
            ).all()

            return {
                "period_hours": hours,
                "total_events": total,
                "failed_events": failed,
                "success_rate": round((total - failed) / total * 100, 2) if total > 0 else 100,
                "actions": {action: count for action, count in action_counts},
                "top_failing_ips": {ip: count for ip, count in ip_failures if ip},
            }
        finally:
            self._close_session(session)

    def cleanup(self, days: int = 90) -> int:
        """
        Remove audit entries older than N days.

        Returns number of deleted entries.
        """
        cutoff = datetime.utcnow() - timedelta(days=days)

        session = self._get_session()
        try:
            old_entries = list(session.exec(
                select(AuditLog).where(AuditLog.timestamp < cutoff)
            ).all())

            count = len(old_entries)
            for entry in old_entries:
                session.delete(entry)

            session.commit()
            if count > 0:
                logger.info(f"Cleaned up {count} audit entries older than {days} days")
            return count
        finally:
            self._close_session(session)

    def count(self) -> int:
        """Get total number of audit entries."""
        session = self._get_session()
        try:
            return session.exec(select(func.count(AuditLog.id))).one()
        finally:
            self._close_session(session)
