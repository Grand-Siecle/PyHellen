"""Repository for model metrics using SQLModel."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlmodel import Session, select, func

from app.core.database.models import Model, ModelMetrics
from app.core.database.repositories.base import BaseRepository
from app.core.logger import logger


class MetricsRepository(BaseRepository):
    """
    Repository for persistent model metrics.

    Stores metrics in SQLite for persistence across restarts,
    providing historical performance data.
    """

    def _get_or_create_metrics(self, session: Session, model_code: str) -> Optional[ModelMetrics]:
        """Get or create metrics for a model."""
        model = session.exec(
            select(Model).where(Model.code == model_code)
        ).first()

        if not model:
            return None

        metrics = session.exec(
            select(ModelMetrics).where(ModelMetrics.model_id == model.id)
        ).first()

        if not metrics:
            metrics = ModelMetrics(model_id=model.id)
            session.add(metrics)
            session.commit()
            session.refresh(metrics)

        return metrics

    def get_or_create(self, model_code: str) -> Optional[ModelMetrics]:
        """Get metrics for a model, creating if not exists."""
        session = self._get_session()
        try:
            return self._get_or_create_metrics(session, model_code)
        finally:
            self._close_session(session)

    def get_by_model(self, model_code: str) -> Optional[ModelMetrics]:
        """Get metrics for a specific model."""
        session = self._get_session()
        try:
            model = session.exec(
                select(Model).where(Model.code == model_code)
            ).first()

            if not model:
                return None

            return session.exec(
                select(ModelMetrics).where(ModelMetrics.model_id == model.id)
            ).first()
        finally:
            self._close_session(session)

    def get_all(self) -> Dict[str, ModelMetrics]:
        """Get metrics for all models."""
        session = self._get_session()
        try:
            # Join metrics with models to get codes
            results = session.exec(
                select(ModelMetrics, Model.code).join(Model)
            ).all()

            return {code: metrics for metrics, code in results}
        finally:
            self._close_session(session)

    def update_load_metrics(self, model_code: str, load_time_ms: float) -> bool:
        """Update metrics after a model load."""
        session = self._get_session()
        try:
            metrics = self._get_or_create_metrics(session, model_code)
            if not metrics:
                return False

            metrics.load_count += 1
            metrics.load_time_total_ms += load_time_ms
            metrics.last_loaded_at = datetime.utcnow()
            metrics.last_used_at = datetime.utcnow()
            metrics.updated_at = datetime.utcnow()

            session.add(metrics)
            session.commit()
            return True
        finally:
            self._close_session(session)

    def update_process_metrics(self, model_code: str, process_time_ms: float) -> bool:
        """Update metrics after processing text."""
        session = self._get_session()
        try:
            metrics = self._get_or_create_metrics(session, model_code)
            if not metrics:
                return False

            metrics.process_count += 1
            metrics.process_time_total_ms += process_time_ms
            metrics.last_used_at = datetime.utcnow()
            metrics.updated_at = datetime.utcnow()

            session.add(metrics)
            session.commit()
            return True
        finally:
            self._close_session(session)

    def update_download_metrics(
        self,
        model_code: str,
        download_time_ms: float,
        download_bytes: int
    ) -> bool:
        """Update metrics after downloading model files."""
        session = self._get_session()
        try:
            metrics = self._get_or_create_metrics(session, model_code)
            if not metrics:
                return False

            metrics.download_count += 1
            metrics.download_time_total_ms += download_time_ms
            metrics.download_size_bytes += download_bytes
            metrics.updated_at = datetime.utcnow()

            session.add(metrics)
            session.commit()
            return True
        finally:
            self._close_session(session)

    def increment_error(self, model_code: str) -> bool:
        """Increment error count for a model."""
        session = self._get_session()
        try:
            metrics = self._get_or_create_metrics(session, model_code)
            if not metrics:
                return False

            metrics.error_count += 1
            metrics.updated_at = datetime.utcnow()

            session.add(metrics)
            session.commit()
            return True
        finally:
            self._close_session(session)

    def update_last_used(self, model_code: str) -> bool:
        """Update last_used_at timestamp for a model."""
        session = self._get_session()
        try:
            metrics = self._get_or_create_metrics(session, model_code)
            if not metrics:
                return False

            metrics.last_used_at = datetime.utcnow()
            metrics.updated_at = datetime.utcnow()

            session.add(metrics)
            session.commit()
            return True
        finally:
            self._close_session(session)

    def get_global_statistics(self) -> Dict[str, Any]:
        """Get global statistics across all models."""
        session = self._get_session()
        try:
            result = session.exec(
                select(
                    func.coalesce(func.sum(ModelMetrics.load_count), 0),
                    func.coalesce(func.sum(ModelMetrics.process_count), 0),
                    func.coalesce(func.sum(ModelMetrics.download_count), 0),
                    func.coalesce(func.sum(ModelMetrics.download_size_bytes), 0),
                    func.coalesce(func.sum(ModelMetrics.error_count), 0),
                )
            ).first()

            total_loads = result[0]
            total_processed = result[1]
            total_downloads = result[2]
            total_download_bytes = result[3]
            total_errors = result[4]

            return {
                "total_loads": total_loads,
                "total_processed": total_processed,
                "total_downloads": total_downloads,
                "total_download_mb": round(total_download_bytes / (1024 * 1024), 2),
                "total_errors": total_errors,
            }
        finally:
            self._close_session(session)

    def reset(self, model_code: str) -> bool:
        """Reset metrics for a specific model."""
        session = self._get_session()
        try:
            model = session.exec(
                select(Model).where(Model.code == model_code)
            ).first()

            if not model:
                return False

            metrics = session.exec(
                select(ModelMetrics).where(ModelMetrics.model_id == model.id)
            ).first()

            if metrics:
                session.delete(metrics)
                session.commit()

            logger.info(f"Reset metrics for model '{model_code}'")
            return True
        finally:
            self._close_session(session)

    def reset_all(self) -> int:
        """Reset all metrics. Returns number of reset models."""
        session = self._get_session()
        try:
            all_metrics = list(session.exec(select(ModelMetrics)).all())
            count = len(all_metrics)

            for metrics in all_metrics:
                session.delete(metrics)

            session.commit()
            logger.info(f"Reset metrics for {count} models")
            return count
        finally:
            self._close_session(session)
