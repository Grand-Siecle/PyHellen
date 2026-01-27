"""Repository for model management using SQLModel."""

from datetime import datetime
from typing import List, Optional

from sqlmodel import Session, select, func

from app.core.database.models import Model, ModelFile, ModelCreate, ModelUpdate
from app.core.database.engine import get_db_engine
from app.core.logger import logger


class ModelRepository:
    """
    Repository for managing NLP models.

    Replaces the hardcoded PieLanguage enum with database-driven
    model management, allowing dynamic addition/removal of models.
    """

    def __init__(self, session: Optional[Session] = None):
        """
        Initialize repository.

        Args:
            session: Optional session to use. If None, will create sessions as needed.
        """
        self._session = session
        self._engine = get_db_engine()

    def _get_session(self) -> Session:
        """Get a session for operations."""
        if self._session:
            return self._session
        return self._engine.get_new_session()

    def _close_session(self, session: Session):
        """Close session if we created it."""
        if session != self._session:
            session.close()

    def get_all(self, include_inactive: bool = False) -> List[Model]:
        """Get all models, optionally including inactive ones."""
        session = self._get_session()
        try:
            query = select(Model).order_by(Model.priority, Model.code)
            if not include_inactive:
                query = query.where(Model.is_active == True)
            return list(session.exec(query).all())
        finally:
            self._close_session(session)

    def get_by_code(self, code: str) -> Optional[Model]:
        """Get a model by its code."""
        session = self._get_session()
        try:
            return session.exec(
                select(Model).where(Model.code == code)
            ).first()
        finally:
            self._close_session(session)

    def get_by_id(self, model_id: int) -> Optional[Model]:
        """Get a model by its ID."""
        session = self._get_session()
        try:
            return session.get(Model, model_id)
        finally:
            self._close_session(session)

    def is_valid_model(self, code: str) -> bool:
        """Check if a model code is valid and active."""
        session = self._get_session()
        try:
            result = session.exec(
                select(Model.id).where(Model.code == code, Model.is_active == True)
            ).first()
            return result is not None
        finally:
            self._close_session(session)

    def get_active_codes(self) -> List[str]:
        """Get list of active model codes (for validation)."""
        session = self._get_session()
        try:
            results = session.exec(
                select(Model.code).where(Model.is_active == True).order_by(Model.priority)
            ).all()
            return list(results)
        finally:
            self._close_session(session)

    def create(
        self,
        code: str,
        name: str,
        pie_module: str,
        description: Optional[str] = None,
        batch_size: Optional[int] = None,
        priority: int = 100,
    ) -> Model:
        """Create a new model."""
        session = self._get_session()
        try:
            model = Model(
                code=code,
                name=name,
                description=description,
                pie_module=pie_module,
                batch_size=batch_size,
                priority=priority,
                is_active=True,
                is_builtin=False,
            )
            session.add(model)
            session.commit()
            session.refresh(model)
            logger.info(f"Created model '{code}' (id={model.id})")
            return model
        finally:
            self._close_session(session)

    def update(
        self,
        code: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        pie_module: Optional[str] = None,
        batch_size: Optional[int] = None,
        priority: Optional[int] = None,
    ) -> Optional[Model]:
        """Update a model's properties."""
        session = self._get_session()
        try:
            model = session.exec(
                select(Model).where(Model.code == code)
            ).first()

            if not model:
                return None

            if name is not None:
                model.name = name
            if description is not None:
                model.description = description
            if pie_module is not None:
                model.pie_module = pie_module
            if batch_size is not None:
                model.batch_size = batch_size
            if priority is not None:
                model.priority = priority

            model.updated_at = datetime.utcnow()
            session.add(model)
            session.commit()
            session.refresh(model)
            logger.info(f"Updated model '{code}'")
            return model
        finally:
            self._close_session(session)

    def activate(self, code: str) -> bool:
        """Activate a model."""
        session = self._get_session()
        try:
            model = session.exec(
                select(Model).where(Model.code == code)
            ).first()

            if not model:
                return False

            model.is_active = True
            model.updated_at = datetime.utcnow()
            session.add(model)
            session.commit()
            logger.info(f"Activated model '{code}'")
            return True
        finally:
            self._close_session(session)

    def deactivate(self, code: str) -> bool:
        """Deactivate a model."""
        session = self._get_session()
        try:
            model = session.exec(
                select(Model).where(Model.code == code)
            ).first()

            if not model:
                return False

            model.is_active = False
            model.updated_at = datetime.utcnow()
            session.add(model)
            session.commit()
            logger.info(f"Deactivated model '{code}'")
            return True
        finally:
            self._close_session(session)

    def delete(self, code: str) -> bool:
        """
        Delete a model (only non-builtin models can be deleted).

        Returns False if model doesn't exist or is builtin.
        """
        session = self._get_session()
        try:
            model = session.exec(
                select(Model).where(Model.code == code)
            ).first()

            if not model:
                return False

            if model.is_builtin:
                logger.warning(f"Cannot delete builtin model '{code}'")
                return False

            session.delete(model)
            session.commit()
            logger.info(f"Deleted model '{code}'")
            return True
        finally:
            self._close_session(session)

    def get_model_files(self, code: str) -> List[ModelFile]:
        """Get all files for a model."""
        session = self._get_session()
        try:
            model = session.exec(
                select(Model).where(Model.code == code)
            ).first()

            if not model:
                return []

            return list(session.exec(
                select(ModelFile).where(ModelFile.model_id == model.id)
            ).all())
        finally:
            self._close_session(session)

    def add_model_file(
        self,
        code: str,
        filename: str,
        url: str,
        size_bytes: Optional[int] = None,
        checksum: Optional[str] = None,
    ) -> Optional[ModelFile]:
        """Add a file to a model."""
        session = self._get_session()
        try:
            model = session.exec(
                select(Model).where(Model.code == code)
            ).first()

            if not model:
                return None

            # Check if file already exists
            existing = session.exec(
                select(ModelFile).where(
                    ModelFile.model_id == model.id,
                    ModelFile.filename == filename
                )
            ).first()

            if existing:
                existing.url = url
                existing.size_bytes = size_bytes
                existing.checksum = checksum
                session.add(existing)
                session.commit()
                session.refresh(existing)
                return existing

            model_file = ModelFile(
                model_id=model.id,
                filename=filename,
                url=url,
                size_bytes=size_bytes,
                checksum=checksum,
            )
            session.add(model_file)
            session.commit()
            session.refresh(model_file)
            return model_file
        finally:
            self._close_session(session)

    def mark_file_downloaded(
        self,
        code: str,
        filename: str,
        size_bytes: Optional[int] = None
    ) -> bool:
        """Mark a model file as downloaded."""
        session = self._get_session()
        try:
            model = session.exec(
                select(Model).where(Model.code == code)
            ).first()

            if not model:
                return False

            model_file = session.exec(
                select(ModelFile).where(
                    ModelFile.model_id == model.id,
                    ModelFile.filename == filename
                )
            ).first()

            if not model_file:
                return False

            model_file.is_downloaded = True
            model_file.downloaded_at = datetime.utcnow()
            if size_bytes is not None:
                model_file.size_bytes = size_bytes

            session.add(model_file)
            session.commit()
            return True
        finally:
            self._close_session(session)

    def get_statistics(self) -> dict:
        """Get model statistics."""
        session = self._get_session()
        try:
            total = session.exec(select(func.count(Model.id))).one()
            active = session.exec(
                select(func.count(Model.id)).where(Model.is_active == True)
            ).one()
            builtin = session.exec(
                select(func.count(Model.id)).where(Model.is_builtin == True)
            ).one()

            return {
                "total": total,
                "active": active,
                "inactive": total - active,
                "builtin": builtin,
                "custom": total - builtin,
            }
        finally:
            self._close_session(session)
