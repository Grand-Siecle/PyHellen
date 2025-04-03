from typing import Dict, Optional, List, Union, Callable, Tuple
import asyncio
import importlib
import requests
from pie_extended.cli.utils import get_model, get_tagger
from fastapi import HTTPException

from app.core.utils import get_path_models, get_device
from app.core.logger import logger
from app.schemas.nlp import PieLanguage, ModelStatusSchema


class ModelManager:
    """
    Manages the lifecycle of NLP models including loading, downloading, and tracking status.
    Provides both synchronous and asynchronous interfaces for model operations.
    """

    def __init__(self):
        self.taggers: Dict[str, object] = {}
        self.iterator_processors: Dict[str, Callable[[], Tuple]] = {}
        self.download_locks: Dict[str, asyncio.Lock] = {}
        self.is_downloading: Dict[str, bool] = {}
        logger.info(f"ModelManager initialized with device: {self.device}")

    @property
    def device(self) -> str:
        """Get device disponibility"""
        return get_device()

    @property
    def batch_size(self) -> int:
        """Batch size"""
        return getattr(self, "_batch_size", 256)

    @batch_size.setter
    def batch_size(self, value: int):
        """Batch size check error"""
        self._batch_size = max(1, value)

    def get_model_status(self, model_name: str) -> str:
        """
        Get the current status of a model.
        """
        if self.taggers.get(model_name):
            return "loaded"
        if self.is_downloading.get(model_name, False):
            return "downloading"
        return "not loaded" if self._is_model_available(model_name) else "not available"

    def _is_model_available(self, model_name: str) -> bool:
        """
        Check if a model is available for download.
        """
        try:
            return get_model(model_name) is not None
        except ImportError:
            return False

    def get_all_models_status(self) -> Dict[str, ModelStatusSchema]:
        """
        Get status information for all supported models.
        """
        models_status = {}
        for language in PieLanguage:
            status = self.get_model_status(language.name)
            models_status[language.name] = ModelStatusSchema(
                language=language.value,
                status=status,
                files=self._get_model_files(language.name) if status in ["loaded", "not loaded"] else None
            )
        return models_status

    def _get_model_files(self, model_name: str) -> Optional[List[str]]:
        """
        Get a list of files associated with a model.
        """
        try:
            module_info = get_model(model_name)
            if module_info and hasattr(module_info, 'DOWNLOADS'):
                return [file.name for file in module_info.DOWNLOADS]
            return None
        except Exception:
            return None

    async def task_download(self, url: str, filename: str):
        """
        Efficiently download a file in chunks without Rich progress feedback.
        """
        response = requests.get(url, stream=True)
        total = int(response.headers.get('content-length', 0))

        if total == 0:
            raise RuntimeError(f"Failed to retrieve content length for {url}")

        logger.info(f"Downloading {filename} ({total / (1024 * 1024):.2f} MB)...")

        with open(filename, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    progress = (downloaded / total) * 100
                    logger.info(f"Progress: {progress:.1f}%")

        logger.info(f"✅ {filename} downloaded successfully.")

    async def download_model(self, module: str) -> bool:
        """
        Download model files for the specified module.
        """
        if self.download_locks.get(module) is None:
            self.download_locks[module] = asyncio.Lock()

        async with self.download_locks[module]:
            if self.is_downloading.get(module, False):
                logger.warning(f"⚠️ Download for '{module}' already in progress...")
                return False

            try:
                self.is_downloading[module] = True
                module_info = get_model(module)
                if module_info is None:
                    raise ValueError(f"Module '{module}' not found")

                if not hasattr(module_info, 'DOWNLOADS') or not module_info.DOWNLOADS:
                    raise ValueError(f"No files available for download in module '{module}'")

                import os
                os.makedirs(get_path_models(module, ""), exist_ok=True)

                total_files = len(module_info.DOWNLOADS)
                logger.info(f"Starting download of {total_files} files for module '{module}'...")

                downloaded_files = []
                for file in module_info.DOWNLOADS:
                    file_path = get_path_models(module, file.name)
                    logger.info(f"→ Downloading: {file.name}")

                    await self.task_download(file.url, file_path)
                    downloaded_files.append(file.name)

                logger.info(f"✅ Download completed: {', '.join(downloaded_files)}")
                return True

            except Exception as e:
                logger.error(f"❌ Error downloading module '{module}': {e}")
                return False

            finally:
                self.is_downloading[module] = False

    async def get_or_load_model(self, module: str) -> object:
        """
        Get a loaded tagger or load it if not already loaded.
        If not available locally, download it first.
        """
        # Check if already loaded
        if module in self.taggers and self.taggers[module]:
            return self.taggers[module]

        # Wait if currently downloading
        if self.is_downloading.get(module, False):
            logger.warning(f"⏳ Model '{module}' is currently being downloaded...")
            while self.is_downloading.get(module, False):
                await asyncio.sleep(1)

        # Try to load the model
        try:
            # First check if we need to download the model
            if not self._is_model_available(module):
                logger.warning(f"Model '{module}' not available. Attempting to download...")
                success = await self.download_model(module)
                if not success:
                    raise RuntimeError(f"Failed to download model '{module}'")

            # Load the tagger using pie_extended
            logger.info(f"Loading tagger for model '{module}'...")
            tagger = get_tagger(
                module,
                batch_size=self.batch_size,
                device=self.device,
                model_path=None  # Use default path
            )

            if not tagger:
                raise RuntimeError(f"Failed to load tagger for model '{module}'")

            # Try to import the iterator and processor
            try:
                # Dynamic import of the module-specific iterator and processor
                module_path = f"pie_extended.models.{module}.imports"
                module_imports = importlib.import_module(module_path)

                if hasattr(module_imports, 'get_iterator_and_processor'):
                    self.iterator_processors[module] = module_imports.get_iterator_and_processor
                    logger.info(f"Successfully loaded iterator and processor for '{module}'")
                else:
                    logger.warning(f"No get_iterator_and_processor found for '{module}', using default processing")
            except ImportError as e:
                logger.warning(f"Could not import iterator and processor for '{module}': {e}")
                self.iterator_processors[module] = None

            # Store the loaded tagger
            self.taggers[module] = tagger
            return tagger

        except Exception as e:
            logger.error(f"❌ Error loading model '{module}': {str(e)}")
            raise RuntimeError(f"Failed to load model '{module}': {str(e)}")

    def process_text(self, model_name: str, tagger, text: str, lower: bool = False) -> List[Dict]:
        """
        Process text with the given tagger using the appropriate iterator and processor.
        """
        if lower:
            text = text.lower()

        # Get the appropriate iterator and processor if available
        if model_name in self.iterator_processors and self.iterator_processors[model_name]:
            try:
                iterator, processor = self.iterator_processors[model_name]()
                return tagger.tag_str(text, iterator=iterator, processor=processor)
            except Exception as e:
                logger.error(f"Error using custom iterator/processor for '{model_name}': {e}")
                # Fall back to default tagging if custom iterator/processor fails
                return tagger.tag(text)
        else:
            # Use default tagging method if no custom iterator/processor is available
            return tagger.tag(text)

    async def stream_process(self, model_name: str, texts: List[str], lower: bool = False):
        """
        Process a list of texts and yield results as they become available.
        """
        tagger = await self.get_or_load_model(model_name)
        if not tagger:
            raise HTTPException(status_code=500, detail=f"Failed to load tagger for '{model_name}'")

        for text in texts:
            result = self.process_text(model_name, tagger, text, lower)
            yield f"{result}\n"


model_manager = ModelManager()
