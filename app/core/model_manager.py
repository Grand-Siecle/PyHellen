from typing import Dict, Optional, List, Union, Callable, Tuple, AsyncIterator, Any
import asyncio
import concurrent.futures
import functools
import importlib
import json
import os
import time
import requests
from pie_extended.cli.utils import get_model, get_tagger
from fastapi import HTTPException

from app.core.utils import get_path_models, get_device, get_n_workers
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
        self.models: Dict[str, Any] = {}  # For backwards compatibility
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=get_n_workers())
        self._processing_semaphore = asyncio.Semaphore(10)  # Limit concurrent processing
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
            return get_model(model_name)
        except ImportError:
            return False

    def _check_model_files_exist(self, module: str, model_path: str) -> bool:
        """
        Check if all required model files exist at the specified path.
        """
        try:
            module_info = get_model(module)
            if module_info and hasattr(module_info, 'DOWNLOADS'):
                for file in module_info.DOWNLOADS:
                    file_path = os.path.join(model_path, file.name)
                    if not os.path.exists(file_path):
                        logger.warning(f"Model file {file_path} not found")
                        return False
                return True
            return False
        except Exception as e:
            logger.error(f"Error checking model files for '{module}': {str(e)}")
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
            # Get module metadata
            module_info = self._is_model_available(module)

            if module_info is None:
                logger.error(f"❌ Model '{module}' not available!")
                raise RuntimeError(f"Model '{module}' not available!")

            model_path = get_path_models(module, "")

            # Check if model neet to be downloading
            if not self._check_model_files_exist(module, model_path):
                logger.warning(f"⚠️ Model files for '{module}' not found at {model_path}. Attempting to download...")
                success = await self.download_model(module)
                if not success:
                    raise RuntimeError(f"Failed to download model '{module}'")

            # Load the tagger using pie_extended
            logger.info(f"☕ Loading tagger for model '{module}'...")
            tagger = get_tagger(
                module,
                batch_size=self.batch_size,
                device=self.device,
                model_path=None
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
                    logger.info(f"✅ Successfully loaded iterator and processor for '{module}'")
                else:
                    logger.warning(f"⚠️ No get_iterator_and_processor found for '{module}', using default processing")
            except ImportError as e:
                logger.warning(f"⚠️ Could not import iterator and processor for '{module}': {e}")
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
                logger.error(f"❌ Error using custom iterator/processor for '{model_name}': {e}")
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

    async def process_text_async(
        self,
        model_name: str,
        tagger,
        text: str,
        lower: bool = False
    ) -> Dict[str, Any]:
        """
        Process text asynchronously using thread pool to avoid blocking.
        """
        loop = asyncio.get_event_loop()
        async with self._processing_semaphore:
            result = await loop.run_in_executor(
                self._executor,
                functools.partial(self.process_text, model_name, tagger, text, lower)
            )
            return result

    async def batch_process_concurrent(
        self,
        model_name: str,
        texts: List[str],
        lower: bool = False,
        max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Process multiple texts concurrently for better performance.

        Args:
            model_name: Name of the model to use
            texts: List of texts to process
            lower: Whether to lowercase the text
            max_concurrent: Maximum number of concurrent processing tasks

        Returns:
            List of results in the same order as input texts
        """
        from app.core.cache import cache

        tagger = await self.get_or_load_model(model_name)
        if not tagger:
            raise HTTPException(status_code=500, detail=f"Failed to load tagger for '{model_name}'")

        results = [None] * len(texts)
        tasks_to_process = []

        # Check cache first
        for idx, text in enumerate(texts):
            cached = await cache.get(model_name, text, lower)
            if cached is not None:
                results[idx] = cached
            else:
                tasks_to_process.append((idx, text))

        # Process uncached texts concurrently
        if tasks_to_process:
            semaphore = asyncio.Semaphore(max_concurrent)

            async def process_with_semaphore(idx: int, text: str):
                async with semaphore:
                    result = await self.process_text_async(model_name, tagger, text, lower)
                    await cache.set(model_name, text, lower, result)
                    return idx, result

            tasks = [
                process_with_semaphore(idx, text)
                for idx, text in tasks_to_process
            ]

            completed = await asyncio.gather(*tasks, return_exceptions=True)

            for item in completed:
                if isinstance(item, Exception):
                    logger.error(f"Error in batch processing: {item}")
                    raise item
                idx, result = item
                results[idx] = result

        return results

    async def stream_process_ndjson(
        self,
        model_name: str,
        texts: List[str],
        lower: bool = False
    ) -> AsyncIterator[str]:
        """
        Stream process texts and yield results as NDJSON (Newline Delimited JSON).
        Each line is a valid JSON object.
        """
        from app.core.cache import cache

        tagger = await self.get_or_load_model(model_name)
        if not tagger:
            raise HTTPException(status_code=500, detail=f"Failed to load tagger for '{model_name}'")

        for idx, text in enumerate(texts):
            start_time = time.time()

            # Check cache
            cached = await cache.get(model_name, text, lower)
            if cached is not None:
                result = cached
                from_cache = True
            else:
                result = await self.process_text_async(model_name, tagger, text, lower)
                await cache.set(model_name, text, lower, result)
                from_cache = False

            processing_time = time.time() - start_time

            output = {
                "index": idx,
                "text": text[:100] + "..." if len(text) > 100 else text,
                "result": result,
                "processing_time_ms": round(processing_time * 1000, 2),
                "from_cache": from_cache
            }
            yield json.dumps(output) + "\n"

    async def stream_process_sse(
        self,
        model_name: str,
        texts: List[str],
        lower: bool = False
    ) -> AsyncIterator[str]:
        """
        Stream process texts using Server-Sent Events (SSE) format.
        """
        from app.core.cache import cache

        tagger = await self.get_or_load_model(model_name)
        if not tagger:
            raise HTTPException(status_code=500, detail=f"Failed to load tagger for '{model_name}'")

        total = len(texts)

        # Send initial event
        yield f"event: start\ndata: {json.dumps({'total': total, 'model': model_name})}\n\n"

        for idx, text in enumerate(texts):
            start_time = time.time()

            # Check cache
            cached = await cache.get(model_name, text, lower)
            if cached is not None:
                result = cached
                from_cache = True
            else:
                result = await self.process_text_async(model_name, tagger, text, lower)
                await cache.set(model_name, text, lower, result)
                from_cache = False

            processing_time = time.time() - start_time

            data = {
                "index": idx,
                "progress": f"{idx + 1}/{total}",
                "result": result,
                "processing_time_ms": round(processing_time * 1000, 2),
                "from_cache": from_cache
            }
            yield f"event: result\ndata: {json.dumps(data)}\n\n"

        # Send completion event
        yield f"event: complete\ndata: {json.dumps({'total_processed': total})}\n\n"

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a model.
        """
        try:
            module_info = get_model(model_name)
            if not module_info:
                return None

            info = {
                "name": model_name,
                "status": self.get_model_status(model_name),
                "device": self.device,
                "batch_size": self.batch_size,
                "files": [],
                "total_size_mb": 0
            }

            if hasattr(module_info, 'DOWNLOADS'):
                for file in module_info.DOWNLOADS:
                    file_info = {"name": file.name, "url": file.url}
                    file_path = get_path_models(model_name, file.name)
                    if os.path.exists(file_path):
                        size_mb = os.path.getsize(file_path) / (1024 * 1024)
                        file_info["size_mb"] = round(size_mb, 2)
                        file_info["downloaded"] = True
                        info["total_size_mb"] += size_mb
                    else:
                        file_info["downloaded"] = False
                    info["files"].append(file_info)

                info["total_size_mb"] = round(info["total_size_mb"], 2)

            # Check if iterator/processor is available
            info["has_custom_processor"] = model_name in self.iterator_processors

            return info

        except Exception as e:
            logger.error(f"Error getting model info for '{model_name}': {e}")
            return None


model_manager = ModelManager()
