from typing import Dict, Optional, List, Union, Callable, Tuple, AsyncIterator, Any
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import concurrent.futures
import functools
import importlib
import json
import os
import time

import httpx
from pie_extended.cli.utils import get_model, get_tagger
from fastapi import HTTPException

from app.core.utils import get_path_models, get_device, get_n_workers
from app.core.logger import logger
from app.core.settings import settings
from app.schemas.nlp import PieLanguage, ModelStatusSchema


@dataclass
class ModelMetrics:
    """
    Metrics for tracking a single model's performance and usage.

    Attributes:
        load_count: Number of times the model has been loaded
        load_time_total_ms: Total time spent loading the model
        last_loaded_at: Timestamp of the last load
        process_count: Number of texts processed
        process_time_total_ms: Total processing time
        last_used_at: Timestamp of the last usage
        download_count: Number of download attempts
        download_time_total_ms: Total download time
        download_size_bytes: Total bytes downloaded
        error_count: Number of errors encountered
    """
    load_count: int = 0
    load_time_total_ms: float = 0.0
    last_loaded_at: Optional[datetime] = None
    process_count: int = 0
    process_time_total_ms: float = 0.0
    last_used_at: Optional[datetime] = None
    download_count: int = 0
    download_time_total_ms: float = 0.0
    download_size_bytes: int = 0
    error_count: int = 0

    @property
    def avg_load_time_ms(self) -> float:
        """Average model load time in milliseconds."""
        return self.load_time_total_ms / self.load_count if self.load_count > 0 else 0.0

    @property
    def avg_process_time_ms(self) -> float:
        """Average text processing time in milliseconds."""
        return self.process_time_total_ms / self.process_count if self.process_count > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a dictionary for JSON serialization."""
        return {
            "load_count": self.load_count,
            "avg_load_time_ms": round(self.avg_load_time_ms, 2),
            "last_loaded_at": self.last_loaded_at.isoformat() if self.last_loaded_at else None,
            "process_count": self.process_count,
            "avg_process_time_ms": round(self.avg_process_time_ms, 2),
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "download_count": self.download_count,
            "download_size_mb": round(self.download_size_bytes / (1024 * 1024), 2),
            "error_count": self.error_count
        }


@dataclass
class GlobalMetrics:
    """
    Global metrics for the ModelManager instance.

    Tracks overall application performance including uptime,
    total requests, errors, and per-model metrics.
    """
    started_at: datetime = field(default_factory=datetime.now)
    total_requests: int = 0
    total_errors: int = 0
    models: Dict[str, ModelMetrics] = field(default_factory=dict)

    def get_model_metrics(self, model_name: str) -> ModelMetrics:
        """Get or create metrics for a specific model."""
        if model_name not in self.models:
            self.models[model_name] = ModelMetrics()
        return self.models[model_name]

    def to_dict(self) -> Dict[str, Any]:
        """Convert all metrics to a dictionary for JSON serialization."""
        uptime = (datetime.now() - self.started_at).total_seconds()
        return {
            "started_at": self.started_at.isoformat(),
            "uptime_seconds": round(uptime, 2),
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "requests_per_minute": round(self.total_requests / (uptime / 60), 2) if uptime > 0 else 0,
            "models": {name: m.to_dict() for name, m in self.models.items()}
        }


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
        self._processing_semaphore = asyncio.Semaphore(settings.max_concurrent_processing)
        self._http_client: Optional[httpx.AsyncClient] = None
        self._metrics = GlobalMetrics() if settings.enable_metrics else None
        self._shutdown_event = asyncio.Event()
        logger.info(f"ModelManager initialized with device: {self.device}")

    @property
    def device(self) -> str:
        """Get device disponibility"""
        return get_device()

    @property
    def batch_size(self) -> int:
        """Batch size for model processing."""
        return getattr(self, "_batch_size", settings.batch_size)

    @batch_size.setter
    def batch_size(self, value: int):
        """Set batch size with minimum value validation."""
        self._batch_size = max(1, value)

    @property
    def metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get current metrics as a dictionary.

        Returns:
            Dictionary containing all metrics, or None if metrics are disabled.
        """
        if self._metrics is None:
            return None
        return self._metrics.to_dict()

    async def _get_http_client(self) -> httpx.AsyncClient:
        """
        Get or create the async HTTP client for downloads.

        Returns:
            An active httpx.AsyncClient instance.
        """
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(settings.download_timeout_seconds),
                follow_redirects=True
            )
        return self._http_client

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

    async def _download_file_async(
        self,
        url: str,
        filename: str,
        model_name: str
    ) -> int:
        """
        Asynchronously download a file with retry logic and cleanup on failure.

        Args:
            url: URL to download from
            filename: Local path to save the file
            model_name: Name of the model (for metrics tracking)

        Returns:
            Number of bytes downloaded

        Raises:
            RuntimeError: If download fails after all retry attempts
            asyncio.CancelledError: If shutdown is requested during download
        """
        client = await self._get_http_client()
        total_bytes = 0
        last_error = None

        for attempt in range(settings.download_max_retries):
            try:
                # Clean up partial file from previous attempt
                if os.path.exists(filename) and attempt > 0:
                    os.remove(filename)

                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    total = int(response.headers.get('content-length', 0))

                    if total == 0:
                        raise RuntimeError(f"Failed to retrieve content length for {url}")

                    logger.info(f"Downloading {os.path.basename(filename)} ({total / (1024 * 1024):.2f} MB)...")

                    with open(filename, 'wb') as f:
                        downloaded = 0
                        last_progress = 0
                        async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                            if self._shutdown_event.is_set():
                                raise asyncio.CancelledError("Shutdown requested")
                            f.write(chunk)
                            downloaded += len(chunk)
                            progress = int((downloaded / total) * 100)
                            # Log progress every 20%
                            if progress >= last_progress + 20:
                                logger.info(f"Progress: {progress}%")
                                last_progress = progress

                    total_bytes = downloaded
                    logger.info(f"✅ {os.path.basename(filename)} downloaded successfully.")
                    return total_bytes

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(f"⚠️ Timeout downloading {url} (attempt {attempt + 1}/{settings.download_max_retries})")
            except httpx.HTTPStatusError as e:
                last_error = e
                logger.warning(f"⚠️ HTTP error {e.response.status_code} for {url} (attempt {attempt + 1}/{settings.download_max_retries})")
            except asyncio.CancelledError:
                # Clean up partial file on cancellation
                if os.path.exists(filename):
                    os.remove(filename)
                raise
            except Exception as e:
                last_error = e
                logger.warning(f"⚠️ Error downloading {url}: {e} (attempt {attempt + 1}/{settings.download_max_retries})")

            # Exponential backoff before retry
            if attempt < settings.download_max_retries - 1:
                await asyncio.sleep(2 ** attempt)

        # Clean up partial file after all retries failed
        if os.path.exists(filename):
            os.remove(filename)

        if self._metrics:
            self._metrics.get_model_metrics(model_name).error_count += 1

        raise RuntimeError(f"Failed to download {url} after {settings.download_max_retries} attempts: {last_error}")

    async def download_model(self, module: str) -> bool:
        """
        Download model files for the specified module.

        Uses async HTTP client with retry logic and metrics tracking.
        Ensures only one download per model at a time via locking.

        Args:
            module: Name of the model module to download

        Returns:
            True if download succeeded, False otherwise
        """
        if self.download_locks.get(module) is None:
            self.download_locks[module] = asyncio.Lock()

        async with self.download_locks[module]:
            if self.is_downloading.get(module, False):
                logger.warning(f"⚠️ Download for '{module}' already in progress...")
                return False

            start_time = time.time()

            try:
                self.is_downloading[module] = True
                module_info = get_model(module)
                if module_info is None:
                    raise ValueError(f"Module '{module}' not found")

                if not hasattr(module_info, 'DOWNLOADS') or not module_info.DOWNLOADS:
                    raise ValueError(f"No files available for download in module '{module}'")

                os.makedirs(get_path_models(module, ""), exist_ok=True)

                total_files = len(module_info.DOWNLOADS)
                logger.info(f"Starting download of {total_files} files for module '{module}'...")

                total_bytes = 0
                downloaded_files = []
                for file in module_info.DOWNLOADS:
                    file_path = get_path_models(module, file.name)
                    logger.info(f"→ Downloading: {file.name}")

                    bytes_downloaded = await self._download_file_async(file.url, file_path, module)
                    total_bytes += bytes_downloaded
                    downloaded_files.append(file.name)

                download_time = (time.time() - start_time) * 1000

                # Update metrics
                if self._metrics:
                    metrics = self._metrics.get_model_metrics(module)
                    metrics.download_count += 1
                    metrics.download_time_total_ms += download_time
                    metrics.download_size_bytes += total_bytes

                logger.info(f"✅ Download completed: {', '.join(downloaded_files)} ({download_time:.0f}ms)")
                return True

            except Exception as e:
                logger.error(f"❌ Error downloading module '{module}': {e}")
                if self._metrics:
                    self._metrics.get_model_metrics(module).error_count += 1
                    self._metrics.total_errors += 1
                return False

            finally:
                self.is_downloading[module] = False

    async def get_or_load_model(self, module: str) -> object:
        """
        Get a loaded tagger or load it if not already loaded.
        If not available locally, download it first.

        Args:
            module: Name of the model module to load

        Returns:
            The loaded tagger instance

        Raises:
            RuntimeError: If model is not available or fails to load
        """
        # Check if already loaded
        if module in self.taggers and self.taggers[module]:
            # Update last used timestamp
            if self._metrics:
                self._metrics.get_model_metrics(module).last_used_at = datetime.now()
            return self.taggers[module]

        # Wait if currently downloading
        if self.is_downloading.get(module, False):
            logger.warning(f"⏳ Model '{module}' is currently being downloaded...")
            while self.is_downloading.get(module, False):
                await asyncio.sleep(1)

        start_time = time.time()

        # Try to load the model
        try:
            # Get module metadata
            module_info = self._is_model_available(module)

            if module_info is None:
                logger.error(f"❌ Model '{module}' not available!")
                raise RuntimeError(f"Model '{module}' not available!")

            model_path = get_path_models(module, "")

            # Check if model needs to be downloaded
            if not self._check_model_files_exist(module, model_path):
                logger.warning(f"⚠️ Model files for '{module}' not found at {model_path}. Downloading...")
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

            # Update metrics
            load_time = (time.time() - start_time) * 1000
            if self._metrics:
                metrics = self._metrics.get_model_metrics(module)
                metrics.load_count += 1
                metrics.load_time_total_ms += load_time
                metrics.last_loaded_at = datetime.now()
                metrics.last_used_at = datetime.now()

            logger.info(f"✅ Model '{module}' loaded in {load_time:.0f}ms")
            return tagger

        except Exception as e:
            logger.error(f"❌ Error loading model '{module}': {str(e)}")
            if self._metrics:
                self._metrics.get_model_metrics(module).error_count += 1
                self._metrics.total_errors += 1
            raise RuntimeError(f"Failed to load model '{module}': {str(e)}")

    def process_text(self, model_name: str, tagger, text: str, lower: bool = False) -> List[Dict]:
        """
        Process text with the given tagger using the appropriate iterator and processor.

        Args:
            model_name: Name of the model being used
            tagger: The tagger instance
            text: Text to process
            lower: Whether to lowercase the text before processing

        Returns:
            List of dictionaries containing tagged tokens
        """
        start_time = time.time()

        if lower:
            text = text.lower()

        try:
            # Get the appropriate iterator and processor if available
            if model_name in self.iterator_processors and self.iterator_processors[model_name]:
                try:
                    iterator, processor = self.iterator_processors[model_name]()
                    result = tagger.tag_str(text, iterator=iterator, processor=processor)
                except Exception as e:
                    logger.error(f"❌ Error using custom iterator/processor for '{model_name}': {e}")
                    # Fall back to default tagging if custom iterator/processor fails
                    result = tagger.tag(text)
            else:
                # Use default tagging method if no custom iterator/processor is available
                result = tagger.tag(text)

            # Update metrics
            if self._metrics:
                process_time = (time.time() - start_time) * 1000
                metrics = self._metrics.get_model_metrics(model_name)
                metrics.process_count += 1
                metrics.process_time_total_ms += process_time
                metrics.last_used_at = datetime.now()
                self._metrics.total_requests += 1

            return result

        except Exception as e:
            if self._metrics:
                self._metrics.get_model_metrics(model_name).error_count += 1
                self._metrics.total_errors += 1
            raise

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

            # Add metrics if available
            if self._metrics and model_name in self._metrics.models:
                info["metrics"] = self._metrics.models[model_name].to_dict()

            return info

        except Exception as e:
            logger.error(f"Error getting model info for '{model_name}': {e}")
            return None

    async def unload_model(self, model_name: str) -> bool:
        """
        Unload a model from memory to free resources.

        Args:
            model_name: Name of the model to unload

        Returns:
            True if model was unloaded, False if model was not loaded
        """
        if model_name not in self.taggers:
            logger.warning(f"Model '{model_name}' is not loaded")
            return False

        try:
            del self.taggers[model_name]
            if model_name in self.iterator_processors:
                del self.iterator_processors[model_name]
            logger.info(f"✅ Model '{model_name}' unloaded from memory")
            return True
        except Exception as e:
            logger.error(f"❌ Error unloading model '{model_name}': {e}")
            return False

    async def shutdown(self):
        """
        Gracefully shutdown the ModelManager.

        Closes HTTP client, shuts down thread pool executor,
        clears all loaded models, and logs final metrics.
        """
        logger.info("🛑 Shutting down ModelManager...")
        self._shutdown_event.set()

        # Close HTTP client
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            logger.info("HTTP client closed")

        # Shutdown executor (wait for pending tasks)
        self._executor.shutdown(wait=True, cancel_futures=False)
        logger.info("Thread pool executor shut down")

        # Clear models
        self.taggers.clear()
        self.iterator_processors.clear()

        # Log final metrics
        if self._metrics:
            logger.info(f"Final metrics: {self._metrics.total_requests} requests, {self._metrics.total_errors} errors")

        logger.info("✅ ModelManager shutdown complete")


model_manager = ModelManager()
