import os
import asyncio
import requests
from pie_extended.cli.utils import get_model

from app.core.utils import get_path_models
from app.core.logger import logger


async def task_download(url: str, filename: str):
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


async def download_model(module: str, state):
    """
    Download model files for the specified module using `task_download`.
    Uses app state for lock management.
    """
    download_locks = state.download_locks
    is_downloading = state.is_downloading

    if download_locks.get(module) is None:
        download_locks[module] = asyncio.Lock()

    async with download_locks[module]:
        if is_downloading.get(module, False):
            logger.warning(f"⚠️ Download for '{module}' already in progress...")
            return

        try:
            is_downloading[module] = True
            lemmatizer = get_model(module)
            if lemmatizer is None:
                raise ValueError(f"Module '{module}' not found")

            if not lemmatizer.DOWNLOADS:
                raise ValueError(f"No files available for download in module '{module}'")

            os.makedirs(get_path_models(module, ""), exist_ok=True)

            total_files = len(lemmatizer.DOWNLOADS)
            logger.info(f"Starting download of {total_files} files for module '{module}'...")

            downloaded_files = []
            for file in lemmatizer.DOWNLOADS:
                file_path = get_path_models(module, file.name)
                logger.info(f"→ Downloading: {file.name}")

                await task_download(file.url, file_path)
                downloaded_files.append(file.name)

            logger.success(f"✅ Download completed: {', '.join(downloaded_files)}")

        except Exception as e:
            logger.error(f"❌ Error downloading module '{module}': {e}")

        finally:
            is_downloading[module] = False


async def get_or_load_model(module: str, state):
    """
    Check if the model is loaded; if not, download it.
    If a download is already in progress, wait for completion.
    """
    try:
        lemmatizer = get_model(module)
        if lemmatizer:
            return lemmatizer

        if state.is_downloading.get(module, False):
            logger.warn(f"⏳ Model '{module}' is currently being downloaded...")
            while state.is_downloading.get(module, False):
                await asyncio.sleep(1)

        logger.warn(f"Model '{module}' not loaded. Attempting to download...")

        await download_model(module, state)
        return get_model(module)

    except Exception as e:
        logger.error(f"❌ Error loading model '{module}': {str(e)}")
        raise RuntimeError(f"Failed to load model '{module}': {str(e)}")
