import requests
from typing import AsyncGenerator


async def task_download(url: str, filename: str) -> AsyncGenerator[bytes, None]:
    """
        Download a file from a URL and stream it to the client in chunks.

        Args:
            url (str): The URL of the file to download.
            filename (str): The local file path where the downloaded file will be stored.

        Yields:
            bytes: Chunks of the downloaded file.

        Raises:
            requests.RequestException: If the HTTP request fails.
    """
    response = requests.get(url, stream=True)
    total = response.headers.get('content-length')

    if total is None:
        # No content length header, download whole file at once
        with open(filename, 'wb') as f:
            f.write(response.content)
        yield response.content
    else:
        total = int(total)
        downloaded = 0
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=max(int(total/1000), 1024*1024)):
                if chunk:
                    downloaded += len(chunk)
                    f.write(chunk)
                    yield chunk
                    # Optional: Print progress to console
                    done = int(50 * downloaded / total)
                    print(f'\r[{"█" * done}{"." * (50 - done)}] {downloaded}/{total} bytes', end='')


async def download_file(module: str) -> str:
    """
    Internal function to download Pie tagger models for application needs.

    Args:
        module (str): The name of the module to download files from.

    Returns:
        str: Success message or an error message.

    Raises:
        Exception: If the download process fails.
    """
    try:
        # Load the lemmatizer model
        lemmatizer = get_model(module)
        os.makedirs(os.path.join(DOWNLOAD_MODEL_PATH, module), exist_ok=True)

        if not lemmatizer.DOWNLOADS:
            raise ValueError("No files available for download")

        downloaded_files = []
        for file in lemmatizer.DOWNLOADS:
            file_path = get_path_models(module, file.name)
            async for _ in task_download(file.url, file_path):
                pass
            downloaded_files.append(file.name)

        return f"Downloaded {len(downloaded_files)} file(s): {', '.join(downloaded_files)}"

    except Exception as e:
        raise RuntimeError(f"Download failed: {str(e)}")


