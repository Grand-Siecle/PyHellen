import os
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, Depends, HTTPException
from pie_extended.cli.utils import get_model

from app.schemas.nlp import SupportedLanguages, PieLanguage
from app.core.model_manager import task_download
from app.core.utils import get_path_models
from app.constants import DOWNLOAD_MODEL_PATH

router = APIRouter()

@router.get("/languages", response_model=SupportedLanguages)
async def get_languages():
    """
    Return a list of supported language schemas.
    """
    languages = [lang.value for lang in PieLanguage]
    return {"languages": languages}

@router.get("/download/{module}/", response_class=StreamingResponse)
async def download_model(module: str):
    """
    Download a Pie taggers model for future availability. Check list for available models

    This endpoint downloads the first available file in the specified module's `DOWNLOADS` list
    and streams it to the client in chunks.

    **Parameters**:
    - **module** (str): The name of the Pie tagger module to download files from.

    **Returns**:
    - **200**: The streamed file as an HTTP response with the `Content-Disposition` header
      set for download.
    - **404**: If no files are available for download in the module.
    - **500**: If the download process fails due to a server error.

    **Example**:
    ```bash
    curl -O http://127.0.0.1:8000/download/mymodel/
    ```

    **Notes**:
    - The file will be saved locally to `{DOWNLOAD_MODEL_PATH}/{module}/{file.name}`.
    - The client receives the file as a binary stream.
    """
    try:
        lemmatizer = get_model(module)
        os.makedirs(os.path.join(DOWNLOAD_MODEL_PATH, module), exist_ok=True)

        if not lemmatizer.DOWNLOADS:
            raise HTTPException(status_code=404, detail="No files available for download")

        file_count = len(lemmatizer.DOWNLOADS)
        print(f"Starting download of {file_count} files for module '{module}'...")

        # Handle multiple files in sequence
        for index, file in enumerate(lemmatizer.DOWNLOADS):
            file_path = get_path_models(module, file.name)
            print(f"Downloading file {index + 1}/{file_count}: {file.name}...")

            # Stream one file at a time
            yield f"Downloading {file.name}...\n".encode()

            async for chunk in task_download(file.url, file_path):
                yield chunk

            yield f"\n- {file.name} downloaded\n".encode()

        yield "\n✅ Download completed successfully!\n".encode()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))