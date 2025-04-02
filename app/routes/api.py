from typing import List, Dict, Optional
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.schemas.nlp import SupportedLanguages, PieLanguage
from app.core.model_manager import model_manager
from app.core.logger import logger

router = APIRouter()


class TextInput(BaseModel):
    text: str
    lower: bool = False


class BatchTextInput(BaseModel):
    texts: List[str]
    lower: bool = False


@router.get("/languages", response_model=SupportedLanguages)
async def get_languages():
    """
    Return a list of supported language schemas.

    **Returns**:
    - **200**: A JSON object with a list of supported language codes.
    - **500**: If retrieving the language list fails due to a server error.

    **Example**:
    ```bash
    curl -X GET http://{address}/api/languages
    ```
    """
    languages = list(PieLanguage)
    return {"languages": languages}


@router.post("/tag/{model}")
async def tag_text(model: str, input_data: TextInput):
    """
    Tag text using the specified model.

    This endpoint processes a single text input and returns the analysis results.
    If the model is not loaded, it will be downloaded automatically.

    **Parameters**:
    - **model**: The name of the model to use (e.g., "lasla", "grc")
    - **input_data**: JSON object containing the text to analyze and processing options

    **Returns**:
    - **200**: A JSON object with the tagging results
    - **500**: If model loading or processing fails

    **Example**:
    ```bash
    curl -X POST http://{address}/api/tag/lasla -H "Content-Type: application/json" -d '{"text": "Lorem ipsum", "lower": true}'
    ```
    """
    try:
        tagger = await model_manager.get_or_load_model(model)
        if not tagger:
            raise HTTPException(status_code=500, detail=f"Failed to load tagger for model '{model}'")

        result = model_manager.process_text(model, tagger, input_data.text, input_data.lower)
        return {"result": result}

    except Exception as e:
        logger.error(f"Error processing text with model '{model}': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch/{model}")
async def batch_process(model: str, batch_data: BatchTextInput):
    """
    Process multiple texts using the specified model.

    This endpoint handles batch processing of multiple text inputs and returns all results together.

    **Parameters**:
    - **model**: The name of the model to use
    - **batch_data**: JSON object containing a list of texts and processing options

    **Returns**:
    - **200**: A JSON object with results for all texts
    - **500**: If model loading or processing fails

    **Example**:
    ```bash
    curl -X POST http://{address}/api/batch/lasla -H "Content-Type: application/json" -d '{"texts": ["Text 1", "Text 2"], "lower": false}'
    ```
    """
    try:
        tagger = await model_manager.get_or_load_model(model)
        if not tagger:
            raise HTTPException(status_code=500, detail=f"Failed to load tagger for model '{model}'")

        results = []
        for text in batch_data.texts:
            result = model_manager.process_text(model, tagger, text, batch_data.lower)
            results.append(result)

        return {"results": results}

    except Exception as e:
        logger.error(f"Error processing batch with model '{model}': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream/{model}")
async def stream_process(model: str, batch_data: BatchTextInput):
    """
    Stream process multiple texts using the specified model.

    This endpoint processes texts sequentially and streams the results back as they become available.
    The response is a text stream with each result on a new line.

    **Parameters**:
    - **model**: The name of the model to use
    - **batch_data**: JSON object containing a list of texts and processing options

    **Returns**:
    - **200**: A text stream with processing results
    - **500**: If model loading or processing fails

    **Example**:
    ```bash
    curl -X POST http://{address}/api/stream/lasla -H "Content-Type: application/json" -d '{"texts": ["Text 1", "Text 2"], "lower": false}'
    ```
    """
    try:
        async def stream_generator():
            async for result in model_manager.stream_process(model, batch_data.texts, batch_data.lower):
                yield result

        return StreamingResponse(
            stream_generator(),
            media_type="text/plain",
            headers={
                'Access-Control-Allow-Origin': "*",
                'Cache-Control': 'no-cache'
            }
        )

    except Exception as e:
        logger.error(f"Error streaming process with model '{model}': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_models():
    """
    Get the status of all available models.

    This endpoint provides information about all supported language models,
    including whether they are loaded, downloading, or not available.

    **Returns**:
    - **200**: A JSON object with model status information

    **Example**:
    ```bash
    curl -X GET http://{address}/api/models
    ```
    """
    models_status = model_manager.get_all_models_status()
    return {"models": models_status}


@router.post("/models/{model}/download")
async def download_model(model: str, background_tasks: BackgroundTasks):
    """
    Trigger download of a model.

    This endpoint initiates the download of a specified model in the background.

    **Parameters**:
    - **model**: The name of the model to download

    **Returns**:
    - **202**: Accepted, with information about the download process
    - **400**: If the model is already being downloaded
    - **404**: If the model is not found

    **Example**:
    ```bash
    curl -X POST http://{address}/api/models/lasla/download
    ```
    """
    if model_manager.is_downloading.get(model, False):
        return {
            "status": "already_downloading",
            "message": f"Model '{model}' is already being downloaded"
        }

    if not model_manager._is_model_available(model):
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found")

    background_tasks.add_task(model_manager.download_model, model)

    return {
        "status": "downloading",
        "message": f"Download of model '{model}' started in the background"
    }