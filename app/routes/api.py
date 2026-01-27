from typing import List, Dict, Optional, Any
import time
from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from app.schemas.nlp import SupportedLanguages, PieLanguage
from app.core.model_manager import model_manager
from app.core.cache import cache
from app.core.logger import logger

router = APIRouter()


class TextInput(BaseModel):
    """Input model for single text tagging."""
    text: str = Field(..., min_length=1, max_length=100000, description="Text to process")
    lower: bool = Field(False, description="Lowercase the text before processing")

    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return v


class BatchTextInput(BaseModel):
    """Input model for batch text tagging."""
    texts: List[str] = Field(..., min_length=1, max_length=100, description="List of texts to process")
    lower: bool = Field(False, description="Lowercase all texts before processing")

    @field_validator('texts')
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError('Texts list cannot be empty')
        return [t for t in v if t.strip()]  # Filter out empty strings


class TagResponse(BaseModel):
    """Response model for single text tagging."""
    result: List[Dict[str, Any]]
    processing_time_ms: float
    model: str
    from_cache: bool = False


class BatchResponse(BaseModel):
    """Response model for batch text tagging."""
    results: List[List[Dict[str, Any]]]
    total_texts: int
    processing_time_ms: float
    model: str
    cache_hits: int = 0


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    name: str
    status: str
    device: str
    batch_size: int
    files: List[Dict[str, Any]]
    total_size_mb: float
    has_custom_processor: bool


@router.get("/languages", response_model=SupportedLanguages)
async def get_languages():
    """
    Return a list of supported language schemas.

    **Returns**:
    - **200**: A JSON object with a list of supported language codes.

    **Example**:
    ```bash
    curl -X GET http://{address}/api/languages
    ```
    """
    languages = list(PieLanguage)
    return {"languages": languages}


@router.get("/tag/{model}")
async def tag_text_get(
    model: str,
    text: str = Query(..., min_length=1, max_length=10000, description="Text to process"),
    lower: bool = Query(False, description="Lowercase text before processing")
):
    """
    Tag text using GET request (for simple queries).

    **Parameters**:
    - **model**: The name of the model to use (e.g., "lasla", "grc")
    - **text**: Query parameter with the text to analyze
    - **lower**: Optional query parameter to lowercase text

    **Returns**:
    - **200**: A JSON object with the tagging results

    **Example**:
    ```bash
    curl "http://{address}/api/tag/lasla?text=Lorem%20ipsum&lower=true"
    ```
    """
    start_time = time.time()

    try:
        # Check cache first
        cached = await cache.get(model, text, lower)
        if cached is not None:
            processing_time = (time.time() - start_time) * 1000
            return TagResponse(
                result=cached,
                processing_time_ms=round(processing_time, 2),
                model=model,
                from_cache=True
            )

        tagger = await model_manager.get_or_load_model(model)
        if not tagger:
            raise HTTPException(status_code=500, detail=f"Failed to load tagger for model '{model}'")

        result = model_manager.process_text(model, tagger, text, lower)

        # Cache the result
        await cache.set(model, text, lower, result)

        processing_time = (time.time() - start_time) * 1000
        return TagResponse(
            result=result,
            processing_time_ms=round(processing_time, 2),
            model=model,
            from_cache=False
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing text with model '{model}': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tag/{model}", response_model=TagResponse)
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
    start_time = time.time()

    try:
        # Check cache first
        cached = await cache.get(model, input_data.text, input_data.lower)
        if cached is not None:
            processing_time = (time.time() - start_time) * 1000
            return TagResponse(
                result=cached,
                processing_time_ms=round(processing_time, 2),
                model=model,
                from_cache=True
            )

        tagger = await model_manager.get_or_load_model(model)
        if not tagger:
            raise HTTPException(status_code=500, detail=f"Failed to load tagger for model '{model}'")

        result = model_manager.process_text(model, tagger, input_data.text, input_data.lower)

        # Cache the result
        await cache.set(model, input_data.text, input_data.lower, result)

        processing_time = (time.time() - start_time) * 1000
        return TagResponse(
            result=result,
            processing_time_ms=round(processing_time, 2),
            model=model,
            from_cache=False
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing text with model '{model}': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch/{model}", response_model=BatchResponse)
async def batch_process(
    model: str,
    batch_data: BatchTextInput,
    concurrent: bool = Query(True, description="Use concurrent processing for better performance")
):
    """
    Process multiple texts using the specified model.

    This endpoint handles batch processing of multiple text inputs and returns all results together.
    Uses concurrent processing by default for better performance.

    **Parameters**:
    - **model**: The name of the model to use
    - **batch_data**: JSON object containing a list of texts and processing options
    - **concurrent**: Query param to enable/disable concurrent processing (default: true)

    **Returns**:
    - **200**: A JSON object with results for all texts
    - **500**: If model loading or processing fails

    **Example**:
    ```bash
    curl -X POST "http://{address}/api/batch/lasla?concurrent=true" -H "Content-Type: application/json" -d '{"texts": ["Text 1", "Text 2"], "lower": false}'
    ```
    """
    start_time = time.time()

    try:
        if concurrent:
            # Use optimized concurrent processing
            results = await model_manager.batch_process_concurrent(
                model,
                batch_data.texts,
                batch_data.lower
            )
        else:
            # Fall back to sequential processing
            tagger = await model_manager.get_or_load_model(model)
            if not tagger:
                raise HTTPException(status_code=500, detail=f"Failed to load tagger for model '{model}'")

            results = []
            for text in batch_data.texts:
                # Check cache
                cached = await cache.get(model, text, batch_data.lower)
                if cached is not None:
                    results.append(cached)
                else:
                    result = model_manager.process_text(model, tagger, text, batch_data.lower)
                    await cache.set(model, text, batch_data.lower, result)
                    results.append(result)

        processing_time = (time.time() - start_time) * 1000

        # Count cache hits from stats
        cache_stats = cache.stats
        cache_hits = cache_stats.get("hits", 0)

        return BatchResponse(
            results=results,
            total_texts=len(batch_data.texts),
            processing_time_ms=round(processing_time, 2),
            model=model,
            cache_hits=cache_hits
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing batch with model '{model}': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream/{model}")
async def stream_process(
    model: str,
    batch_data: BatchTextInput,
    format: str = Query("ndjson", description="Output format: 'ndjson' (default), 'sse', or 'plain'")
):
    """
    Stream process multiple texts using the specified model.

    This endpoint processes texts and streams the results back as they become available.
    Supports multiple output formats:
    - **ndjson**: Newline Delimited JSON (default) - each line is a JSON object
    - **sse**: Server-Sent Events - for real-time web applications
    - **plain**: Simple text stream (legacy format)

    **Parameters**:
    - **model**: The name of the model to use
    - **batch_data**: JSON object containing a list of texts and processing options
    - **format**: Output format query parameter

    **Returns**:
    - **200**: A stream with processing results in the specified format
    - **500**: If model loading or processing fails

    **Example (NDJSON)**:
    ```bash
    curl -X POST "http://{address}/api/stream/lasla?format=ndjson" -H "Content-Type: application/json" -d '{"texts": ["Text 1", "Text 2"]}'
    ```

    **Example (SSE)**:
    ```bash
    curl -X POST "http://{address}/api/stream/lasla?format=sse" -H "Content-Type: application/json" -d '{"texts": ["Text 1", "Text 2"]}'
    ```
    """
    try:
        if format == "sse":
            async def sse_generator():
                async for result in model_manager.stream_process_sse(model, batch_data.texts, batch_data.lower):
                    yield result

            return StreamingResponse(
                sse_generator(),
                media_type="text/event-stream",
                headers={
                    'Access-Control-Allow-Origin': "*",
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive'
                }
            )
        elif format == "ndjson":
            async def ndjson_generator():
                async for result in model_manager.stream_process_ndjson(model, batch_data.texts, batch_data.lower):
                    yield result

            return StreamingResponse(
                ndjson_generator(),
                media_type="application/x-ndjson",
                headers={
                    'Access-Control-Allow-Origin': "*",
                    'Cache-Control': 'no-cache'
                }
            )
        else:  # plain format (legacy)
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


@router.get("/models/{model}")
async def get_model_info(model: str):
    """
    Get detailed information about a specific model.

    **Parameters**:
    - **model**: The name of the model

    **Returns**:
    - **200**: Detailed model information
    - **404**: If the model is not found

    **Example**:
    ```bash
    curl -X GET http://{address}/api/models/lasla
    ```
    """
    info = model_manager.get_model_info(model)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found")
    return info


@router.post("/models/{model}/load")
async def preload_model(model: str):
    """
    Preload a model into memory.

    This endpoint loads the model immediately (downloading if needed) and keeps it ready for use.

    **Parameters**:
    - **model**: The name of the model to preload

    **Returns**:
    - **200**: Model loaded successfully
    - **500**: If loading fails

    **Example**:
    ```bash
    curl -X POST http://{address}/api/models/lasla/load
    ```
    """
    try:
        start_time = time.time()
        tagger = await model_manager.get_or_load_model(model)
        if not tagger:
            raise HTTPException(status_code=500, detail=f"Failed to load model '{model}'")

        load_time = (time.time() - start_time) * 1000
        return {
            "status": "loaded",
            "model": model,
            "load_time_ms": round(load_time, 2),
            "device": model_manager.device
        }
    except Exception as e:
        logger.error(f"Error preloading model '{model}': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/stats")
async def get_cache_stats():
    """
    Get cache statistics.

    **Returns**:
    - **200**: Cache statistics including size, hit rate, etc.

    **Example**:
    ```bash
    curl -X GET http://{address}/api/cache/stats
    ```
    """
    return cache.stats


@router.post("/cache/clear")
async def clear_cache():
    """
    Clear all cached results.

    **Returns**:
    - **200**: Number of cleared entries

    **Example**:
    ```bash
    curl -X POST http://{address}/api/cache/clear
    ```
    """
    count = await cache.clear()
    return {"cleared_entries": count, "message": "Cache cleared successfully"}


@router.post("/cache/cleanup")
async def cleanup_cache():
    """
    Remove expired entries from cache.

    **Returns**:
    - **200**: Number of removed entries

    **Example**:
    ```bash
    curl -X POST http://{address}/api/cache/cleanup
    ```
    """
    count = await cache.cleanup_expired()
    return {"removed_entries": count, "message": "Expired entries cleaned up"}


@router.get("/metrics")
async def get_metrics():
    """
    Get performance metrics for the model manager.

    Returns metrics including:
    - Uptime and total requests
    - Per-model statistics (load times, process times, error counts)
    - Request rates

    **Returns**:
    - **200**: Metrics data or message if metrics are disabled

    **Example**:
    ```bash
    curl -X GET http://{address}/api/metrics
    ```
    """
    metrics = model_manager.metrics
    if metrics is None:
        return {"message": "Metrics collection is disabled"}
    return metrics


@router.post("/models/{model}/unload")
async def unload_model(model: str):
    """
    Unload a model from memory to free resources.

    **Parameters**:
    - **model**: The name of the model to unload

    **Returns**:
    - **200**: Model unloaded successfully
    - **404**: If the model is not loaded

    **Example**:
    ```bash
    curl -X POST http://{address}/api/models/lasla/unload
    ```
    """
    success = await model_manager.unload_model(model)
    if not success:
        raise HTTPException(status_code=404, detail=f"Model '{model}' is not loaded")
    return {
        "status": "unloaded",
        "model": model,
        "message": f"Model '{model}' has been unloaded from memory"
    }