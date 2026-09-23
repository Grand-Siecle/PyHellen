"""NLP API routes with optional authentication and secure error handling."""

from typing import List, Dict, Optional, Any
import asyncio
import time
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.core.model_manager import model_manager
from app.core.cache import cache
from app.core.logger import logger
from app.core.security import require_auth, require_admin, Token, TokenScope
from app.core.security.auth import require_scope
from app.core.security.middleware import validate_model_name, get_allowed_models
from app.core.database import ModelRepository

router = APIRouter()


# ===================
# Request/Response Models
# ===================


class TextInput(BaseModel):
    """Input model for single text tagging."""

    text: str = Field(..., min_length=1, max_length=100000, description="Text to process")
    lower: bool = Field(False, description="Lowercase the text before processing")

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v


class BatchTextInput(BaseModel):
    """Input model for batch text tagging."""

    texts: List[str] = Field(..., min_length=1, max_length=100, description="List of texts to process")
    lower: bool = Field(False, description="Lowercase all texts before processing")

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("Texts list cannot be empty")
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


# ===================
# Helper Functions
# ===================


def _validate_model(model: str) -> str:
    """Validate model name against allowed models."""
    allowed = get_allowed_models()
    return validate_model_name(model, allowed)


async def _tag_with_cache(model: str, text: str, lower: bool) -> "TagResponse":
    """Serve a tagging request from the cache, or tag it off the event loop and cache the result."""
    start_time = time.time()

    async def compute():
        tagger = await model_manager.get_or_load_model(model)
        if not tagger:
            raise HTTPException(status_code=503, detail=f"Model '{model}' could not be loaded")
        return await model_manager.process_text_async(model, tagger, text, lower)

    result, from_cache = await cache.get_or_compute(model, text, lower, compute)
    processing_time = (time.time() - start_time) * 1000
    return TagResponse(result=result, processing_time_ms=round(processing_time, 2), model=model, from_cache=from_cache)


def _handle_processing_error(model: str, error: Exception) -> None:
    """Log error and raise appropriate HTTP exception based on error type."""
    error_msg = str(error)
    logger.error(f"Error processing with model '{model}': {type(error).__name__}: {error_msg}")

    # Model not available or not found
    if "not available" in error_msg.lower() or "not found" in error_msg.lower():
        raise HTTPException(status_code=404, detail=f"Model '{model}' is not available")

    # Model failed to load (download failed, file missing, etc.)
    if "failed to load" in error_msg.lower() or "failed to download" in error_msg.lower():
        raise HTTPException(status_code=503, detail=f"Model '{model}' could not be loaded. Please try again later.")

    # Invalid input rejected during processing. RuntimeError is deliberately excluded: torch/CUDA
    # failures (NotImplementedError, OutOfMemoryError...) subclass it and are server-side faults.
    if isinstance(error, ValueError):
        raise HTTPException(status_code=400, detail=f"Processing failed: {error_msg}")

    # Unexpected server errors
    raise HTTPException(status_code=500, detail="An unexpected error occurred while processing the request")


# ===================
# Public Endpoints (no auth or read scope)
# ===================


@router.get("/languages")
async def get_languages():
    """
    Return a list of supported language schemas.

    **Returns**:
    - **200**: A JSON object with a list of supported languages from the database.
    """
    model_repo = ModelRepository()
    models = model_repo.get_all(include_inactive=False)

    return {
        "languages": [{"code": m.code, "name": m.name, "description": m.description} for m in models],
        "count": len(models),
    }


@router.get("/tag/{model}")
async def tag_text_get(
    model: str,
    text: str = Query(..., min_length=1, max_length=10000, description="Text to process"),
    lower: bool = Query(False, description="Lowercase text before processing"),
    _: Optional[Token] = Depends(require_auth),
):
    """
    Tag text using GET request (for simple queries).

    **Parameters**:
    - **model**: The name of the model to use (e.g., "lasla", "grc")
    - **text**: Query parameter with the text to analyze
    - **lower**: Optional query parameter to lowercase text

    **Requires**: `read` scope if authentication is enabled.
    """
    # Validate model name
    model = _validate_model(model)

    try:
        return await _tag_with_cache(model, text, lower)
    except HTTPException:
        raise
    except Exception as e:
        _handle_processing_error(model, e)


@router.post("/tag/{model}", response_model=TagResponse)
async def tag_text(model: str, input_data: TextInput, _: Optional[Token] = Depends(require_auth)):
    """
    Tag text using the specified model.

    This endpoint processes a single text input and returns the analysis results.
    If the model is not loaded, it will be downloaded automatically.

    **Requires**: `read` scope if authentication is enabled.
    """
    # Validate model name
    model = _validate_model(model)

    try:
        return await _tag_with_cache(model, input_data.text, input_data.lower)
    except HTTPException:
        raise
    except Exception as e:
        _handle_processing_error(model, e)


@router.post("/batch/{model}", response_model=BatchResponse)
async def batch_process(
    model: str,
    batch_data: BatchTextInput,
    concurrent: bool = Query(False, description="Use concurrent processing (experimental)"),
    _: Optional[Token] = Depends(require_auth),
):
    """
    Process multiple texts using the specified model.

    This endpoint handles batch processing of multiple text inputs.
    Uses concurrent processing by default for better performance.

    **Requires**: `read` scope if authentication is enabled.
    """
    # Validate model name
    model = _validate_model(model)
    start_time = time.time()
    texts, lower = batch_data.texts, batch_data.lower

    try:
        results = await cache.get_many(model, texts, lower)
        cache_hits = sum(result is not None for result in results)

        # Tag each distinct uncached text once
        uncached: Dict[str, List[int]] = {}
        for idx, result in enumerate(results):
            if result is None:
                uncached.setdefault(texts[idx], []).append(idx)

        if uncached:
            tagger = await model_manager.get_or_load_model(model)
            if not tagger:
                raise HTTPException(status_code=503, detail=f"Model '{model}' could not be loaded")

            if concurrent:
                tagged = await asyncio.gather(
                    *(model_manager.process_text_async(model, tagger, text, lower) for text in uncached)
                )
            else:
                tagged = [await model_manager.process_text_async(model, tagger, text, lower) for text in uncached]

            for text, result in zip(uncached, tagged):
                for idx in uncached[text]:
                    results[idx] = result
            await cache.set_many(model, list(zip(uncached, tagged)), lower)

        processing_time = (time.time() - start_time) * 1000

        return BatchResponse(
            results=results,
            total_texts=len(batch_data.texts),
            processing_time_ms=round(processing_time, 2),
            model=model,
            cache_hits=cache_hits,
        )

    except HTTPException:
        raise
    except Exception as e:
        _handle_processing_error(model, e)


@router.post("/stream/{model}")
async def stream_process(
    model: str,
    batch_data: BatchTextInput,
    format: str = Query("ndjson", description="Output format: 'ndjson' (default), 'sse', or 'plain'"),
    _: Optional[Token] = Depends(require_auth),
):
    """
    Stream process multiple texts using the specified model.

    Supports multiple output formats:
    - **ndjson**: Newline Delimited JSON (default)
    - **sse**: Server-Sent Events for real-time web applications
    - **plain**: Simple text stream

    **Requires**: `read` scope if authentication is enabled.
    """
    # Validate model name
    model = _validate_model(model)

    # Validate format
    if format not in ("ndjson", "sse", "plain"):
        raise HTTPException(status_code=400, detail="Invalid format. Use 'ndjson', 'sse', or 'plain'")

    try:
        if format == "sse":

            async def sse_generator():
                async for result in model_manager.stream_process_sse(model, batch_data.texts, batch_data.lower):
                    yield result

            return StreamingResponse(
                sse_generator(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        elif format == "ndjson":

            async def ndjson_generator():
                async for result in model_manager.stream_process_ndjson(model, batch_data.texts, batch_data.lower):
                    yield result

            return StreamingResponse(
                ndjson_generator(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache"}
            )
        else:  # plain format (legacy)

            async def stream_generator():
                async for result in model_manager.stream_process(model, batch_data.texts, batch_data.lower):
                    yield result

            return StreamingResponse(stream_generator(), media_type="text/plain", headers={"Cache-Control": "no-cache"})

    except Exception as e:
        _handle_processing_error(model, e)


# ===================
# Model Management Endpoints
# ===================


@router.get("/models")
async def list_models(_: Optional[Token] = Depends(require_auth)):
    """
    Get the status of all available models.

    **Requires**: `read` scope if authentication is enabled.
    """
    models_status = model_manager.get_all_models_status()
    return {"models": models_status}


@router.get("/models/{model}")
async def get_model_info(model: str, _: Optional[Token] = Depends(require_auth)):
    """
    Get detailed information about a specific model.

    **Requires**: `read` scope if authentication is enabled.
    """
    model = _validate_model(model)
    info = model_manager.get_model_info(model)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found")
    return info


@router.post("/models/{model}/load")
async def preload_model(model: str, _: Optional[Token] = Depends(require_scope(TokenScope.WRITE))):
    """
    Preload a model into memory.

    This endpoint loads the model immediately (downloading if needed).

    **Requires**: `write` scope if authentication is enabled.
    """
    model = _validate_model(model)

    try:
        start_time = time.time()
        tagger = await model_manager.get_or_load_model(model)
        if not tagger:
            raise HTTPException(status_code=503, detail=f"Model '{model}' could not be loaded")

        load_time = (time.time() - start_time) * 1000
        return {"status": "loaded", "model": model, "load_time_ms": round(load_time, 2), "device": model_manager.device}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error preloading model '{model}': {e}")
        raise HTTPException(status_code=500, detail="Failed to load model")


@router.post("/models/{model}/unload")
async def unload_model(model: str, _: Optional[Token] = Depends(require_scope(TokenScope.WRITE))):
    """
    Unload a model from memory to free resources.

    **Requires**: `write` scope if authentication is enabled.
    """
    model = _validate_model(model)

    success = await model_manager.unload_model(model)
    if not success:
        raise HTTPException(status_code=404, detail=f"Model '{model}' is not loaded")
    return {"status": "unloaded", "model": model, "message": f"Model '{model}' has been unloaded from memory"}


# ===================
# Cache Management (requires write scope)
# ===================


@router.get("/cache/stats")
async def get_cache_stats(_: Optional[Token] = Depends(require_auth)):
    """
    Get cache statistics.

    **Requires**: `read` scope if authentication is enabled.
    """
    return await cache.get_stats()


@router.post("/cache/clear")
async def clear_cache(_: Optional[Token] = Depends(require_scope(TokenScope.WRITE))):
    """
    Clear all cached results.

    **Requires**: `write` scope if authentication is enabled.
    """
    count = await cache.clear()
    logger.info(f"Cache cleared: {count} entries removed")
    return {"cleared_entries": count, "message": "Cache cleared successfully"}


@router.post("/cache/cleanup")
async def cleanup_cache(_: Optional[Token] = Depends(require_scope(TokenScope.WRITE))):
    """
    Remove expired entries from cache.

    **Requires**: `write` scope if authentication is enabled.
    """
    count = await cache.cleanup_expired()
    return {"removed_entries": count, "message": "Expired entries cleaned up"}


# ===================
# Metrics (requires admin scope)
# ===================


@router.get("/metrics")
async def get_metrics(_: Optional[Token] = Depends(require_admin)):
    """
    Get performance metrics for the model manager.

    Returns metrics including uptime, request counts, and per-model statistics.

    **Requires**: `admin` scope if authentication is enabled.
    """
    metrics = model_manager.metrics
    if metrics is None:
        return {"message": "Metrics collection is disabled"}
    return metrics
