# API Reference

Complete documentation of all PyHellen API endpoints.

## Base URL

```
http://localhost:8000
```

## NLP Endpoints

### List Languages

```http
GET /api/languages
```

Returns all available language models.

**Response:**
```json
{
  "languages": [
    {"code": "lasla", "name": "Classical Latin", "description": "..."},
    {"code": "grc", "name": "Ancient Greek", "description": "..."}
  ],
  "count": 7
}
```

### Tag Text (GET)

```http
GET /api/tag/{model}?text={text}&lower={boolean}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Model code (e.g., `lasla`, `grc`) |
| `text` | string | Text to process (max 10,000 chars) |
| `lower` | boolean | Lowercase text before processing (default: false) |

**Response:**
```json
{
  "result": [
    {"form": "Gallia", "lemma": "Gallia", "pos": "NOMpro", "morph": "Case=Nom|Numb=Sing"},
    {"form": "est", "lemma": "sum", "pos": "VER", "morph": "Mood=Ind|Numb=Sing|Tense=Pres"}
  ],
  "processing_time_ms": 45.23,
  "model": "lasla",
  "from_cache": false
}
```

### Tag Text (POST)

```http
POST /api/tag/{model}
Content-Type: application/json

{
  "text": "Gallia est omnis divisa",
  "lower": false
}
```

Same response as GET version.

### Batch Processing

```http
POST /api/batch/{model}?concurrent={boolean}
Content-Type: application/json

{
  "texts": ["Text 1", "Text 2", "Text 3"],
  "lower": false
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `concurrent` | boolean | Use concurrent processing (default: true) |

**Response:**
```json
{
  "results": [[...], [...], [...]],
  "total_texts": 3,
  "processing_time_ms": 123.45,
  "model": "lasla",
  "cache_hits": 1
}
```

### Streaming

```http
POST /api/stream/{model}?format={format}
Content-Type: application/json

{
  "texts": ["Text 1", "Text 2"],
  "lower": false
}
```

| Format | Content-Type | Description |
|--------|--------------|-------------|
| `ndjson` | `application/x-ndjson` | Newline Delimited JSON (default) |
| `sse` | `text/event-stream` | Server-Sent Events |
| `plain` | `text/plain` | Plain text |

**NDJSON Response:**
```json
{"index": 0, "text": "Text 1", "result": [...], "processing_time_ms": 42.1, "from_cache": false}
{"index": 1, "text": "Text 2", "result": [...], "processing_time_ms": 38.5, "from_cache": false}
```

**SSE Response:**
```
event: result
data: {"index": 0, "text": "Text 1", "result": [...], "progress": "1/2"}

event: result
data: {"index": 1, "text": "Text 2", "result": [...], "progress": "2/2"}

event: complete
data: {"total": 2, "processing_time_ms": 80.6}
```

## Model Management

### List Models

```http
GET /api/models
```

**Response:**
```json
{
  "models": {
    "lasla": {"status": "loaded", "device": "cpu"},
    "grc": {"status": "not loaded", "device": null}
  }
}
```

### Get Model Info

```http
GET /api/models/{model}
```

**Response:**
```json
{
  "name": "lasla",
  "status": "loaded",
  "device": "cpu",
  "batch_size": 256,
  "files": [...],
  "total_size_mb": 150.5,
  "has_custom_processor": false
}
```

### Load Model

```http
POST /api/models/{model}/load
```

Preloads a model into memory.

**Response:**
```json
{
  "status": "loaded",
  "model": "lasla",
  "load_time_ms": 2500.0,
  "device": "cpu"
}
```

### Unload Model

```http
POST /api/models/{model}/unload
```

Unloads a model from memory to free resources.

## Cache Management

### Get Cache Stats

```http
GET /api/cache/stats
```

**Response:**
```json
{
  "size": 150,
  "max_size": 1000,
  "hits": 500,
  "misses": 200,
  "hit_rate": 0.71
}
```

### Clear Cache

```http
POST /api/cache/clear
```

### Cleanup Expired

```http
POST /api/cache/cleanup
```

## Metrics

### Get Metrics

```http
GET /api/metrics
```

Requires `admin` scope if authentication is enabled.

## Service Endpoints

### Health Check

```http
GET /service/health
```

**Response:**
```json
{
  "status": "healthy"
}
```

### API Status

```http
GET /service/api/status
```

**Response:**
```json
{
  "status": "running",
  "version": "0.0.1",
  "device": "cpu",
  "cuda_available": false,
  "models_loaded": ["lasla"],
  "uptime_seconds": 3600
}
```

## Admin Endpoints

See [Authentication](authentication.md) for details on admin endpoints.

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error message here"
}
```

| Status Code | Description |
|-------------|-------------|
| 400 | Bad request (invalid input) |
| 401 | Unauthorized (missing/invalid token) |
| 403 | Forbidden (insufficient permissions) |
| 404 | Model or resource not found |
| 503 | Model could not be loaded |
| 500 | Internal server error |
