# PyHellen - Historical Languages NLP API

[![CI/CD](https://github.com/Grand-Siecle/PyHellen/actions/workflows/ci.yml/badge.svg)](https://github.com/Grand-Siecle/PyHellen/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Grand-Siecle/PyHellen/branch/main/graph/badge.svg)](https://codecov.io/gh/Grand-Siecle/PyHellen)
[![Python 3.8-3.10](https://img.shields.io/badge/python-3.8--3.10-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg?logo=docker&logoColor=white)](https://ghcr.io/grand-siecle/pyhellen)

FastAPI-based REST API providing access to historical linguistic taggers using [Pie Extended](https://github.com/hipster-philology/nlp-pie-taggers). Supports Classical Latin, Ancient Greek, Old French, Early Modern French, Classical French, Old Dutch, and Occitan.

## Features

- **Multiple Languages**: 7 built-in historical language models (can be activated/deactivated)
- **High Performance**: Concurrent batch processing, LRU caching with TTL
- **Streaming**: NDJSON and Server-Sent Events (SSE) formats
- **GPU Support**: Automatic CUDA detection and utilization
- **Production Ready**: Health checks, model preloading, comprehensive API
- **Token Authentication**: Optional token-based security with scopes (read, write, admin)
- **SQLite Database**: Persistent storage for models, tokens, cache, audit logs, and metrics
- **Admin API**: Model activation/deactivation and token management
- **Request Logging**: Track all API requests with detailed metrics
- **Audit Trail**: Complete audit logging for security-sensitive operations

## Supported Languages

| Code | Language |
|------|----------|
| `lasla` | Classical Latin |
| `grc` | Ancient Greek |
| `fro` | Old French |
| `freem` | Early Modern French |
| `fr` | Classical French |
| `dum` | Old Dutch |
| `occ_cont` | Occitan Contemporain |

## Installation

```bash
# Clone the repository
git clone https://github.com/Grand-Siecle/PyHellen.git
cd PyHellen

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp edit_dot_env .env
# Edit .env to set DOWNLOAD_MODEL_PATH="/path/to/store/models"
```

**Requirements**: Python 3.8-3.10

## Quick Start

```bash
# Start the server
python -m app.main

# Server runs on http://localhost:8000
# API docs: http://localhost:8000/docs
```

## API Endpoints

### Text Processing

#### Single Text (GET)
```bash
curl "http://localhost:8000/api/tag/lasla?text=Gallia%20est%20omnis%20divisa"
```

#### Single Text (POST)
```bash
curl -X POST "http://localhost:8000/api/tag/lasla" \
  -H "Content-Type: application/json" \
  -d '{"text": "Gallia est omnis divisa in partes tres", "lower": false}'
```

#### Batch Processing (Concurrent)
```bash
curl -X POST "http://localhost:8000/api/batch/lasla?concurrent=true" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Lorem ipsum dolor sit amet",
      "Consectetur adipiscing elit",
      "Sed do eiusmod tempor"
    ],
    "lower": false
  }'
```

#### Streaming (NDJSON)
```bash
curl -X POST "http://localhost:8000/api/stream/lasla?format=ndjson" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Text 1", "Text 2", "Text 3"], "lower": true}'
```

#### Streaming (Server-Sent Events)
```bash
curl -X POST "http://localhost:8000/api/stream/lasla?format=sse" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Text 1", "Text 2"], "lower": false}'
```

### Model Management

```bash
# List all models and their status
curl "http://localhost:8000/api/models"

# Get detailed model info
curl "http://localhost:8000/api/models/lasla"

# Preload a model into memory
curl -X POST "http://localhost:8000/api/models/lasla/load"

# Unload a model from memory
curl -X POST "http://localhost:8000/api/models/lasla/unload"
```

### Cache Management

```bash
# Get cache statistics
curl "http://localhost:8000/api/cache/stats"

# Clear all cached results
curl -X POST "http://localhost:8000/api/cache/clear"

# Remove expired entries
curl -X POST "http://localhost:8000/api/cache/cleanup"
```

### Service Health

```bash
# Health check
curl "http://localhost:8000/service/health"

# Detailed status (GPU/CPU info)
curl "http://localhost:8000/service/api/status"
```

### Admin API (requires admin token)

```bash
# Authentication status
curl -H "Authorization: Bearer YOUR_TOKEN" "http://localhost:8000/admin/auth/status"

# Token management
curl -H "Authorization: Bearer YOUR_TOKEN" "http://localhost:8000/admin/tokens"
curl -X POST -H "Authorization: Bearer YOUR_TOKEN" -H "Content-Type: application/json" \
  -d '{"name": "my-app", "scopes": ["read"]}' \
  "http://localhost:8000/admin/tokens"

# Model management (activate/deactivate only)
curl -H "Authorization: Bearer YOUR_TOKEN" "http://localhost:8000/admin/models"
curl -X POST -H "Authorization: Bearer YOUR_TOKEN" "http://localhost:8000/admin/models/lasla/deactivate"
curl -X POST -H "Authorization: Bearer YOUR_TOKEN" "http://localhost:8000/admin/models/lasla/activate"

# Audit logs
curl -H "Authorization: Bearer YOUR_TOKEN" "http://localhost:8000/admin/audit"

# Request logs and statistics
curl -H "Authorization: Bearer YOUR_TOKEN" "http://localhost:8000/admin/requests/stats"

# Metrics
curl -H "Authorization: Bearer YOUR_TOKEN" "http://localhost:8000/admin/metrics/persistent"
```

## Response Examples

### Tag Response
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

### Batch Response
```json
{
  "results": [[...], [...], [...]],
  "total_texts": 3,
  "processing_time_ms": 123.45,
  "model": "lasla",
  "cache_hits": 1
}
```

### NDJSON Stream (one JSON per line)
```json
{"index": 0, "text": "Lorem ipsum...", "result": [...], "processing_time_ms": 42.1, "from_cache": false}
{"index": 1, "text": "Consectetur...", "result": [...], "processing_time_ms": 38.5, "from_cache": false}
```

## JavaScript Integration

### NDJSON Streaming
```javascript
async function processTexts(texts) {
  const response = await fetch('http://localhost:8000/api/stream/lasla?format=ndjson', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ texts, lower: false })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const lines = decoder.decode(value).split('\n').filter(l => l);
    for (const line of lines) {
      const result = JSON.parse(line);
      console.log(`Processed ${result.index}: ${result.processing_time_ms}ms`);
    }
  }
}
```

### Server-Sent Events
```javascript
async function processWithSSE(texts) {
  const response = await fetch('http://localhost:8000/api/stream/lasla?format=sse', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ texts, lower: false })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split('\n\n');
    buffer = events.pop();

    for (const event of events) {
      const [eventLine, dataLine] = event.split('\n');
      const eventType = eventLine.replace('event: ', '');
      const data = JSON.parse(dataLine.replace('data: ', ''));

      if (eventType === 'result') {
        console.log(`Progress: ${data.progress}`);
      }
    }
  }
}
```

## Authentication

Authentication is **disabled by default**. To enable token-based authentication:

1. Generate a secret key:
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

2. Configure `.env`:
   ```bash
   AUTH_ENABLED=true
   SECRET_KEY="your-generated-secret-key"
   ```

3. On first run with `AUTO_CREATE_ADMIN_TOKEN=true`, an admin token is created and logged.

### Token Scopes

| Scope | Permissions |
|-------|-------------|
| `read` | Use tagging endpoints (GET/POST /api/tag, /api/batch, /api/stream) |
| `write` | Modify cache, load/unload models |
| `admin` | Full access including token management and model activation |

### Using Tokens

```bash
# Via Authorization header (Bearer)
curl -H "Authorization: Bearer YOUR_TOKEN" "http://localhost:8000/api/tag/lasla?text=..."

# Via X-API-Key header
curl -H "X-API-Key: YOUR_TOKEN" "http://localhost:8000/api/tag/lasla?text=..."
```

## Testing

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

## Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `DOWNLOAD_MODEL_PATH` | Path to store model files | `~/.local/share/pyhellen` |
| `PRELOAD_MODELS` | Models to preload at startup (comma-separated) | `""` |
| **Security** | | |
| `AUTH_ENABLED` | Enable token-based authentication | `false` |
| `SECRET_KEY` | Secret key for token hashing (required if auth enabled) | `""` |
| `TOKEN_DB_PATH` | Path to SQLite token database | `tokens.db` |
| `AUTO_CREATE_ADMIN_TOKEN` | Auto-create admin token on first run | `true` |
| **CORS** | | |
| `CORS_ORIGINS` | Allowed origins (comma-separated, `*` for all) | `*` |
| `CORS_ALLOW_CREDENTIALS` | Allow credentials in CORS | `false` |
| **Rate Limiting** | | |
| `RATE_LIMIT_ENABLED` | Enable rate limiting | `false` |
| `RATE_LIMIT_REQUESTS` | Max requests per window | `100` |
| `RATE_LIMIT_WINDOW_SECONDS` | Rate limit window | `60` |
| **Processing** | | |
| `MAX_CONCURRENT_PROCESSING` | Max concurrent processing tasks | `10` |
| `BATCH_SIZE` | Batch size for model processing | `256` |
| `DOWNLOAD_TIMEOUT_SECONDS` | Model download timeout | `300` |
| `DOWNLOAD_MAX_RETRIES` | Download retry attempts | `3` |
| **Metrics & Logging** | | |
| `ENABLE_METRICS` | Enable metrics collection | `true` |
| `LOG_LEVEL` | Log level (DEBUG, INFO, WARNING, ERROR) | `INFO` |
| `LOG_FORMAT` | Log format (json, text, auto) | `auto` |
| **SSL** | | |
| `SSL_KEYFILE` | SSL key file path | `""` |
| `SSL_CERTFILE` | SSL certificate path | `""` |

## Architecture

```
app/
├── main.py                  # FastAPI app factory, lifespan manager
├── constants.py             # Logger and path constants
├── core/
│   ├── model_manager.py     # Model lifecycle, concurrent processing
│   ├── cache.py             # LRU cache with TTL
│   ├── settings.py          # Pydantic Settings configuration
│   ├── environment.py       # PIE_EXTENDED_DOWNLOADS env setup
│   ├── logger.py            # Logging configuration
│   ├── utils.py             # GPU/CPU detection
│   ├── database/            # SQLite database layer
│   │   ├── engine.py        # Database engine and session
│   │   ├── models.py        # SQLModel ORM models
│   │   └── repositories/    # Data access layer
│   │       ├── model_repo.py
│   │       ├── token_repo.py
│   │       ├── cache_repo.py
│   │       ├── audit_repo.py
│   │       ├── request_log_repo.py
│   │       └── metrics_repo.py
│   ├── security/            # Authentication system
│   │   ├── auth.py          # AuthManager, dependencies
│   │   ├── database.py      # Token database
│   │   ├── models.py        # Token schemas
│   │   └── middleware.py    # Security headers, validation
│   └── middleware/
│       └── request_logger.py # Request logging middleware
├── routes/
│   ├── api.py               # NLP endpoints (/api/*)
│   ├── admin.py             # Admin endpoints (/admin/*)
│   └── service.py           # Health endpoints (/service/*)
└── schemas/
    ├── nlp.py               # NLP schemas
    ├── models.py            # Model management schemas
    └── services.py          # Service schemas

tests/
├── conftest.py              # Pytest fixtures
├── test_api.py              # API endpoint tests
├── test_admin.py            # Admin endpoint tests
├── test_security.py         # Security tests
├── test_database.py         # Database tests
├── test_cache.py            # Cache tests
├── test_settings.py         # Settings tests
├── test_schemas.py          # Schema tests
├── test_token_repo.py       # Token repository tests
└── test_model_manager.py    # ModelManager tests
```

## License

See LICENSE file.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `pytest tests/ -v`
4. Submit a pull request
