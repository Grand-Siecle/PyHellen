# PyHellen - Historical Languages NLP API

[![CI/CD](https://github.com/Grand-Siecle/PyHellen/actions/workflows/ci.yml/badge.svg)](https://github.com/Grand-Siecle/PyHellen/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Grand-Siecle/PyHellen/branch/main/graph/badge.svg)](https://codecov.io/gh/Grand-Siecle/PyHellen)
[![Python 3.8-3.10](https://img.shields.io/badge/python-3.8--3.10-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg?logo=docker&logoColor=white)](https://ghcr.io/grand-siecle/pyhellen)

REST API for historical linguistic tagging using [Pie Extended](https://github.com/hipster-philology/nlp-pie-taggers).

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

| Code | Language | Code | Language |
|------|----------|------|----------|
| `lasla` | Classical Latin | `freem` | Early Modern French |
| `grc` | Ancient Greek | `fr` | Classical French |
| `fro` | Old French | `dum` | Old Dutch |
| | | `occ_cont` | Occitan |

## Quick Start

### With Docker (recommended)

```bash
# CPU
docker run -p 8000:8000 ghcr.io/grand-siecle/pyhellen:latest

# Or with docker-compose
docker-compose -f docker/docker-compose.yml up -d pyhellen
```

### Manual Installation

```bash
pip install -r requirements.txt
cp edit_dot_env .env
python -m app.main
```

API available at `http://localhost:8000` | Docs at `/docs`

## Usage

```bash
# Tag text
curl "http://localhost:8000/api/tag/lasla?text=Gallia%20est%20omnis%20divisa"

# Batch processing
curl -X POST "http://localhost:8000/api/batch/lasla" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Text 1", "Text 2"], "lower": false}'

# Streaming (NDJSON)
curl -X POST "http://localhost:8000/api/stream/lasla?format=ndjson" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Text 1", "Text 2"]}'
```

## Documentation

| Document | Description |
|----------|-------------|
| [API Reference](docs/api.md) | Complete API endpoints documentation |
| [Authentication](docs/authentication.md) | Token-based security setup |
| [Configuration](docs/configuration.md) | Environment variables reference |
| [Docker](docs/docker.md) | Docker deployment guide |

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

## License

GPL-3.0 - See [LICENSE](LICENSE)
