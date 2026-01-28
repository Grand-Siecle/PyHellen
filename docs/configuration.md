# Configuration

All configuration is done via environment variables. Copy `edit_dot_env` to `.env` and customize.

```bash
cp edit_dot_env .env
```

## Environment Variables

### Model Storage

| Variable | Description | Default |
|----------|-------------|---------|
| `DOWNLOAD_MODEL_PATH` | Path where NLP models are stored | `~/.local/share/pyhellen` |
| `PRELOAD_MODELS` | Models to preload at startup (comma-separated) | `""` |

**Example:**
```bash
DOWNLOAD_MODEL_PATH="/data/models"
PRELOAD_MODELS="lasla,grc"
```

### Security

| Variable | Description | Default |
|----------|-------------|---------|
| `AUTH_ENABLED` | Enable token-based authentication | `false` |
| `SECRET_KEY` | Secret key for token hashing (required if auth enabled) | `""` |
| `TOKEN_DB_PATH` | Path to SQLite token database | `tokens.db` |
| `AUTO_CREATE_ADMIN_TOKEN` | Auto-create admin token on first run | `true` |

**Generate a secret key:**
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### CORS

| Variable | Description | Default |
|----------|-------------|---------|
| `CORS_ORIGINS` | Allowed origins (comma-separated, `*` for all) | `*` |
| `CORS_ALLOW_CREDENTIALS` | Allow credentials in CORS | `false` |

**Production example:**
```bash
CORS_ORIGINS="https://myapp.com,https://api.myapp.com"
CORS_ALLOW_CREDENTIALS=true
```

**Note:** `CORS_ALLOW_CREDENTIALS=true` cannot be used with `CORS_ORIGINS="*"`

### Rate Limiting

| Variable | Description | Default |
|----------|-------------|---------|
| `RATE_LIMIT_ENABLED` | Enable rate limiting | `false` |
| `RATE_LIMIT_REQUESTS` | Max requests per time window | `100` |
| `RATE_LIMIT_WINDOW_SECONDS` | Rate limit window in seconds | `60` |

### Processing

| Variable | Description | Default |
|----------|-------------|---------|
| `MAX_CONCURRENT_PROCESSING` | Max concurrent text processing tasks | `10` |
| `BATCH_SIZE` | Batch size for model processing | `256` |
| `DOWNLOAD_TIMEOUT_SECONDS` | Model download timeout | `300` |
| `DOWNLOAD_MAX_RETRIES` | Download retry attempts | `3` |

### Metrics & Logging

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_METRICS` | Enable metrics collection | `true` |
| `LOG_LEVEL` | Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) | `INFO` |
| `LOG_FORMAT` | Log format (`json`, `text`, `auto`) | `auto` |
| `LOG_FILE` | Optional log file path | `""` |

**Log formats:**
- `json` - Structured JSON logging (recommended for production)
- `text` - Human-readable format (development)
- `auto` - Detects environment (Docker → json, else text)

### SSL/TLS

| Variable | Description | Default |
|----------|-------------|---------|
| `SSL_KEYFILE` | Path to SSL private key | `""` |
| `SSL_CERTFILE` | Path to SSL certificate | `""` |

**Example:**
```bash
SSL_KEYFILE="/etc/ssl/private/server.key"
SSL_CERTFILE="/etc/ssl/certs/server.crt"
```

## Example Configurations

### Development

```bash
# .env for development
AUTH_ENABLED=false
LOG_LEVEL=DEBUG
LOG_FORMAT=text
CORS_ORIGINS="*"
```

### Production

```bash
# .env for production
AUTH_ENABLED=true
SECRET_KEY="your-secure-secret-key-here"
LOG_LEVEL=INFO
LOG_FORMAT=json
CORS_ORIGINS="https://yourapp.com"
CORS_ALLOW_CREDENTIALS=true
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW_SECONDS=60
PRELOAD_MODELS="lasla,grc"
```

### Docker

When running in Docker, some variables are set automatically:

```bash
# Set by Docker
PIE_EXTENDED_DOWNLOADS=/data/models
DOCKER_CONTAINER=true
LOG_FORMAT=json
```

You can override via docker-compose environment:

```yaml
services:
  pyhellen:
    environment:
      - AUTH_ENABLED=true
      - SECRET_KEY=${SECRET_KEY}
      - PRELOAD_MODELS=lasla,grc
```

## Semantic Versioning

The `VERSION` setting follows [Semantic Versioning](https://semver.org/):

```
MAJOR.MINOR.PATCH
```

- MAJOR: Breaking API changes
- MINOR: New features (backwards compatible)
- PATCH: Bug fixes
