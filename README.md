# PyHellen - Historical Languages NLP API

FastAPI-based REST API providing access to historical linguistic taggers using [Pie Extended](https://github.com/hipster-philology/nlp-pie-taggers). Supports Classical Latin, Ancient Greek, Old French, Early Modern French, Classical French, Old Dutch, and Occitan.

## Features

- **Multiple Languages**: 7 historical language models available
- **High Performance**: Concurrent batch processing, LRU caching with TTL
- **Streaming**: NDJSON and Server-Sent Events (SSE) formats
- **GPU Support**: Automatic CUDA detection and utilization
- **Production Ready**: Health checks, model preloading, comprehensive API

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

# Download a model
curl -X POST "http://localhost:8000/api/models/lasla/download"

# Preload a model into memory
curl -X POST "http://localhost:8000/api/models/lasla/load"
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
| `SSL_KEYFILE` | SSL key file path | None |
| `SSL_CERTFILE` | SSL certificate path | None |

## Architecture

```
app/
├── main.py              # FastAPI app factory
├── core/
│   ├── model_manager.py # Model lifecycle, concurrent processing
│   ├── cache.py         # LRU cache with TTL
│   ├── settings.py      # Configuration
│   └── utils.py         # GPU/CPU detection
├── routes/
│   ├── api.py           # NLP endpoints
│   └── service.py       # Health endpoints
└── schemas/
    ├── nlp.py           # NLP schemas
    └── services.py      # Service schemas
```

## License

See LICENSE file.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `pytest tests/ -v`
4. Submit a pull request
