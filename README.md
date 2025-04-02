# PyHellen - NLP API with Pie models

This document describes the improved model management and streaming API for the PyHellen NLP service, which provides access to various historical linguistic taggers for multiple languages using Pie Extended.

## Key Improvements

1. **Enhanced Model Management**
   - Centralized model manager singleton for better state management
   - Proper integration with Pie Extended's `get_tagger` and iterator/processor pattern
   - Improved download handling with proper locking mechanisms
   - Status tracking for models
   - Support for batch processing and streaming responses

2. **New API Endpoints**
   - `/api/tag/{model}` - Process a single text input
   - `/api/batch/{model}` - Process multiple texts in one request
   - `/api/stream/{model}` - Stream process multiple texts
   - `/api/models` - Get status of all models
   - `/api/models/{model}/download` - Explicitly trigger model download

## Configuration

The application requires Python 3.8 to 3.10 and the following environment variable:

```
DOWNLOAD_MODEL_PATH="/path/to/store/models"
```

## Usage Examples

### Single Text Processing

```bash
curl -X POST "http://localhost:8000/api/tag/lasla" \
  -H "Content-Type: application/json" \
  -d '{"text": "Lorem ipsum dolor sit amet", "lower": true}'
```

### Batch Processing

```bash
curl -X POST "http://localhost:8000/api/batch/lasla" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Lorem ipsum dolor sit amet", 
      "Consectetur adipiscing elit"
    ],
    "lower": false
  }'
```

### Streaming Processing

```bash
curl -X POST "http://localhost:8000/api/stream/lasla" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Lorem ipsum dolor sit amet", 
      "Consectetur adipiscing elit",
      "Sed do eiusmod tempor incididunt"
    ],
    "lower": true
  }'
```

### Get Model Status

```bash
curl -X GET "http://localhost:8000/api/models"
```

### Download a Model

```bash
curl -X POST "http://localhost:8000/api/models/lasla/download"
```

## Integration Example

Here's how to integrate the streaming API with JavaScript:

```javascript
async function streamProcess() {
  const response = await fetch('http://localhost:8000/api/stream/lasla', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      texts: [
        "Text 1 to analyze",
        "Text 2 to analyze",
        "Text 3 to analyze"
      ],
      lower: true
    }),
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  
  let resultArea = document.getElementById('results');
  
  while (true) {
    const { done, value } = await reader.read();
    
    if (done) {
      break;
    }
    
    // Process the stream chunks as they arrive
    const text = decoder.decode(value);
    resultArea.innerHTML += text;
  }
}
```

## API Documentation

Full API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Running the Application

```bash
python -m app.main
```

The server will start on `http://localhost:8000`.