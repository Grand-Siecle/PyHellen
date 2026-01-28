# Docker Deployment

PyHellen provides Docker images for both CPU and GPU environments.

## Quick Start

### Pull from Registry

```bash
# CPU version
docker pull ghcr.io/grand-siecle/pyhellen:latest

# Run
docker run -p 8000:8000 ghcr.io/grand-siecle/pyhellen:latest
```

### Using Docker Compose

```bash
# CPU version
docker-compose -f docker/docker-compose.yml up -d pyhellen

# GPU version (requires NVIDIA Container Toolkit)
docker-compose -f docker/docker-compose.yml up -d pyhellen-gpu
```

### Auto-detect Script

```bash
chmod +x docker/scripts/run.sh
./docker/scripts/run.sh
```

Automatically detects GPU availability and runs the appropriate container.

## Building Locally

### CPU Version

```bash
docker build -f docker/Dockerfile -t pyhellen:latest .
```

### GPU Version

```bash
docker build -f docker/Dockerfile.gpu -t pyhellen:gpu .
```

## Configuration

### Environment Variables

Pass environment variables via `-e` or docker-compose:

```bash
docker run -p 8000:8000 \
  -e AUTH_ENABLED=true \
  -e SECRET_KEY="your-secret-key" \
  -e PRELOAD_MODELS="lasla,grc" \
  ghcr.io/grand-siecle/pyhellen:latest
```

### Persistent Model Storage

Models are stored in `/data/models` inside the container. Mount a volume to persist:

```bash
docker run -p 8000:8000 \
  -v pyhellen_models:/data/models \
  ghcr.io/grand-siecle/pyhellen:latest
```

Or with a host directory:

```bash
docker run -p 8000:8000 \
  -v /path/on/host:/data/models \
  ghcr.io/grand-siecle/pyhellen:latest
```

### Token Database

For authentication, persist the token database:

```bash
docker run -p 8000:8000 \
  -e AUTH_ENABLED=true \
  -e SECRET_KEY="your-secret-key" \
  -e TOKEN_DB_PATH=/data/tokens.db \
  -v pyhellen_data:/data \
  ghcr.io/grand-siecle/pyhellen:latest
```

## Docker Compose Examples

### Basic CPU

```yaml
version: '3.8'

services:
  pyhellen:
    image: ghcr.io/grand-siecle/pyhellen:latest
    ports:
      - "8000:8000"
    volumes:
      - model_data:/data/models
    restart: unless-stopped

volumes:
  model_data:
```

### Production with Auth

```yaml
version: '3.8'

services:
  pyhellen:
    image: ghcr.io/grand-siecle/pyhellen:latest
    ports:
      - "8000:8000"
    volumes:
      - model_data:/data/models
      - token_data:/data/db
    environment:
      - AUTH_ENABLED=true
      - SECRET_KEY=${SECRET_KEY}
      - TOKEN_DB_PATH=/data/db/tokens.db
      - PRELOAD_MODELS=lasla,grc
      - LOG_LEVEL=INFO
      - LOG_FORMAT=json
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/service/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  model_data:
  token_data:
```

### GPU Version

```yaml
version: '3.8'

services:
  pyhellen-gpu:
    image: ghcr.io/grand-siecle/pyhellen:gpu
    ports:
      - "8000:8000"
    volumes:
      - model_data:/data/models
    environment:
      - USE_CUDA=1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

volumes:
  model_data:
```

## GPU Requirements

To use GPU acceleration:

1. **NVIDIA GPU** with CUDA support
2. **NVIDIA Driver** installed on host
3. **NVIDIA Container Toolkit** (nvidia-docker)

### Install NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Verify GPU Access

```bash
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
```

## Health Checks

The container includes a health check:

```bash
curl http://localhost:8000/service/health
```

Docker will mark the container as unhealthy if this fails.

## Logs

### View Logs

```bash
docker logs pyhellen-api
docker logs -f pyhellen-api  # Follow
```

### Log Format

By default, Docker containers use JSON logging:

```json
{"timestamp": "2024-01-15T10:00:00", "level": "INFO", "message": "..."}
```

## Resource Limits

Recommended limits for production:

| Resource | CPU | GPU |
|----------|-----|-----|
| CPU | 2 cores | 4 cores |
| Memory | 4 GB | 8 GB |
| Disk (models) | 10 GB | 10 GB |

## Networking

### Behind Reverse Proxy (nginx)

```nginx
upstream pyhellen {
    server localhost:8000;
}

server {
    listen 443 ssl;
    server_name api.example.com;

    location / {
        proxy_pass http://pyhellen;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Docker Network

```bash
docker network create pyhellen-network
docker run --network pyhellen-network ...
```

## Troubleshooting

### Container Won't Start

Check logs:
```bash
docker logs pyhellen-api
```

### Out of Memory

Increase memory limit or reduce batch size:
```bash
-e BATCH_SIZE=128
```

### Model Download Fails

Check network and increase timeout:
```bash
-e DOWNLOAD_TIMEOUT_SECONDS=600
-e DOWNLOAD_MAX_RETRIES=5
```

### GPU Not Detected

Verify NVIDIA Container Toolkit:
```bash
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
```
