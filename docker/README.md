## Docker Support

PyHellen includes full Docker support for both CPU and GPU environments. All Docker-related files are organized in the `docker/` directory.

### Quick Start

The easiest way to run PyHellen with Docker:

```bash
# Make the run script executable
chmod +x docker/scripts/run.sh

# Run it
./docker/scripts/run.sh
```

The script will automatically detect if your system has GPU capabilities and select the appropriate container.

### Docker Directory Structure

- `docker/` - All Docker-related files
  - `Dockerfile` - For CPU environments
  - `Dockerfile.gpu` - For GPU environments with NVIDIA CUDA support
  - `docker-compose.yml` - Configuration for both CPU and GPU services
  - `scripts/` - Helper scripts
  - `.dockerignore` - Files excluded from Docker build

For complete Docker documentation, see [docker/README.md](docker/README.md).

### Running with Different Environments

```bash
# CPU mode
docker-compose -f docker/docker-compose.yml up -d pyhellen

# GPU mode (requires NVIDIA Docker)
docker-compose -f docker/docker-compose.yml up -d pyhellen-gpu
```

### GPU Requirements

To use GPU acceleration:

1. You need an NVIDIA GPU with CUDA support
2. You need to install the NVIDIA Container Toolkit (nvidia-docker)
   - Installation guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html