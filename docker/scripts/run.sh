#!/bin/bash

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Set the working directory to the docker directory
cd "$(dirname "$0")/.." || exit 1

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: docker-compose is not installed${NC}"
    echo "Please install docker-compose first:"
    echo "https://docs.docker.com/compose/install/"
    exit 1
fi

# Check for NVIDIA GPU
echo -e "${BLUE}Checking for NVIDIA GPU...${NC}"
echo "PATH is: $PATH"
if command -v nvidia-smi 2> /dev/null; then
    nvidia-smi 2> /dev/null
    if [ $? -eq 0 ]; then
        HAS_GPU=1
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
        echo -e "${GREEN}✓ NVIDIA GPU detected: $GPU_NAME${NC}"
    else
        HAS_GPU=0
        echo -e "${YELLOW}✗ NVIDIA GPU command failed${NC}"
    fi
else
    HAS_GPU=0
    echo -e "${YELLOW}✗ No NVIDIA GPU detected${NC}"
fi

# Check for nvidia-docker
if [ $HAS_GPU -eq 1 ]; then
    echo -e "${BLUE}Checking for NVIDIA Docker support...${NC}"
    if command -v nvidia-container-cli 2>/dev/null;  then
        version=$(dpkg -l | grep nvidia-container-toolkit | awk '{print $3}')
        HAS_NVIDIA_DOCKER=1
        echo -e "${GREEN}✓ NVIDIA Docker support detected${NC}"
        echo "nvidia-container-toolkit is installed. Version: $version"
    else
        HAS_NVIDIA_DOCKER=0
        echo -e "${YELLOW}✗ NVIDIA Docker support not detected${NC}"
        echo -e "${YELLOW}To use GPU acceleration, install the NVIDIA Container Toolkit:${NC}"
        echo "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    fi
else
    HAS_NVIDIA_DOCKER=0
fi

# Ask user which version to run
if [ $HAS_GPU -eq 1 ] && [ $HAS_NVIDIA_DOCKER -eq 1 ]; then
    echo ""
    echo -e "${BLUE}PyHellen can run with GPU acceleration.${NC}"
    echo "1) GPU mode (recommended)"
    echo "2) CPU mode"
    echo ""
    read -p "Which mode do you want to use? [1/2]: " mode

    if [[ "$mode" == "1" ]]; then
        CONTAINER="pyhellen-gpu"
    else
        CONTAINER="pyhellen"
    fi
else
    CONTAINER="pyhellen"
fi

# Function to handle container operation
run_container() {
    local action=$1

    echo -e "${BLUE}Running docker-compose $action $CONTAINER${NC}"

    if [ "$action" == "up" ]; then
        docker-compose -f docker-compose.yml up -d "$CONTAINER"

        # Wait for container to start
        echo -e "${BLUE}Waiting for container to start...${NC}"
        sleep 3

        # Check if container is running
        if docker-compose -f docker-compose.yml ps "$CONTAINER" | grep -q "Up"; then
            echo -e "${GREEN}✓ PyHellen is now running!${NC}"
            echo -e "${GREEN}✓ API is available at: http://localhost:8000${NC}"
            echo -e "${GREEN}✓ Swagger UI: http://localhost:8000/docs${NC}"
            echo ""
            echo -e "${BLUE}To view logs:${NC} docker-compose -f docker-compose.yml logs -f $CONTAINER"
            echo -e "${BLUE}To stop:${NC} ./scripts/run.sh stop"
        else
            echo -e "${RED}✗ Container failed to start. Checking logs:${NC}"
            docker-compose -f docker-compose.yml logs "$CONTAINER"
        fi
    else
        docker-compose -f docker-compose.yml "$action" "$CONTAINER"

        if [ "$action" == "stop" ] || [ "$action" == "down" ]; then
            echo -e "${GREEN}✓ PyHellen has been stopped${NC}"
        fi
    fi
}

# Process command line arguments
case "$1" in
    start|up)
        run_container "up"
        ;;
    stop)
        run_container "stop"
        ;;
    restart)
        run_container "restart"
        ;;
    logs)
        docker-compose -f docker-compose.yml logs -f "$CONTAINER"
        ;;
    down)
        run_container "down"
        ;;
    build)
        docker-compose -f docker-compose.yml build "$CONTAINER"
        ;;
    *)
        # Default action is to start
        run_container "up"
        ;;
esac

exit 0