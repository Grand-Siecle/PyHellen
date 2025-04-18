#!/bin/bash

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Set the working directory to the docker directory
cd "$(dirname "$0")/.." || exit 1

# Check if we're in sudo mode
SUDO_PREFIX=""
if [ "$EUID" -ne 0 ]; then
    # Check if docker commands need sudo
    if ! docker info > /dev/null 2>&1; then
        if sudo docker info > /dev/null 2>&1; then
            echo -e "${YELLOW}Docker requires sudo privileges. Using sudo for docker commands.${NC}"
            SUDO_PREFIX="sudo"
        else
            echo -e "${RED}Error: Cannot access Docker. Please ensure Docker is installed and you have proper permissions.${NC}"
            exit 1
        fi
    fi
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null && ! $SUDO_PREFIX command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: docker-compose is not installed${NC}"
    echo "Please install docker-compose first:"
    echo "https://docs.docker.com/compose/install/"
    exit 1
fi

# Check for NVIDIA GPU
echo -e "${BLUE}Checking for NVIDIA GPU...${NC}"
HAS_GPU=0

# Try with and without sudo
for CMD_PREFIX in "" "sudo"; do
    if $CMD_PREFIX command -v nvidia-smi &> /dev/null; then
        GPU_OUTPUT=$($CMD_PREFIX nvidia-smi 2>&1)
        if [ $? -eq 0 ]; then
            HAS_GPU=1
            GPU_NAME=$($CMD_PREFIX nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
            echo -e "${GREEN}✓ NVIDIA GPU detected: $GPU_NAME${NC}"
            if [ -n "$CMD_PREFIX" ]; then
                echo -e "${YELLOW}Note: nvidia-smi requires sudo on this system${NC}"
            fi
            break
        fi
    fi
done

if [ $HAS_GPU -eq 0 ]; then
    echo -e "${YELLOW}✗ No NVIDIA GPU detected${NC}"
fi

# Check for nvidia-docker
HAS_NVIDIA_DOCKER=0
if [ $HAS_GPU -eq 1 ]; then
    echo -e "${BLUE}Checking for NVIDIA Docker support...${NC}"

    # Try with direct docker command
    if $SUDO_PREFIX docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
        HAS_NVIDIA_DOCKER=1
        echo -e "${GREEN}✓ NVIDIA Docker runtime detected${NC}"
    # Try checking for nvidia-container-cli
    elif $SUDO_PREFIX command -v nvidia-container-cli &>/dev/null; then
        HAS_NVIDIA_DOCKER=1
        echo -e "${GREEN}✓ NVIDIA Container CLI detected${NC}"
    # Look for the package (might need sudo)
    elif $SUDO_PREFIX dpkg -l | grep -q nvidia-container-toolkit; then
        version=$($SUDO_PREFIX dpkg -l | grep nvidia-container-toolkit | awk '{print $3}')
        HAS_NVIDIA_DOCKER=1
        echo -e "${GREEN}✓ NVIDIA Docker support detected${NC}"
        echo "nvidia-container-toolkit is installed. Version: $version"
    else
        echo -e "${YELLOW}✗ NVIDIA Docker support not detected${NC}"
        echo -e "${YELLOW}To use GPU acceleration, install the NVIDIA Container Toolkit:${NC}"
        echo "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    fi
else
    HAS_NVIDIA_DOCKER=0
fi

# WSL2 specific checks
if grep -q Microsoft /proc/version || grep -q microsoft /proc/version; then
    echo -e "${BLUE}WSL2 environment detected.${NC}"
    # In WSL2, if GPU is present in Windows, we might still use it even if nvidia-docker isn't fully set up
    if [ $HAS_GPU -eq 1 ] && [ $HAS_NVIDIA_DOCKER -eq 0 ]; then
        echo -e "${YELLOW}Note: In WSL2, you may need additional configuration to use GPUs.${NC}"
        echo "See: https://docs.nvidia.com/cuda/wsl-user-guide/index.html"
    fi
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
        $SUDO_PREFIX docker-compose -f docker-compose.yml up -d "$CONTAINER"

        # Wait for container to start
        echo -e "${BLUE}Waiting for container to start...${NC}"
        sleep 3

        # Check if container is running
        if $SUDO_PREFIX docker-compose -f docker-compose.yml ps "$CONTAINER" | grep -q "Up"; then
            echo -e "${GREEN}✓ PyHellen is now running!${NC}"
            echo -e "${GREEN}✓ API is available at: http://localhost:8000${NC}"
            echo -e "${GREEN}✓ Swagger UI: http://localhost:8000/docs${NC}"
            echo ""
            echo -e "${BLUE}To view logs:${NC} ${SUDO_PREFIX} docker-compose -f docker-compose.yml logs -f $CONTAINER"
            echo -e "${BLUE}To stop:${NC} ${SUDO_PREFIX} ./scripts/run.sh stop"
        else
            echo -e "${RED}✗ Container failed to start. Checking logs:${NC}"
            $SUDO_PREFIX docker-compose -f docker-compose.yml logs "$CONTAINER"
        fi
    else
        $SUDO_PREFIX docker-compose -f docker-compose.yml "$action" "$CONTAINER"

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
        $SUDO_PREFIX docker-compose -f docker-compose.yml logs -f "$CONTAINER"
        ;;
    down)
        run_container "down"
        ;;
    build)
        $SUDO_PREFIX docker-compose -f docker-compose.yml build "$CONTAINER"
        ;;
    *)
        # Default action is to start
        run_container "up"
        ;;
esac

exit 0