#!/usr/bin/env python3
"""
Script to detect if NVIDIA GPU is available and set up the appropriate container
"""

import subprocess
import sys
import os


def has_nvidia_gpu():
    """Check if NVIDIA GPU is available on the system"""
    try:
        # Try to run nvidia-smi
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def has_nvidia_docker():
    """Check if nvidia-docker is available"""
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{json .Runtimes}}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False
        )
        return "nvidia" in result.stdout.decode()
    except FileNotFoundError:
        return False


def main():
    """Main function to detect GPU and suggest the right container"""
    has_gpu = has_nvidia_gpu()
    has_docker_support = has_nvidia_docker()

    print("GPU Detection Results:")
    print(f"- NVIDIA GPU detected: {'Yes' if has_gpu else 'No'}")
    print(f"- NVIDIA Docker support: {'Yes' if has_docker_support else 'No'}")

    if has_gpu and has_docker_support:
        print("\n✅ Your system supports GPU acceleration!")
        print("Recommended docker-compose command:")
        print("docker-compose -f docker/docker-compose.yml up pyhellen-gpu")
        return 0
    elif has_gpu and not has_docker_support:
        print("\n⚠️ NVIDIA GPU detected but nvidia-docker is not set up correctly.")
        print("To use GPU acceleration, install nvidia-docker:")
        print("https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html")
        print("\nRunning with CPU mode for now:")
        print("docker-compose -f docker/docker-compose.yml up pyhellen")
        return 1
    else:
        print("\nNo NVIDIA GPU detected. Using CPU mode:")
        print("docker-compose -f docker/docker-compose.yml up pyhellen")
        return 2


if __name__ == "__main__":
    sys.exit(main())