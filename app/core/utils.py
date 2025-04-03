import torch
import os

from app.core.environment import PIE_EXTENDED_DOWNLOADS


def check_gpu_availability():
    """Check GPU availability"""
    if torch.cuda.is_available():
        return True, torch.cuda.get_device_name(0)
    return False, "No GPU available"

def get_device():
    """Get device based on availability"""
    gpu_available, _ = check_gpu_availability()
    return "cuda" if gpu_available else "cpu"

def get_n_workers():
    """Get numbers of cpu cores based on availability"""
    return os.cpu_count()

def get_path_models(module, file):
    return os.path.join(PIE_EXTENDED_DOWNLOADS, module, file)
