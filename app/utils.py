import torch


def check_gpu_availability():
    """Check GPU availability"""
    if torch.cuda.is_available():
        return True, torch.cuda.get_device_name(0)
    return False, "No GPU available"

def get_device():
    """Get device based on availability"""
    gpu_available, _ = check_gpu_availability()
    return "cuda" if gpu_available else "cpu"