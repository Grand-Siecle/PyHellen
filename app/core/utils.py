import torch
import os


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

def initialize_tagger(model_name):
    if model_name not in tagger_cache:
        try:
            tag[model_name] = get_tagger(model_name, batch_size=8, device=device)
        except Exception as e:
            print(f"Error initializing {model_name} tagger: {str(e)}")
            return None
    return taggers[model_name]