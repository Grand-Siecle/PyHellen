import functools
import os
from importlib import metadata

import torch

from app.core.environment import PIE_EXTENDED_DOWNLOADS
from app.core.settings import settings


def check_gpu_availability():
    """Check GPU availability"""
    if torch.cuda.is_available():
        return True, torch.cuda.get_device_name(0)
    return False, "No GPU available"


def get_device():
    """Get device based on availability"""
    gpu_available, _ = check_gpu_availability()
    return "cuda" if gpu_available else "cpu"


def quantization_enabled(device: str) -> bool:
    """INT8 quantization is opt-in (QUANTIZE_CPU) and CPU-only: quantized RNN ops have no CUDA kernels."""
    return settings.quantize_cpu and device == "cpu"


def model_fingerprint(module: str) -> str:
    """Identify everything that can change a model's output, so cached results are never reused across versions."""
    device = get_device()
    return _model_fingerprint(module, device, quantization_enabled(device))


@functools.lru_cache(maxsize=None)
def _model_fingerprint(module: str, device: str, quantize: bool) -> str:
    from pie_extended.cli.utils import get_model

    try:
        model_version = getattr(get_model(module), "VERSION", "")
    except ImportError:
        model_version = ""
    return (
        f"pie-extended={_distribution_version('pie-extended')}|papie={_distribution_version('PaPie')}"
        f"|{module}={model_version}|device={device}|quantize={quantize}"
    )


def _distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def get_n_workers():
    """Get numbers of cpu cores based on availability"""
    return os.cpu_count()


def get_path_models(module, file):
    return os.path.join(PIE_EXTENDED_DOWNLOADS, module, file)
