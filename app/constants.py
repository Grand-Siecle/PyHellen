import os
from pathlib import Path

# Legacy logging constants (deprecated - use LOG_LEVEL and LOG_FORMAT env vars)
# Kept for backwards compatibility
LOGGER_FILE = os.getenv("LOG_FILE", "")  # Empty = stdout only
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOGGER_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

# PATHs default models
PIE_EXTENDED_DOWNLOADS = os.getenv(
    "PIE_EXTENDED_DOWNLOADS",
    os.path.join(os.path.expanduser("~"), ".local", "share", "pyhellen")
)
os.environ["PIE_EXTENDED_DOWNLOADS"] = PIE_EXTENDED_DOWNLOADS