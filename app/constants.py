import os
from pathlib import Path

LOGGER_FILE = Path(".") / "app.log"
DATE_FORMAT = "%d %b %Y | %H:%M:%S"
LOGGER_FORMAT = "%(asctime)s | %(message)s"

# PATHs
DOWNLOAD_MODEL_PATH = os.getenv(
    "DOWNLOAD_MODEL_PATH",
    os.path.join(os.path.expanduser("~"), ".local", "share", "pyhellen")
)