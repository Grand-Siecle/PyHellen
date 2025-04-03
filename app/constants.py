import os
from pathlib import Path

LOGGER_FILE = Path(".") / "app.log"
DATE_FORMAT = "%d %b %Y | %H:%M:%S"
LOGGER_FORMAT = "%(asctime)s | %(message)s"

# PATHs default models
PIE_EXTENDED_DOWNLOADS = os.getenv(
    "PIE_EXTENDED_DOWNLOADS",
    os.path.join(os.path.expanduser("~"), ".local", "share", "pyhellen")
)
os.environ["PIE_EXTENDED_DOWNLOADS"] = PIE_EXTENDED_DOWNLOADS