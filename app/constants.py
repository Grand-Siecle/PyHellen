import os


# PATHs
DOWNLOAD_MODEL_PATH = os.getenv(
    "DOWNLOAD_MODEL_PATH",
    os.path.join(os.path.expanduser("~"), ".local", "share", "pyhellen")
)