import os
import sys
from pathlib import Path


def setup_environment():
    """
    Sets up the environment variables needed for the application
    """
    # Set PIE_EXTENDED_DOWNLOADS (overdrive package default path)
    default_path = os.path.join(os.path.expanduser("~"), ".local", "share", "pyhellen")
    pie_extended_downloads = os.getenv("PIE_EXTENDED_DOWNLOADS", default_path)
    Path(pie_extended_downloads).mkdir(parents=True, exist_ok=True)
    os.environ["PIE_EXTENDED_DOWNLOADS"] = pie_extended_downloads
    return pie_extended_downloads


PIE_EXTENDED_DOWNLOADS = setup_environment()