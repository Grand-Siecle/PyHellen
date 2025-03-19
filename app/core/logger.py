import logging

from app.constants import LOGGER_FILE, LOGGER_FORMAT, DATE_FORMAT

logging.basicConfig(
    filename=LOGGER_FILE,
    format=LOGGER_FORMAT,
    datefmt=DATE_FORMAT,
    level=logging.INFO
)

logger = logging.getLogger(__name__)