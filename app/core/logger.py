"""
Logging configuration for PyHellen API.

Supports both development (human-readable) and production (JSON) formats.
Logs to stdout by default for Docker/Kubernetes compatibility.
"""

import logging
import json
import sys
import os
from datetime import datetime
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging in production.

    Output format compatible with ELK, CloudWatch, and other log aggregators.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        # Add request context if available
        for field in ["request_id", "method", "path", "status_code", "duration_ms"]:
            if hasattr(record, field):
                log_data[field] = getattr(record, field)

        return json.dumps(log_data, default=str)


class DevelopmentFormatter(logging.Formatter):
    """Human-readable formatter for local development."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        # Check if terminal supports colors
        use_colors = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname

        if use_colors:
            color = self.COLORS.get(level, "")
            level_str = f"{color}{level:8}{self.RESET}"
        else:
            level_str = f"{level:8}"

        message = f"{timestamp} | {level_str} | {record.name} | {record.getMessage()}"

        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)

        return message


def setup_logging(
    log_level: str = None,
    json_format: bool = None,
    log_file: str = None
) -> logging.Logger:
    """
    Configure logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON format (True for production, False for development)
        log_file: Optional file path to also write logs (in addition to stdout)

    Environment variables:
        LOG_LEVEL: Sets the logging level (default: INFO)
        LOG_FORMAT: 'json' for production, 'text' for development (default: auto-detect)
        LOG_FILE: Optional file path for logging

    Returns:
        Configured logger instance
    """
    # Determine settings from environment or parameters
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    if json_format is None:
        env_format = os.getenv("LOG_FORMAT", "auto").lower()
        if env_format == "json":
            json_format = True
        elif env_format == "text":
            json_format = False
        else:
            # Auto-detect: use JSON in Docker/production, text in development
            json_format = os.getenv("DOCKER_CONTAINER", "") == "true" or \
                          os.getenv("KUBERNETES_SERVICE_HOST") is not None

    if log_file is None:
        log_file = os.getenv("LOG_FILE")

    # Get or create root logger for the app
    root_logger = logging.getLogger("pyhellen")
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Choose formatter based on environment
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = DevelopmentFormatter()

    # Always add stdout handler (required for Docker/Kubernetes)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.setLevel(getattr(logging, log_level, logging.INFO))
    root_logger.addHandler(stdout_handler)

    # Optionally add file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, log_level, logging.INFO))
        root_logger.addHandler(file_handler)

    # Prevent propagation to root logger
    root_logger.propagate = False

    # Also configure uvicorn loggers to use same format
    for uvicorn_logger in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
        uv_log = logging.getLogger(uvicorn_logger)
        uv_log.handlers.clear()
        uv_log.addHandler(stdout_handler)
        uv_log.propagate = False

    return root_logger


# Initialize logger on module import
logger = setup_logging()


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (will be prefixed with 'pyhellen.')

    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f"pyhellen.{name}")
    return logging.getLogger("pyhellen")
