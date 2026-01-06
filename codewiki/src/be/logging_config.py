"""
Logging configuration for backend modules.

This module provides centralized logging configuration for all backend modules.
Import this module first in all backend files to ensure proper logging configuration.

Usage:
    from codewiki.src.be.logging_config import get_logger

    logger = get_logger(__name__)
"""

import logging
import sys
from typing import Optional


def configure_logging(level: Optional[int] = None) -> None:
    """
    Configure root logger for the backend.

    Args:
        level: Logging level (default: INFO)
               Options: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR

    Note:
        This function should be called once at application startup.
        Multiple calls will reconfigure the logger.
    """
    if level is None:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance

    Example:
        from codewiki.src.be.logging_config import get_logger

        logger = get_logger(__name__)
        logger.info("This is a log message")
    """
    return logging.getLogger(name)


_logging_configured = False


def ensure_logging_configured() -> None:
    """
    Ensure logging is configured at least once.

    This is called automatically when get_logger is first used.
    """
    global _logging_configured
    if not _logging_configured:
        configure_logging()
        _logging_configured = True
