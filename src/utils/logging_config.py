"""
Logging configuration for the platform.
Sets up structured logging with appropriate levels and formats.

Author: Rohit Gangupantulu
"""

import logging
import sys
from pathlib import Path


def setup_logging(debug: bool = False):
    """
    Configure logging for the application.
    
    Args:
        debug: Enable debug level logging
    """
    # Determine log level
    log_level = logging.DEBUG if debug else logging.INFO
    
    # Create logs directory
    log_dir = Path("./outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "platform.log")
        ]
    )
    
    # Set specific loggers
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # Set our module loggers
    logging.getLogger("src.core").setLevel(log_level)
    logging.getLogger("src.processors").setLevel(log_level)
    logging.getLogger("src.integrations").setLevel(log_level)
