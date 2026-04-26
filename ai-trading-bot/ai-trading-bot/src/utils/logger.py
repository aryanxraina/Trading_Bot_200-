"""
logger.py
Structured logger — writes to console + daily log files.
"""

import logging
import os
from datetime import datetime


def get_logger(name: str, log_type: str = "trades") -> logging.Logger:
    """
    Args:
        name: module name e.g. 'download', 'order_manager'
        log_type: 'trades' or 'errors'
    """
    log_dir = f"logs/{log_type}"
    os.makedirs(log_dir, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    log_file = f"{log_dir}/{today}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
