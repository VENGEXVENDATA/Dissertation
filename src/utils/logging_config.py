"""
src/utils/logging_config.py
Centralised logging setup for the dissertation pipeline.
"""

import logging
from pathlib import Path


def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Create and return a configured logger.

    Parameters
    ----------
    name : str
        Logger name — use __name__ from calling module.
    log_dir : str
        Directory for log files. Created if missing.

    Returns
    -------
    logging.Logger
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    log_path = Path(log_dir) / f"{name.replace('.', '_')}.log"
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger
