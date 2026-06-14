"""
src/utils/config_loader.py
Loads and caches configuration from YAML files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

_CONFIG_CACHE: dict[str, Any] | None = None
_CRISIS_CACHE: dict[str, Any] | None = None



def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return contents as dict."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8-sig") as f:
        data = yaml.safe_load(f)

    logger.debug("Loaded config: %s", path)
    return data

def get_config(config_dir: str = "config") -> dict[str, Any]:
    """
    Load and cache main project settings.

    Parameters
    ----------
    config_dir : str
        Path to config directory.

    Returns
    -------
    dict[str, Any]
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        path = Path(config_dir) / "settings.yaml"
        _CONFIG_CACHE = _load_yaml(path)
        logger.info("Project config loaded from %s", path)
    return _CONFIG_CACHE


def get_crisis_periods(config_dir: str = "config") -> dict[str, Any]:
    """
    Load and cache crisis period definitions.

    Parameters
    ----------
    config_dir : str
        Path to config directory.

    Returns
    -------
    dict[str, Any]
    """
    global _CRISIS_CACHE
    if _CRISIS_CACHE is None:
        path = Path(config_dir) / "crisis_periods.yaml"
        _CRISIS_CACHE = _load_yaml(path)
        logger.info(
            "Crisis periods loaded: %d crises defined",
            len(_CRISIS_CACHE.get("crises", []))
        )
    return _CRISIS_CACHE
