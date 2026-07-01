"""
src/utils/timer.py
Execution timing decorators for pipeline performance profiling.
"""

import functools
import time
from typing import Any, Callable, TypeVar

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def timer(func: F) -> F:
    """
    Decorator that logs wall-clock execution time of a function.

    Usage
    -----
    @timer
    def my_function(): ...
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("%-40s completed in %.2fs", func.__qualname__, elapsed)
        return result
    return wrapper  # type: ignore[return-value]


def stage_timer(stage_name: str) -> Callable[[F], F]:
    """
    Decorator factory with a custom stage label.

    Usage
    -----
    @stage_timer("PMFG Construction")
    def build_all_pmfgs(): ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger.info("Starting: %s", stage_name)
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info("Finished: %s — %.2fs (%.1f min)",
                        stage_name, elapsed, elapsed / 60)
            return result
        return wrapper  # type: ignore[return-value]
    return decorator
