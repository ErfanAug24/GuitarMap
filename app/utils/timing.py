import time
import functools
import logging
from contextlib import contextmanager
from typing import Optional

from app.utils.logging_config import get_logger


logger = get_logger(__name__)


class Timer:
    """Simple timing utility for measuring code execution."""

    def __init__(self, name: str = "Timer", logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or get_logger(__name__)
        self.start_time = None
        self.end_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        self.logger.info(f"{self.name} completed in {self.elapsed:.4f}s")

    def reset(self):
        self.start_time = time.perf_counter()
        self.end_time = None
        self.elapsed = None

    def stop(self):
        if self.start_time is None:
            raise RuntimeError("Timer has not been started.")
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        self.logger.debug(f"{self.name} stopped at {self.elapsed:.4f}s")
        return self.elapsed


def timing(func=None, *, name: Optional[str] = None, level: int = logging.INFO):
    """
    Decorator for timing function execution.
    Logs duration automatically via the project's logger.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = name or func.__name__
            start = time.perf_counter()
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start
            logger.log(level, f"{func_name} executed in {duration:.4f}s")
            return result

        return wrapper

    if func:
        return decorator(func)
    return decorator


@contextmanager
def time_block(name: str, level: int = logging.INFO):
    """
    Context manager version for timing arbitrary code blocks.
    Example:
        with time_block("Data preprocessing"):
            preprocess_data()
    """
    start = time.perf_counter()
    yield
    duration = time.perf_counter() - start
    logger.log(level, f"{name} took {duration:.4f}s")
