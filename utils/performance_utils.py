"""
Performance optimization utilities for video processing
"""

import asyncio
import functools
import logging
import time
from typing import Any, Callable, Dict, Optional, TypeVar, Coroutine
from dataclasses import dataclass

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring"""

    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_usage: Optional[float] = None
    success: bool = True
    error: Optional[str] = None

    def complete(self, success: bool = True, error: Optional[str] = None):
        """Mark the operation as complete"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = success
        self.error = error


class PerformanceMonitor:
    """Monitor and log performance metrics"""

    def __init__(self):
        self.metrics: Dict[str, PerformanceMetrics] = {}

    def start_operation(self, operation_name: str) -> PerformanceMetrics:
        """Start tracking an operation"""
        metrics = PerformanceMetrics(
            operation_name=operation_name, start_time=time.time()
        )
        self.metrics[operation_name] = metrics
        return metrics

    def complete_operation(
        self, operation_name: str, success: bool = True, error: Optional[str] = None
    ):
        """Complete tracking an operation"""
        if operation_name in self.metrics:
            self.metrics[operation_name].complete(success, error)
            metrics = self.metrics[operation_name]

            if success:
                logger.info(
                    "✅ %s completed in %.2fs", operation_name, metrics.duration
                )
            else:
                logger.error(
                    "❌ %s failed after %.2fs: %s",
                    operation_name,
                    metrics.duration,
                    error,
                )

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        total_time = sum(m.duration or 0 for m in self.metrics.values())
        successful_ops = sum(1 for m in self.metrics.values() if m.success)
        failed_ops = len(self.metrics) - successful_ops

        return {
            "total_operations": len(self.metrics),
            "successful_operations": successful_ops,
            "failed_operations": failed_ops,
            "total_time": total_time,
            "operations": {
                name: {"duration": m.duration, "success": m.success, "error": m.error}
                for name, m in self.metrics.items()
            },
        }


def performance_monitor(operation_name: Optional[str] = None):
    """Decorator to monitor function performance"""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        op_name = operation_name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            monitor = PerformanceMonitor()
            _ = monitor.start_operation(op_name)  # Store metrics for tracking

            try:
                result = func(*args, **kwargs)
                monitor.complete_operation(op_name, success=True)
                return result
            except (ValueError, TypeError, IOError) as e:
                monitor.complete_operation(op_name, success=False, error=str(e))
                raise

        return wrapper

    return decorator


def async_performance_monitor(operation_name: Optional[str] = None):
    """Decorator to monitor async function performance"""

    def decorator(
        func: Callable[..., Coroutine[Any, Any, T]],
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        op_name = operation_name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            monitor = PerformanceMonitor()
            _ = monitor.start_operation(op_name)  # Store metrics for tracking

            try:
                result = await func(*args, **kwargs)
                monitor.complete_operation(op_name, success=True)
                return result
            except (ValueError, TypeError, IOError) as e:
                monitor.complete_operation(op_name, success=False, error=str(e))
                raise

        return wrapper

    return decorator


class SimpleCache:
    """Simple in-memory cache with weak references"""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache: Dict[str, Any] = {}
        self._access_times: Dict[str, float] = {}

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self._cache:
            self._access_times[key] = time.time()
            return self._cache[key]
        return None

    def set(self, key: str, value: Any):
        """Set value in cache"""
        # Remove oldest items if cache is full
        if len(self._cache) >= self.max_size:
            oldest_key = min(
                self._access_times.keys(), key=lambda k: self._access_times[k]
            )
            del self._cache[oldest_key]
            del self._access_times[oldest_key]

        self._cache[key] = value
        self._access_times[key] = time.time()

    def clear(self):
        """Clear cache"""
        self._cache.clear()
        self._access_times.clear()


# Global instances
performance_monitor_instance = PerformanceMonitor()
simple_cache = SimpleCache()


# Retry decorator with exponential backoff
def retry_with_backoff(
    max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0
):
    """Decorator for retrying async operations with exponential backoff"""

    def decorator(
        func: Callable[..., Coroutine[Any, Any, T]],
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except (ValueError, TypeError, IOError) as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (2**attempt), max_delay)
                        logger.warning(
                            "Attempt %d failed, retrying in %.1fs: %s",
                            attempt + 1,
                            delay,
                            e,
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error("All %d attempts failed: %s", max_retries + 1, e)

            # This should never be None, but satisfy type checker
            if last_exception:
                raise last_exception
            else:
                raise RuntimeError("Retry failed with no exception recorded")

        return wrapper

    return decorator
