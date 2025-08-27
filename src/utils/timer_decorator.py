"""
timer_decorator.py
==================

This module provides a reusable decorator for timing function execution. 
It is designed for use in classes that maintain a `metrics` object 
(such as PathRAGMetrics) and automatically records execution time for 
specific functions.

Features
--------
- Measure function execution duration in seconds.
- Automatically record timing results in a metrics object.
- Log errors if the wrapped function raises exceptions.
- Thread-safe, since it only updates metrics via the provided interface.

Dependencies
------------
- functools.wraps : Preserve original function metadata.
- logging : Log errors if wrapped function fails.
- time : Track execution duration.

Example
-------
>>> from timer_decorator import timer
>>> from pathrag_metrics import PathRAGMetrics
>>>
>>> class Example:
...     def __init__(self):
...         self.metrics = PathRAGMetrics()
...
...     @timer("example_time")
...     def slow_method(self):
...         import time
...         time.sleep(0.5)
...
>>> ex = Example()
>>> ex.slow_method()
>>> print(ex.metrics.get_report())
{'graph_build_time': 0.0, 'retrieval_time': 0.0, 'path_pruning_time': 0.0,
 'nodes_processed': 0, 'edges_created': 0, 'paths_pruned': 0,
 'cache_hits': 0, 'cache_misses': 0, 'example_time': 0.50}
"""

from functools import wraps
import logging
import time


def timer(metric_name: str):
    """
    Decorator factory to time function execution and record metrics.

    Parameters
    ----------
    metric_name : str
        The name of the metric to record execution duration under.

    Returns
    -------
    Callable
        A decorator that wraps a function and records its execution time.

    Notes
    -----
    - The decorated function must belong to a class instance with a 
      `.metrics` attribute that supports `set_metric(name, value)`.
    - Errors in the wrapped function are logged and re-raised.

    Example
    -------
    >>> class MyClass:
    ...     def __init__(self, metrics):
    ...         self.metrics = metrics
    ...
    ...     @timer("process_time")
    ...     def process(self):
    ...         time.sleep(1)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            try:
                result = func(self, *args, **kwargs)
                duration = time.time() - start_time
                self.metrics.set_metric(metric_name, duration)
                return result
            except Exception as e:
                logging.error(f"Error in {func.__name__}: {e}", exc_info=True)
                raise
        return wrapper
    return decorator
