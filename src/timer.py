import time
from functools import wraps
from typing import Callable, Any

## Timings helper
timing_data = {}


def time_this(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        timing_data[func.__name__] = f"{elapsed_time:.5f}s"
        return result

    return wrapper



def time_many(runs: int = 100):
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            total_time = 0
            for _ in range(runs):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                total_time += end_time - start_time
            
            average_time = total_time / runs
            timing_data[func.__name__] = f"{average_time:.5f}s"
            return result
        return wrapper
    return decorator
