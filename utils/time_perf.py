import time
from functools import wraps


def time_performance_measure(func):
    """
    A decorator to measure the execution time of a function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # Use perf_counter for precise timing
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000
        print(f"Function '{func.__name__}' took {execution_time:.4f} ms to execute.")
        return result

    return wrapper
