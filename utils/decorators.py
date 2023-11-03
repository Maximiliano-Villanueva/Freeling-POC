"""
This file contains decoratoros to be used by modules
"""
import logging
import functools
import time

from functools import wraps


def time_tracker(output_format):
    """
    Track tine of a given function.
    Use it like this:
    
    @time_tracker("Function {func_name} executed in {exec_time} seconds.")
    def some_function():
        # Some computation here
        pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(output_format.format(func_name=func.__name__, exec_time=end_time - start_time))
            return result
        return wrapper
    return decorator


def retry(retries=3, wait_time=10):
    """Decorator for retrying a function call."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(retries):
                try:
                    logging.debug("Sending API request.")
                    return func(*args, **kwargs)
                except Exception as e:
                    logging.error(f"An exception occurred: {e}")
                    logging.warning(f"Retrying in {wait_time} seconds.")
                    time.sleep(wait_time)
            
            raise Exception("Max retries reached. Exiting.")
            
        return wrapper
    return decorator