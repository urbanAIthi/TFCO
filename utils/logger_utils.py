import logging
from functools import wraps

def logger(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        setattr(wrapped, 'logger', logger)  # Setting the logger to the function
        return func(*args, **kwargs)
    return wrapped