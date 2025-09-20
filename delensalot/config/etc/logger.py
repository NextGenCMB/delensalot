import logging
import logdecorator
import sys
import inspect

# All classes use this logger without inheritance
log = logging.getLogger("global_logger")

def set_logging_level(verbose: bool):
    pass

np_logger = logging.getLogger("numpy")
np_logger.setLevel(logging.WARNING)
np_logger = logging.getLogger("matplotlib")
np_logger.setLevel(logging.WARNING)
logging.getLogger("healpy").setLevel(logging.WARNING)
np_logger.setLevel(logging.WARNING)


class SafeDict(dict):
    """A dict that returns {key} if key is missing, so str.format won't crash."""
    def __missing__(self, key):
        return "{" + key + "}"

def safe_log_on_start(level, msg, logger):
    def decorator(func):
        sig = inspect.signature(func)

        def wrapper(*args, **kwargs):
            try:
                bound = sig.bind_partial(*args, **kwargs)
                bound.apply_defaults()

                # Safe formatting: missing keys just remain as {key}
                formatted_msg = msg.format_map(SafeDict(bound.arguments))

                logger.log(level, formatted_msg)
            except Exception as e:
                logger.warning(f"Logging failed inside safe_log_on_start: {e}")
                logger.debug(f"Args: {args}, Kwargs: {kwargs}")

            return func(*args, **kwargs)
        return wrapper
    return decorator

# Apply the patch globally
logdecorator.log_on_start = safe_log_on_start
logdecorator.log_on_end = safe_log_on_start
logdecorator.log_on_error = safe_log_on_start
