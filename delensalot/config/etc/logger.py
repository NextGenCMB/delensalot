import logging
import logdecorator
import sys

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


def safe_log_on_start(level, msg, logger):
    """Wrapper around log_on_start to catch formatting errors globally."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                formatted_msg = msg.format(*args, **kwargs)  # Try formatting first
                logger.log(level, formatted_msg)
            except Exception as e:
                logger.warning(f"Logging failed: {e}")  # Suppress long traceback

            return func(*args, **kwargs)  # Run the function normally

        return wrapper

    return decorator

# Apply the patch globally
logdecorator.log_on_start = safe_log_on_start
logdecorator.log_on_end = safe_log_on_start
logdecorator.log_on_error = safe_log_on_start
