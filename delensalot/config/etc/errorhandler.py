class DelensalotError(Exception):
    """Custom exception class for Delensalot-related errors."""
    def __init__(self, message):
        super().__init__(f" - {message}")

    def __str__(self):
        # ANSI escape code for light blue text for the exception name
        message = f"\033[94m{super().__str__()}\033[0m"
        return f" {message}"