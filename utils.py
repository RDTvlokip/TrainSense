# TrainSense/utils.py
import logging
from typing import Union

logger = logging.getLogger(__name__)

def validate_positive_integer(value: int, name: str, allow_zero: bool = False):
    """Raises ValueError if the value is not a positive integer."""
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    limit = 0 if allow_zero else 1
    if value < limit:
        raise ValueError(f"{name} must be greater than {'or equal to ' if allow_zero else ''}{limit-1}, got {value}.")
    return True

def validate_positive_float(value: float, name: str, allow_zero: bool = False):
    """Raises ValueError if the value is not a positive float."""
    if not isinstance(value, (float, int)): # Allow ints too
        raise TypeError(f"{name} must be a float or integer, got {type(value).__name__}.")
    limit = 0.0 if allow_zero else 1e-15 # Use small epsilon instead of 0 for float comparison
    if value < limit:
        raise ValueError(f"{name} must be positive{' or zero' if allow_zero else ''}, got {value}.")
    return True

def print_section(title: str, char: str = '=', length: int = 60):
    """Prints a formatted section header to the console."""
    if not title:
        print(char * length)
    else:
        padding = length - len(title) - 2 # Subtract 2 for spaces around title
        left_padding = padding // 2
        right_padding = padding - left_padding
        print(f"\n{char * left_padding} {title} {char * right_padding}\n")

def format_bytes(size_bytes: Union[int, float]) -> str:
    """Converts bytes to a human-readable string (KB, MB, GB, TB)."""
    if size_bytes is None or size_bytes < 0:
        return "N/A"
    if size_bytes == 0:
        return "0 B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = 0
    while size_bytes >= 1024 and i < len(size_name) - 1:
        size_bytes /= 1024.0
        i += 1
    s = f"{size_bytes:.2f}"
    # Remove trailing zeros after decimal point if they are '.00'
    s = s.rstrip('0').rstrip('.') if '.' in s else s
    return f"{s} {size_name[i]}"

def format_time(seconds: Union[int, float]) -> str:
     """Converts seconds to a human-readable string (ms, s, min, hr, days)."""
     if seconds is None or seconds < 0:
         return "N/A"
     if seconds == 0:
         return "0 s"
     if seconds < 1.0:
         return f"{seconds * 1000:.1f} ms"
     elif seconds < 60:
         return f"{seconds:.2f} s"
     elif seconds < 3600:
         minutes = int(seconds // 60)
         secs = seconds % 60
         return f"{minutes} min {secs:.1f} s"
     elif seconds < 86400:
          hours = int(seconds // 3600)
          minutes = int((seconds % 3600) // 60)
          return f"{hours} hr {minutes} min"
     else:
          days = int(seconds // 86400)
          hours = int((seconds % 86400) // 3600)
          return f"{days} d {hours} hr"