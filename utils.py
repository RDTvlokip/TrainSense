# TrainSense/utils.py
import logging
import math # Used in format_bytesi
from typing import Union, Any, Optional

# Get a logger instance specific to this module
logger = logging.getLogger(__name__)

# --- Input Validation Functions ---

def validate_positive_integer(value: Any, name: str, allow_zero: bool = False):
    """
    Validates if the provided value is an integer and meets the positivity constraint.

    Args:
        value (Any): The value to validate.
        name (str): The name of the variable being validated (for error messages).
        allow_zero (bool): If True, allows the value to be zero. Otherwise, must be strictly positive.
                           Defaults to False.

    Raises:
        TypeError: If the value is not an integer.
        ValueError: If the value does not meet the positivity requirement.

    Returns:
        bool: True if validation passes.
    """
    if not isinstance(value, int):
        raise TypeError(f"'{name}' must be an integer, but got type {type(value).__name__}.")
    # Determine the minimum allowed value
    limit = 0 if allow_zero else 1
    if value < limit:
        condition = "positive or zero" if allow_zero else "strictly positive"
        raise ValueError(f"'{name}' must be {condition}, but got {value}.")
    return True # Indicate validation success

def validate_positive_float(value: Any, name: str, allow_zero: bool = False):
    """
    Validates if the provided value is a float (or int) and meets the positivity constraint.

    Args:
        value (Any): The value to validate.
        name (str): The name of the variable being validated (for error messages).
        allow_zero (bool): If True, allows the value to be zero or very close to zero.
                           Otherwise, must be strictly positive. Defaults to False.

    Raises:
        TypeError: If the value is not a float or integer.
        ValueError: If the value does not meet the positivity requirement.

    Returns:
        bool: True if validation passes.
    """
    if not isinstance(value, (float, int)): # Allow integers as they can be treated as floats
        raise TypeError(f"'{name}' must be a float or integer, but got type {type(value).__name__}.")
    # Determine the minimum allowed value using a small epsilon for float comparison
    limit = 0.0 if allow_zero else 1e-15 # Use a very small positive number instead of strict 0 for floats
    if float(value) < limit: # Convert to float for comparison
        condition = "positive or zero" if allow_zero else "strictly positive"
        raise ValueError(f"'{name}' must be {condition} (greater than ~{limit:.1e}), but got {value}.")
    return True # Indicate validation success

# --- Formatting Functions ---

def print_section(title: str = "", char: str = '=', length: int = 60):
    """
    Prints a formatted section header to the console, useful for separating output sections.

    Example:
        print_section("Results")
        -> ======== Results =========

        print_section()
        -> ==============================

    Args:
        title (str): The title text to display in the header. If empty, prints a simple separator line.
                     Defaults to "".
        char (str): The character used to draw the separator lines. Defaults to '='.
        length (int): The total desired length of the header line. Defaults to 60.
    """
    title_str = str(title) # Ensure title is a string
    # Ensure char is a single character, default to '=' if not
    line_char = char if isinstance(char, str) and len(char) == 1 else '='
    # Ensure length is positive
    line_length = max(10, length) # Minimum length of 10

    if not title_str:
        # Print a simple separator line if no title is provided
        print(line_char * line_length)
    else:
        # Calculate padding needed around the title
        padding = line_length - len(title_str) - 2 # Subtract 2 for spaces around title
        # Ensure padding is not negative if title is too long
        if padding < 0:
            # If title is longer than length, just print title with minimal decoration
            print(f"{line_char} {title_str} {line_char}")
        else:
            # Distribute padding evenly
            left_padding = padding // 2
            right_padding = padding - left_padding
            # Print the formatted header
            print(f"\n{line_char * left_padding} {title_str} {line_char * right_padding}\n")

def format_bytes(size_bytes: Optional[Union[int, float]]) -> str:
    """
    Converts a size in bytes to a human-readable string (e.g., "1.25 MB", "100 KB", "2.00 GB").

    Args:
        size_bytes (Optional[Union[int, float]]): The size in bytes. Handles None input.

    Returns:
        str: A formatted string representing the size, or "N/A" if input is None or negative.
    """
    # Handle invalid inputs
    if size_bytes is None or size_bytes < 0:
        return "N/A"
    if size_bytes == 0:
        return "0 B"

    # Units for byte sizes
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    # Determine the appropriate unit using logarithms
    # log base 1024 of size_bytes gives the index 'i'
    i = int(math.floor(math.log(size_bytes, 1024)))

    # Ensure index doesn't exceed available units (for extremely large numbers)
    i = min(i, len(size_name) - 1)

    # Calculate the value in the chosen unit
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2) # Round to 2 decimal places

    # Format the string
    formatted_size = f"{s:.2f}"
    # Remove trailing '.00' if present
    if formatted_size.endswith('.00'):
        formatted_size = formatted_size[:-3]

    return f"{formatted_size} {size_name[i]}"

def format_time(seconds: Optional[Union[int, float]]) -> str:
     """
     Converts a duration in seconds to a human-readable string
     (e.g., "120.50 ms", "3.45 s", "5 min 10.2 s", "1 hr 15 min", "2 d 4 hr").

     Args:
         seconds (Optional[Union[int, float]]): The duration in seconds. Handles None input.

     Returns:
         str: A formatted string representing the duration, or "N/A" if input is None or negative.
     """
     # Handle invalid inputs
     if seconds is None or seconds < 0:
         return "N/A"
     if seconds == 0:
         return "0 s" # Or "0 ms"? Let's stick to seconds for zero.

     # Milliseconds range
     if seconds < 1.0:
         return f"{seconds * 1000:.1f} ms"
     # Seconds range
     elif seconds < 60:
         return f"{seconds:.2f} s"
     # Minutes range
     elif seconds < 3600:
         minutes = int(seconds // 60)
         remaining_seconds = seconds % 60
         return f"{minutes} min {remaining_seconds:.1f} s"
     # Hours range
     elif seconds < 86400:
          hours = int(seconds // 3600)
          remaining_minutes = int((seconds % 3600) // 60)
          # Optionally include remaining seconds if significant? For now, just hours and minutes.
          return f"{hours} hr {remaining_minutes} min"
     # Days range
     else:
          days = int(seconds // 86400)
          remaining_hours = int((seconds % 86400) // 3600)
          return f"{days} d {remaining_hours} hr"