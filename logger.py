# TrainSense/logger.py
import logging
import os
import sys
import datetime
from logging.handlers import RotatingFileHandler # For log file rotation

class TrainLogger:
    """
    A Singleton class to configure and provide a standardized logger for TrainSense.

    Features:
    - Configurable logging level.
    - Optional logging to a rotating file (to prevent excessively large log files).
    - Optional logging to the console (stdout).
    - Standardized log format including timestamp, logger name, level, and message.
    - Singleton pattern ensures only one instance configures the logger.
    """
    _instance = None # Class attribute to hold the singleton instance

    def __new__(cls, *args, **kwargs):
         """
         Implements the Singleton pattern. Ensures only one instance of TrainLogger is created.
         Subsequent calls to TrainLogger() will return the existing instance.
         """
         if cls._instance is None:
              # If no instance exists, create one using the standard object creation process
              cls._instance = super(TrainLogger, cls).__new__(cls)
              # Add a flag to track if __init__ has been run for this instance
              cls._instance._initialized = False
         return cls._instance

    def __init__(self,
                 log_file: str = "logs/trainsense.log", # Default log file path
                 level: int = logging.INFO,             # Default logging level
                 max_bytes: int = 10*1024*1024,        # Max size per log file (10 MB)
                 backup_count: int = 5,                 # Number of backup log files to keep
                 log_to_console: bool = True,           # Whether to also log to console
                 logger_name: str = "TrainSense"):      # Name of the logger
        """
        Initializes and configures the logger instance.
        This method is guarded by the `_initialized` flag set in `__new__` to ensure
        configuration happens only once, even if TrainLogger() is called multiple times.

        Args:
            log_file (str): Path to the log file. Directories will be created if they don't exist.
            level (int): The minimum logging level (e.g., logging.DEBUG, logging.INFO).
            max_bytes (int): Maximum size in bytes for a single log file before rotation.
            backup_count (int): Number of backup files to keep (e.g., trainsense.log.1, trainsense.log.2).
            log_to_console (bool): If True, logs will also be sent to standard output.
            logger_name (str): The name used for the logger instance retrieved via `logging.getLogger()`.
        """
        # Prevent re-initialization if already done
        if hasattr(self, '_initialized') and self._initialized:
            return

        # Get the specific logger instance
        self.logger = logging.getLogger(logger_name)

        # --- Check if logger is already configured (e.g., by root logger or previous run) ---
        # If handlers already exist, we assume it's configured elsewhere and avoid adding duplicates.
        # This prevents multiple identical log messages if TrainLogger is initialized inadvertently
        # after root logging has been set up.
        if self.logger.hasHandlers():
             # Optionally, could check if existing handlers match desired config, but simplest is to just exit.
             print(f"Logger '{logger_name}' already has handlers. Skipping reconfiguration.", file=sys.stderr) # Use stderr for setup messages
             self.logger.setLevel(min(self.logger.level, level)) # Ensure level is at least as verbose as requested
             self._initialized = True
             return
        # ------------------------------------------------------------------------------------

        # Set the overall minimum logging level for the logger instance
        self.logger.setLevel(level)

        # Define the standard log message format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', # Added file/line number
            datefmt='%Y-%m-%d %H:%M:%S' # Timestamp format
        )

        # --- Configure File Handler (Rotating) ---
        try:
            # Ensure the directory for the log file exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True) # Create directories recursively if needed

            # Create a handler that rotates log files when they reach max_bytes
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8' # Use UTF-8 encoding
            )
            file_handler.setLevel(level) # Set level for this handler
            file_handler.setFormatter(formatter) # Apply the standard format
            self.logger.addHandler(file_handler) # Add the handler to our logger
            log_file_path_for_msg = log_file
        except PermissionError as pe:
             # Log error to console if file logging setup fails due to permissions
             print(f"ERROR: Permission denied configuring file logging to {log_file}: {pe}", file=sys.stderr)
             log_file_path_for_msg = f"N/A (PermissionError: {log_file})"
        except Exception as e:
             # Log other errors during file handler setup
             print(f"ERROR: Failed to configure file logging to {log_file}: {e}", file=sys.stderr)
             log_file_path_for_msg = f"N/A (Error: {e})"
        # -----------------------------------------

        # --- Configure Console Handler ---
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout) # Log to standard output
            console_handler.setLevel(level) # Set level for this handler
            console_handler.setFormatter(formatter) # Apply the standard format
            self.logger.addHandler(console_handler) # Add the handler to our logger
        # ---------------------------------

        # --- Prevent Double Logging ---
        # Set propagate to False to prevent messages sent to this logger
        # from being passed up to the root logger (if the root logger also has handlers).
        # This avoids duplicate log entries if both 'TrainSense' and the root logger log to console/file.
        self.logger.propagate = False
        # ------------------------------

        # Mark this instance as initialized
        self._initialized = True
        # Log initialization message using the configured logger itself
        self.log_info(f"Logger '{logger_name}' initialized. Level: {logging.getLevelName(level)}. Log file: {log_file_path_for_msg}. Console logging: {log_to_console}.")


    # --- Logging Methods ---
    # Provide simple wrapper methods for common logging levels.

    def _log(self, level: int, message: str, exc_info: bool = False):
        """Internal helper to log messages."""
        # Check if logger was successfully initialized before trying to log
        if hasattr(self, 'logger'):
             self.logger.log(level, message, exc_info=exc_info)
        else:
             # Fallback if logger setup failed completely (should be rare)
             print(f"LOGGER NOT INITIALIZED: [{logging.getLevelName(level)}] {message}", file=sys.stderr)


    def log_debug(self, message: str):
        """Logs a message with level DEBUG."""
        self._log(logging.DEBUG, message)

    def log_info(self, message: str):
        """Logs a message with level INFO."""
        self._log(logging.INFO, message)

    def log_warning(self, message: str, exc_info: bool = False):
        """Logs a message with level WARNING."""
        self._log(logging.WARNING, message, exc_info=exc_info)

    def log_error(self, message: str, exc_info: bool = True):
        """Logs a message with level ERROR. Includes exception info by default."""
        self._log(logging.ERROR, message, exc_info=exc_info)

    def log_critical(self, message: str, exc_info: bool = True):
        """Logs a message with level CRITICAL. Includes exception info by default."""
        self._log(logging.CRITICAL, message, exc_info=exc_info)

    def get_logger(self) -> logging.Logger:
        """
        Returns the configured logger instance.

        Returns:
            logging.Logger: The configured logger instance for 'TrainSense'.
        """
        if not hasattr(self, 'logger'):
             # This case should ideally not be reached if __init__ runs correctly.
             print("ERROR: TrainLogger accessed before proper initialization!", file=sys.stderr)
             # Return a basic logger as a fallback
             fallback_logger = logging.getLogger("TrainSense_Fallback")
             if not fallback_logger.hasHandlers():
                  fallback_logger.addHandler(logging.StreamHandler(sys.stderr))
                  fallback_logger.setLevel(logging.WARNING)
             return fallback_logger
        return self.logger

# --- Global Access Function ---
# Provides a convenient way to get the globally configured TrainSense logger instance.
# Ensures that the logger is initialized (using defaults if called for the first time).

def get_trainsense_logger() -> logging.Logger:
    """
    Retrieves the globally configured TrainSense logger instance.
    Initializes the TrainLogger with default settings if it hasn't been initialized yet.

    Returns:
        logging.Logger: The configured 'TrainSense' logger instance.
    """
    # This implicitly calls TrainLogger.__new__ and TrainLogger.__init__ (if not already initialized)
    # using the default arguments of __init__.
    return TrainLogger().get_logger()