# TrainSense/logger.py
import logging
import os
import sys
import datetime
from logging.handlers import RotatingFileHandler

class TrainLogger:
    _instance = None

    def __new__(cls, *args, **kwargs):
         if cls._instance is None:
              cls._instance = super(TrainLogger, cls).__new__(cls)
              cls._instance._initialized = False
         return cls._instance

    def __init__(self,
                 log_file: str = "logs/trainsense.log",
                 level: int = logging.INFO,
                 max_bytes: int = 10*1024*1024, # 10 MB
                 backup_count: int = 5,
                 log_to_console: bool = True,
                 logger_name: str = "TrainSense"):

        if self._initialized:
            return

        self.logger = logging.getLogger(logger_name)
        if self.logger.hasHandlers():
             # If handlers are already configured (e.g., by root logger), don't reconfigure
             # Or allow forceful reconfiguration if needed via a flag?
             # For now, assume if handlers exist, it's configured elsewhere.
             self._initialized = True
             # Optionally check if level matches and adjust?
             # self.logger.setLevel(level)
             return


        self.logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        # File Handler (Rotating)
        try:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        except Exception as e:
             # Log error to console if file logging fails
             print(f"ERROR: Failed to configure file logging to {log_file}: {e}", file=sys.stderr)


        # Console Handler
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        self.logger.propagate = False # Prevent messages from being passed to the root logger if handlers are added here
        self._initialized = True
        self.log_info(f"Logger '{logger_name}' initialized. Level: {logging.getLevelName(level)}. Log file: {log_file if 'file_handler' in locals() else 'N/A'}. Console logging: {log_to_console}.")


    def _log(self, level: int, message: str, exc_info: bool = False):
        self.logger.log(level, message, exc_info=exc_info)

    def log_debug(self, message: str):
        self._log(logging.DEBUG, message)

    def log_info(self, message: str):
        self._log(logging.INFO, message)

    def log_warning(self, message: str, exc_info: bool = False):
        self._log(logging.WARNING, message, exc_info=exc_info)

    def log_error(self, message: str, exc_info: bool = True):
        self._log(logging.ERROR, message, exc_info=exc_info)

    def log_critical(self, message: str, exc_info: bool = True):
        self._log(logging.CRITICAL, message, exc_info=exc_info)

    def get_logger(self) -> logging.Logger:
        return self.logger

# Global access function (optional, promotes using the configured instance)
def get_trainsense_logger() -> logging.Logger:
    return TrainLogger().get_logger()