# TrainSense/monitoring.py
import time
import threading
import logging
from typing import List, Dict, Any, Optional, Union
import psutil
import torch # Pour vérifier torch.cuda si besoin

# Importer depuis les modules TrainSense existants
from .system_diagnostics import SystemDiagnostics
from .gpu_monitor import GPUMonitor

logger = logging.getLogger(__name__)

class RealTimeMonitor:
    """
    Monitors system (CPU, RAM) and GPU resources in a separate thread
    at regular intervals during a code block execution.
    """
    def __init__(self, interval_sec: float = 5.0, monitor_gpu: bool = True):
        """
        Initializes the RealTimeMonitor.

        Args:
            interval_sec (float): How often to collect metrics (in seconds).
            monitor_gpu (bool): Whether to attempt GPU monitoring (requires GPUtil).
        """
        validate_positive_float(interval_sec, "Monitoring interval", allow_zero=False)
        self.interval_sec = interval_sec
        self.monitor_gpu = monitor_gpu
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.history: List[Dict[str, Any]] = []
        self.system_diag = SystemDiagnostics(cpu_interval=0.1) # Use short interval for diagnostics snapshot
        self.gpu_monitor: Optional[GPUMonitor] = None
        self.gpu_available = False

        if self.monitor_gpu:
            try:
                self.gpu_monitor = GPUMonitor()
                self.gpu_available = self.gpu_monitor.is_available()
                if not self.gpu_available:
                    logger.warning("GPU monitoring requested but GPUtil/GPU not available.")
            except Exception as e:
                 logger.error(f"Failed to initialize GPUMonitor: {e}", exc_info=True)
                 self.gpu_available = False
        else:
             logger.info("GPU monitoring explicitly disabled.")


        self.monitoring_active = False
        logger.info(f"RealTimeMonitor initialized with interval {self.interval_sec}s. GPU Monitoring: {'Enabled' if self.monitor_gpu and self.gpu_available else 'Disabled'}.")

    def _monitor_loop(self):
        """The target function for the monitoring thread."""
        logger.info(f"Monitoring thread started (interval: {self.interval_sec}s).")
        while not self._stop_event.is_set():
            start_time = time.perf_counter()
            snapshot = {"timestamp": time.time()}

            # System Diagnostics
            try:
                diag = self.system_diag.diagnostics()
                # Select key metrics to store
                snapshot['cpu_usage_percent'] = diag.get('cpu_usage_percent')
                snapshot['memory_usage_percent'] = diag.get('memory_usage_percent')
                snapshot['memory_used_gb'] = diag.get('memory_used_bytes', 0) / (1024**3) if diag.get('memory_used_bytes') is not None else None
            except Exception as e:
                logger.error(f"Error getting system diagnostics in thread: {e}", exc_info=False) # Avoid flooding logs
                snapshot['system_error'] = str(e)

            # GPU Diagnostics (if enabled and available)
            if self.monitor_gpu and self.gpu_available and self.gpu_monitor:
                try:
                    gpu_statuses = self.gpu_monitor.get_gpu_status()
                    snapshot['gpu_status'] = gpu_statuses # Store list of dicts
                except Exception as e:
                     logger.error(f"Error getting GPU status in thread: {e}", exc_info=False)
                     snapshot['gpu_error'] = str(e)
            elif self.monitor_gpu and not self.gpu_available:
                 snapshot['gpu_status'] = "N/A (Unavailable)"


            self.history.append(snapshot)
            # logger.debug(f"Monitor snapshot taken: {snapshot}") # Can be very verbose

            # Wait for the next interval, accounting for the time taken by diagnostics
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            sleep_time = self.interval_sec - elapsed
            if sleep_time > 0:
                # Use wait with a timeout to be responsive to the stop event
                self._stop_event.wait(timeout=sleep_time)
            # If diagnostics took longer than interval, loop immediately (or log warning?)

        logger.info("Monitoring thread stopped.")

    def start(self):
        """Starts the monitoring thread."""
        if self.monitoring_active:
            logger.warning("Monitoring is already active.")
            return

        if self._thread is not None and self._thread.is_alive():
             logger.warning("Monitoring thread seems alive but inactive flag set. Attempting to rejoin.")
             self._stop_event.set()
             self._thread.join(timeout=self.interval_sec * 2) # Wait briefly

        self._stop_event.clear()
        self.history = [] # Clear history from previous runs
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True) # Daemon thread exits if main exits
        self._thread.start()
        self.monitoring_active = True
        logger.info("RealTimeMonitor started.")

    def stop(self) -> List[Dict[str, Any]]:
        """Stops the monitoring thread and returns the collected history."""
        if not self.monitoring_active or self._thread is None:
            logger.warning("Monitoring is not active or thread not initialized.")
            return self.history

        logger.info("Stopping RealTimeMonitor...")
        self._stop_event.set()
        # Wait for the thread to finish cleanly
        self._thread.join(timeout=self.interval_sec * 2) # Add a timeout
        if self._thread.is_alive():
             logger.warning("Monitoring thread did not stop cleanly within timeout.")

        self.monitoring_active = False
        self._thread = None
        logger.info(f"RealTimeMonitor stopped. Collected {len(self.history)} snapshots.")
        return self.history

    def get_history(self) -> List[Dict[str, Any]]:
        """Returns the currently collected history without stopping."""
        return self.history

    # Context manager support
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        # Can optionally re-raise exceptions if needed:
        # if exc_type is not None:
        #     # Handle or log exception if needed
        #     pass
        # return False # To re-raise exception

# Helper function from utils - duplicated here for standalone module if needed,
# but ideally should be imported if utils is stable.
def validate_positive_float(value: float, name: str, allow_zero: bool = False):
    """Raises ValueError if the value is not a positive float."""
    if not isinstance(value, (float, int)):
        raise TypeError(f"{name} must be a float or integer, got {type(value).__name__}.")
    limit = 0.0 if allow_zero else 1e-15
    if value < limit:
        raise ValueError(f"{name} must be positive{' or zero' if allow_zero else ''}, got {value}.")
    return True