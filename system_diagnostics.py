# TrainSense/system_diagnostics.py
import psutil      # For retrieving system resource usage (CPU, RAM, Disk, Network)
import platform    # For OS info
import socket      # For hostname
import time        # For uptime calculation
import logging
from typing import Dict, Any, Optional, Tuple

# Get a logger instance specific to this module
logger = logging.getLogger(__name__)

class SystemDiagnostics:
    """
    Gathers dynamic system diagnostic information at runtime.

    Provides methods to retrieve current CPU usage, memory (RAM/Swap) usage,
    disk usage (root partition), network I/O counters, and system load average.
    """

    def __init__(self, cpu_interval: float = 0.5):
        """
        Initializes the SystemDiagnostics component.

        Args:
            cpu_interval (float): The interval in seconds over which `psutil.cpu_percent()`
                                  calculates CPU usage when called within `diagnostics()`.
                                  A smaller interval (e.g., 0.1) provides more instantaneous
                                  readings but might have slightly higher overhead. A larger
                                  interval (e.g., 1.0) provides a smoother average over that period.
                                  Must be a positive value. Defaults to 0.5 seconds.

        Raises:
            ValueError: If `cpu_interval` is not positive.
        """
        if cpu_interval <= 0:
             raise ValueError("CPU interval must be positive.")
        # Store the interval for cpu_percent calls
        self.cpu_interval = cpu_interval
        # Perform an initial call to cpu_percent to establish a baseline for subsequent non-blocking calls.
        # This prevents the first non-blocking call from returning 0.0.
        try:
             psutil.cpu_percent(interval=None)
        except Exception as e:
             logger.warning(f"Initial call to psutil.cpu_percent failed: {e}. First diagnostic reading might be inaccurate.")
        logger.info(f"SystemDiagnostics initialized with CPU measurement interval {self.cpu_interval}s.")


    def diagnostics(self) -> Dict[str, Any]:
        """
        Gathers a snapshot of current system diagnostic information.

        Returns:
            Dict[str, Any]: A dictionary containing various metrics:
                - 'timestamp': Time of measurement (epoch seconds).
                - 'cpu_usage_percent': Overall CPU utilization percentage (average over `cpu_interval`).
                - 'cpu_usage_per_core_percent': List of CPU utilization percentages for each logical core.
                - 'memory_total_bytes', 'memory_available_bytes', 'memory_used_bytes': RAM stats.
                - 'memory_usage_percent': Overall RAM utilization percentage.
                - 'swap_total_bytes', 'swap_used_bytes', 'swap_usage_percent': Swap memory stats.
                - 'disk_total_bytes', 'disk_used_bytes', 'disk_free_bytes', 'disk_usage_percent': Root ('/') disk usage.
                - 'net_bytes_sent', 'net_bytes_recv', 'net_packets_sent', 'net_packets_recv': System-wide network counters.
                - 'net_errin', 'net_errout', 'net_dropin', 'net_dropout': System-wide network error/drop counters.
                - 'os_info': Basic OS string (e.g., 'Linux 5.4.0-100-generic').
                - 'hostname': System hostname.
                - 'boot_timestamp': System boot time (epoch seconds).
                - 'uptime_seconds': System uptime in seconds.
                - May also contain keys ending in '_error' if specific metrics failed to be retrieved.
        """
        logger.debug("Starting system diagnostics gathering.")
        diag: Dict[str, Any] = {"timestamp": time.time()} # Start with timestamp

        # --- CPU Usage ---
        try:
             # Blocking call: measures CPU usage over the specified interval
             diag["cpu_usage_percent"] = psutil.cpu_percent(interval=self.cpu_interval)
             # Non-blocking call (per core): returns usage since the *last* call (either blocking or non-blocking)
             # Requires the previous blocking call to have established a baseline.
             diag["cpu_usage_per_core_percent"] = psutil.cpu_percent(interval=None, percpu=True)
        except Exception as e:
             logger.error(f"Failed to get CPU usage: {e}", exc_info=True)
             diag["cpu_usage_error"] = str(e)

        # --- Memory (RAM & Swap) Usage ---
        try:
             mem = psutil.virtual_memory()
             diag["memory_total_bytes"] = mem.total
             diag["memory_available_bytes"] = mem.available # Memory readily available
             diag["memory_used_bytes"] = mem.used       # Memory used (total - free - buffers/cache can differ)
             diag["memory_usage_percent"] = mem.percent   # Overall RAM usage percentage

             swap = psutil.swap_memory()
             diag["swap_total_bytes"] = swap.total
             diag["swap_used_bytes"] = swap.used
             diag["swap_usage_percent"] = swap.percent
        except Exception as e:
             logger.error(f"Failed to get Memory usage: {e}", exc_info=True)
             diag["memory_usage_error"] = str(e)

        # --- Disk Usage (Root Partition '/') ---
        try:
             # Get usage for the root filesystem. Change '/' to another path if needed.
             disk = psutil.disk_usage('/')
             diag["disk_total_bytes"] = disk.total
             diag["disk_used_bytes"] = disk.used
             diag["disk_free_bytes"] = disk.free
             diag["disk_usage_percent"] = disk.percent
        except FileNotFoundError:
             logger.warning("Root path '/' not found for disk usage. Skipping disk diagnostics.")
             diag["disk_usage_error"] = "Root path '/' not found."
        except Exception as e:
             logger.error(f"Failed to get Disk usage for '/': {e}", exc_info=True)
             diag["disk_usage_error"] = str(e)


        # --- Network I/O Counters (System Wide) ---
        try:
             # Get system-wide network statistics since boot (or last counter reset)
             net_io = psutil.net_io_counters()
             diag["net_bytes_sent"] = net_io.bytes_sent
             diag["net_bytes_recv"] = net_io.bytes_recv
             diag["net_packets_sent"] = net_io.packets_sent
             diag["net_packets_recv"] = net_io.packets_recv
             diag["net_errin"] = net_io.errin     # Input errors
             diag["net_errout"] = net_io.errout   # Output errors
             diag["net_dropin"] = net_io.dropin   # Input packets dropped
             diag["net_dropout"] = net_io.dropout # Output packets dropped (often 0 on Linux)
        except Exception as e:
             logger.error(f"Failed to get Network I/O counters: {e}", exc_info=True)
             diag["network_io_error"] = str(e)


        # --- Basic System Info (less dynamic, but useful context) ---
        try:
             diag["os_info"] = f"{platform.system()} {platform.release()}"
             diag["hostname"] = socket.gethostname()
             # Get system boot time (seconds since epoch)
             boot_ts = psutil.boot_time()
             diag["boot_timestamp"] = boot_ts
             # Calculate uptime based on current time and boot time
             diag["uptime_seconds"] = time.time() - boot_ts
        except Exception as e:
             logger.error(f"Failed to get basic system info (hostname/uptime): {e}", exc_info=True)
             diag["basic_info_error"] = str(e)

        logger.debug("System diagnostics gathering complete.")
        return diag

    def get_load_average(self) -> Optional[Tuple[float, float, float]]:
        """
        Returns the system load average over the last 1, 5, and 15 minutes.
        This metric represents the average number of processes in the system run queue
        (running or waiting for CPU).

        Note: This function is typically available on Unix-like systems (Linux, macOS)
              but *not* on Windows.

        Returns:
            Optional[Tuple[float, float, float]]: A tuple containing the 1, 5, and 15-minute
                                                 load averages, or None if the metric is
                                                 unavailable on the current platform or an
                                                 error occurred.
        """
        # Check if the function exists in psutil for the current platform
        if hasattr(psutil, "getloadavg"):
            try:
                 load_tuple = psutil.getloadavg() # Returns (1min, 5min, 15min) tuple
                 logger.debug(f"System load average (1, 5, 15 min): {load_tuple}")
                 return load_tuple
            except OSError as e:
                 # Catch specific OS errors (e.g., occurs on some WSL versions)
                 logger.warning(f"Could not retrieve load average (OS Error): {e}")
                 return None
            except Exception as e:
                 # Catch any other unexpected errors
                 logger.error(f"Unexpected error getting load average: {e}", exc_info=True)
                 return None
        else:
             # Function doesn't exist (likely Windows)
             logger.info("psutil.getloadavg() is not available on this platform (likely Windows). Load average cannot be retrieved.")
             return None