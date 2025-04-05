# TrainSense/system_diagnostics.py
import psutil
import platform
import socket
import time
import logging
from typing import Dict, Any, Optional, Tuple # Added Tuple

logger = logging.getLogger(__name__)

class SystemDiagnostics:
    def __init__(self, cpu_interval: float = 1.0):
        """
        Args:
            cpu_interval: The interval in seconds over which to measure CPU usage.
                          A smaller interval is more responsive but might be less accurate
                          or impose slightly more overhead. Must be > 0.
        """
        if cpu_interval <= 0:
             raise ValueError("CPU interval must be positive.")
        self.cpu_interval = cpu_interval
        logger.info(f"SystemDiagnostics initialized with CPU interval {cpu_interval}s.")


    def diagnostics(self) -> Dict[str, Any]:
        """Gathers current system diagnostic information."""
        logger.debug("Starting system diagnostics gathering.")
        diag: Dict[str, Any] = {}

        # CPU Usage
        try:
             # psutil.cpu_percent(interval=None) requires a previous call to establish baseline.
             # Calling with interval blocks for that duration and gives usage during that interval.
             diag["cpu_usage_percent"] = psutil.cpu_percent(interval=self.cpu_interval)
             # Ensure the per-core measurement uses the now-established baseline
             diag["cpu_usage_per_core_percent"] = psutil.cpu_percent(interval=None, percpu=True)
        except Exception as e:
             logger.error(f"Failed to get CPU usage: {e}", exc_info=True)
             diag["cpu_usage_error"] = str(e)

        # Memory Usage
        try:
             mem = psutil.virtual_memory()
             diag["memory_total_bytes"] = mem.total
             diag["memory_available_bytes"] = mem.available
             diag["memory_used_bytes"] = mem.used
             diag["memory_usage_percent"] = mem.percent
             swap = psutil.swap_memory()
             diag["swap_total_bytes"] = swap.total
             diag["swap_used_bytes"] = swap.used
             diag["swap_usage_percent"] = swap.percent
        except Exception as e:
             logger.error(f"Failed to get Memory usage: {e}", exc_info=True)
             diag["memory_usage_error"] = str(e)

        # Disk Usage (Root Partition)
        try:
             disk = psutil.disk_usage('/')
             diag["disk_total_bytes"] = disk.total
             diag["disk_used_bytes"] = disk.used
             diag["disk_free_bytes"] = disk.free
             diag["disk_usage_percent"] = disk.percent
        except Exception as e:
             logger.error(f"Failed to get Disk usage for '/': {e}", exc_info=True)
             diag["disk_usage_error"] = str(e)


        # Network I/O Counters (System Wide)
        try:
             net_io = psutil.net_io_counters()
             diag["net_bytes_sent"] = net_io.bytes_sent
             diag["net_bytes_recv"] = net_io.bytes_recv
             diag["net_packets_sent"] = net_io.packets_sent
             diag["net_packets_recv"] = net_io.packets_recv
             diag["net_errin"] = net_io.errin
             diag["net_errout"] = net_io.errout
             diag["net_dropin"] = net_io.dropin
             diag["net_dropout"] = net_io.dropout
        except Exception as e:
             logger.error(f"Failed to get Network I/O counters: {e}", exc_info=True)
             diag["network_io_error"] = str(e)


        # Basic System Info (less likely to change often, but useful context)
        try:
             diag["os_info"] = f"{platform.system()} {platform.release()}"
             diag["hostname"] = socket.gethostname()
             diag["boot_timestamp"] = psutil.boot_time()
             diag["uptime_seconds"] = time.time() - psutil.boot_time()
        except Exception as e:
             logger.error(f"Failed to get basic system info (hostname/uptime): {e}", exc_info=True)
             diag["basic_info_error"] = str(e)

        logger.debug("System diagnostics gathering complete.")
        return diag

    def get_load_average(self) -> Optional[Tuple[float, float, float]]:
        """Returns the system load average (1, 5, 15 min). May not be available on Windows."""
        # Check if the function exists before calling, more robust than try-except AttributeError
        if hasattr(psutil, "getloadavg"):
            try:
                 load = psutil.getloadavg()
                 logger.debug(f"System load average: {load}")
                 return load
            except OSError as e: # Catch potential OS errors (e.g., on WSL1)
                 logger.warning(f"Could not retrieve load average: {e}")
                 return None
            except Exception as e:
                 logger.error(f"Failed to get load average: {e}", exc_info=True)
                 return None
        else:
             logger.warning("psutil.getloadavg() not available on this platform (likely Windows).")
             return None