# TrainSense/gpu_monitor.py
import logging
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

try:
    import GPUtil
except ImportError:
    GPUtil = None
    logger.warning("GPUtil library not found. GPU monitoring features will be unavailable. Install with 'pip install GPUtil'")

class GPUMonitor:
    def __init__(self):
        if GPUtil is None:
            self._available = False
            logger.error("GPUtil is not installed. Cannot monitor GPUs.")
        else:
             try:
                 # Attempt to get GPUs to check if the driver/library link is working
                 GPUtil.getGPUs()
                 self._available = True
                 logger.info("GPUMonitor initialized successfully.")
             except Exception as e:
                 logger.error(f"Failed to initialize GPUtil or access NVIDIA drivers: {e}. GPU monitoring unavailable.", exc_info=True)
                 self._available = False


    def is_available(self) -> bool:
        return self._available

    def get_gpu_status(self) -> List[Dict[str, Any]]:
        if not self._available:
            logger.warning("Attempted to get GPU status, but GPUtil is not available or failed to initialize.")
            return []

        try:
            gpus = GPUtil.getGPUs()
            status = []
            for gpu in gpus:
                status.append({
                    "id": gpu.id,
                    "uuid": gpu.uuid,
                    "name": gpu.name,
                    "load": gpu.load * 100,  # Percentage
                    "memory_used_mb": gpu.memoryUsed,
                    "memory_total_mb": gpu.memoryTotal,
                    "memory_free_mb": gpu.memoryFree,
                    "memory_utilization_percent": gpu.memoryUtil * 100, # Percentage
                    "temperature_celsius": gpu.temperature,
                    "driver_version": gpu.driver,
                    "serial": gpu.serial,
                    "display_mode": gpu.display_mode,
                    "display_active": gpu.display_active,
                })
            logger.debug(f"Retrieved status for {len(gpus)} GPUs.")
            return status
        except Exception as e:
            logger.error(f"Error getting GPU status from GPUtil: {e}", exc_info=True)
            return []

    # Renamed from συνοπτική_κατάσταση and correctly indented
    def get_status_summary(self) -> Optional[Dict[str, Any]]:
         if not self._available:
              logger.warning("Attempted to get GPU summary status, but GPUtil is not available.")
              return None

         statuses = self.get_gpu_status()
         if not statuses:
              # Check if the reason for no status was an error during retrieval
              if not self._available: # Double check availability after get_gpu_status call
                   return {"count": 0, "error": "GPUtil initialization failed or not installed."}
              else:
                   # If GPUtil is available but get_gpu_status returned empty, maybe no GPUs found
                   logger.info("GPUtil available but no GPUs detected or status retrieval failed.")
                   # We return summary with count 0 rather than None if GPUtil itself is usable
                   return {"count": 0, "error": "No GPUs detected or failed to retrieve status."}


         summary = {
              "count": len(statuses),
              "total_memory_mb": sum(s['memory_total_mb'] for s in statuses),
              "total_memory_used_mb": sum(s['memory_used_mb'] for s in statuses),
              "avg_load_percent": sum(s['load'] for s in statuses) / len(statuses) if statuses else 0,
              "avg_memory_utilization_percent": sum(s['memory_utilization_percent'] for s in statuses) / len(statuses) if statuses else 0,
              "max_temperature_celsius": max(s['temperature_celsius'] for s in statuses if s.get('temperature_celsius') is not None) if any(s.get('temperature_celsius') is not None for s in statuses) else None,
              "min_temperature_celsius": min(s['temperature_celsius'] for s in statuses if s.get('temperature_celsius') is not None) if any(s.get('temperature_celsius') is not None for s in statuses) else None,
              "gpu_names": sorted(list(set(s['name'] for s in statuses))), # Use sorted list for deterministic output
         }
         logger.debug(f"Generated GPU status summary: {summary}")
         return summary