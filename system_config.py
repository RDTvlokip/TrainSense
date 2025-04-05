# TrainSense/system_config.py
import psutil
import torch
import platform
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

try:
    import GPUtil
except ImportError:
    GPUtil = None
    logger.warning("GPUtil library not found. GPU information will be limited. Install with 'pip install GPUtil'")


class SystemConfig:
    def __init__(self):
        self._config_cache: Optional[Dict[str, Any]] = None
        logger.info("SystemConfig initialized. Gathering system information...")
        self._gather_config()


    def _gather_config(self):
        if self._config_cache is not None:
            return self._config_cache

        logger.debug("Starting system configuration gathering.")
        config: Dict[str, Any] = {}

        # OS Info
        try:
             config["os_platform"] = platform.system()
             config["os_release"] = platform.release()
             config["os_version"] = platform.version()
             config["platform_details"] = platform.platform()
             config["architecture"] = platform.machine()
             config["python_version"] = platform.python_version()
        except Exception as e:
             logger.error(f"Failed to get OS/Platform info: {e}", exc_info=True)
             config["os_info_error"] = str(e)


        # CPU Info
        try:
             config["cpu_physical_cores"] = psutil.cpu_count(logical=False)
             config["cpu_logical_cores"] = psutil.cpu_count(logical=True)
             freq = psutil.cpu_freq()
             config["cpu_max_freq_mhz"] = freq.max if freq else None
             config["cpu_current_freq_mhz"] = freq.current if freq else None
        except Exception as e:
             logger.error(f"Failed to get CPU info: {e}", exc_info=True)
             config["cpu_info_error"] = str(e)

        # Memory Info
        try:
             mem = psutil.virtual_memory()
             config["total_memory_bytes"] = mem.total
             config["total_memory_gb"] = mem.total / (1024 ** 3)
             config["available_memory_bytes"] = mem.available
             config["available_memory_gb"] = mem.available / (1024 ** 3)
        except Exception as e:
             logger.error(f"Failed to get Memory info: {e}", exc_info=True)
             config["memory_info_error"] = str(e)

        # PyTorch and CUDA/cuDNN Info
        try:
             config["pytorch_version"] = torch.__version__
             config["is_cuda_available"] = torch.cuda.is_available()
             if config["is_cuda_available"]:
                 config["cuda_version"] = torch.version.cuda
                 config["cudnn_version"] = torch.backends.cudnn.version()
                 config["gpu_count_torch"] = torch.cuda.device_count()
                 devices = []
                 for i in range(config["gpu_count_torch"]):
                      props = torch.cuda.get_device_properties(i)
                      devices.append({
                          "id_torch": i,
                          "name_torch": props.name,
                          "total_memory_mb_torch": props.total_memory / (1024**2),
                          "multi_processor_count": props.multi_processor_count,
                          "major_minor": f"{props.major}.{props.minor}"
                      })
                 config["gpu_details_torch"] = devices
             else:
                  config["cuda_version"] = "N/A"
                  config["cudnn_version"] = "N/A"
                  config["gpu_count_torch"] = 0
                  config["gpu_details_torch"] = []
        except Exception as e:
             logger.error(f"Failed to get PyTorch/CUDA info: {e}", exc_info=True)
             config["pytorch_cuda_info_error"] = str(e)
             # Ensure defaults if partial failure
             if "is_cuda_available" not in config: config["is_cuda_available"] = False
             if "gpu_count_torch" not in config: config["gpu_count_torch"] = 0


        # GPU Info (GPUtil) - supplements torch info
        config["gpu_info_gputil"] = []
        if GPUtil is not None and config.get("is_cuda_available", False):
            try:
                gpus = GPUtil.getGPUs()
                gpu_info_list = []
                for gpu in gpus:
                    gpu_info_list.append({
                        "id": gpu.id,
                        "uuid": gpu.uuid,
                        "name": gpu.name,
                        "memory_total_mb": gpu.memoryTotal,
                        "driver_version": gpu.driver,
                    })
                config["gpu_info_gputil"] = gpu_info_list
                if len(gpus) != config.get("gpu_count_torch", 0):
                     logger.warning(f"Mismatch between torch.cuda.device_count() [{config.get('gpu_count_torch')}] and GPUtil.getGPUs() [{len(gpus)}]")
            except Exception as e:
                logger.error(f"Failed to get GPU info via GPUtil: {e}", exc_info=True)
                config["gputil_error"] = str(e)
        elif GPUtil is None:
             config["gputil_error"] = "GPUtil library not installed."
        else: # CUDA not available
             config["gputil_error"] = "CUDA not available, GPUtil info not applicable."


        logger.debug("System configuration gathering complete.")
        self._config_cache = config
        return self._config_cache

    def get_config(self) -> Dict[str, Any]:
        return self._config_cache or self._gather_config()

    def get_summary(self) -> Dict[str, Any]:
         # Provide a cleaner subset for quick overview
         config = self.get_config()
         summary = {
             "os": f"{config.get('os_platform', 'N/A')} {config.get('os_release', 'N/A')}",
             "python_version": config.get('python_version', 'N/A'),
             "cpu_cores": config.get('cpu_logical_cores', 'N/A'),
             "total_memory_gb": round(config.get('total_memory_gb', 0), 2),
             "pytorch_version": config.get('pytorch_version', 'N/A'),
             "cuda_available": config.get('is_cuda_available', False),
             "cuda_version": config.get('cuda_version', 'N/A'),
             "cudnn_version": config.get('cudnn_version', 'N/A'),
             "gpu_count": config.get('gpu_count_torch', 0),
             "gpu_info": config.get("gpu_info_gputil") # Use GPUtil summary as it's often cleaner
                         if config.get("gpu_info_gputil")
                         else [{"name": d.get("name_torch", "N/A"), "memory_total_mb": d.get("total_memory_mb_torch", "N/A")} for d in config.get("gpu_details_torch", [])] # Fallback to torch details
         }
         return summary

    def refresh(self):
        logger.info("Refreshing system configuration.")
        self._config_cache = None
        self._gather_config()