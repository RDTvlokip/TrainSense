# TrainSense/system_config.py
import psutil      # For CPU, RAM, OS info
import torch       # For PyTorch version, CUDA availability, GPU details via Torch
import platform    # For OS and platform details
import logging
from typing import Dict, Any, List, Optional

# Get a logger instance specific to this module
logger = logging.getLogger(__name__)

# --- Optional Dependency: GPUtil ---
# GPUtil provides more detailed GPU info like driver version, UUID, sometimes richer than torch.cuda
try:
    import GPUtil
except ImportError:
    GPUtil = None # Set to None if not installed
    # Log warning only once during import
    logger.info("GPUtil library not found. GPU information from GPUtil will be unavailable. "
                "Install with 'pip install GPUtil' for more detailed GPU specs. "
                "Basic GPU info via PyTorch will still be attempted.")
# ------------------------------------

class SystemConfig:
    """
    Gathers and caches static system configuration information.

    Collects details about the operating system, Python version, CPU,
    total memory, PyTorch installation, CUDA/cuDNN versions (if available),
    and GPU hardware specifications (using both PyTorch and GPUtil if available).

    The information is gathered once upon initialization and cached.
    Use `refresh()` to force re-gathering of information.
    Use `get_config()` for the full details or `get_summary()` for a concise overview.
    """

    def __init__(self):
        """Initializes SystemConfig and gathers system information immediately."""
        self._config_cache: Optional[Dict[str, Any]] = None # Initialize cache as None
        logger.info("SystemConfig initialized. Gathering static system information...")
        self._gather_config() # Populate the cache


    def _gather_config(self):
        """
        Internal method to gather system configuration details.
        Populates the `_config_cache` dictionary.
        This method performs the actual data collection.
        """
        # If cache already exists, no need to gather again (unless refresh is called)
        if self._config_cache is not None:
            logger.debug("Using cached system configuration.")
            return self._config_cache

        logger.debug("Starting system configuration gathering.")
        config: Dict[str, Any] = {} # Initialize empty dictionary for this run

        # --- OS and Platform Information ---
        try:
             config["os_platform"] = platform.system()      # e.g., 'Linux', 'Windows', 'Darwin'
             config["os_release"] = platform.release()      # e.g., '5.4.0-100-generic', '10', '20.3.0'
             config["os_version"] = platform.version()      # Detailed version string
             config["platform_details"] = platform.platform() # More comprehensive platform string
             config["architecture"] = platform.machine()    # e.g., 'x86_64', 'arm64'
             config["python_version"] = platform.python_version() # e.g., '3.9.7'
             logger.debug("OS/Platform info gathered.")
        except Exception as e:
             logger.error(f"Failed to get OS/Platform info: {e}", exc_info=True)
             config["os_info_error"] = str(e)


        # --- CPU Information ---
        try:
             config["cpu_physical_cores"] = psutil.cpu_count(logical=False) # Number of physical cores
             config["cpu_logical_cores"] = psutil.cpu_count(logical=True)   # Number of logical processors (threads)
             # Get CPU frequency (may require special permissions or not be available on all OS)
             try:
                 freq = psutil.cpu_freq()
                 config["cpu_max_freq_mhz"] = freq.max if freq else None    # Max frequency advertised
                 config["cpu_current_freq_mhz"] = freq.current if freq else None # Current frequency (can fluctuate)
             except Exception as freq_e:
                 logger.warning(f"Could not get CPU frequency details: {freq_e}")
                 config["cpu_max_freq_mhz"] = None
                 config["cpu_current_freq_mhz"] = None
             logger.debug("CPU info gathered.")
        except Exception as e:
             logger.error(f"Failed to get CPU info: {e}", exc_info=True)
             config["cpu_info_error"] = str(e)

        # --- Memory (RAM) Information ---
        try:
             mem = psutil.virtual_memory()
             config["total_memory_bytes"] = mem.total       # Total physical RAM in bytes
             config["total_memory_gb"] = mem.total / (1024 ** 3) # Total physical RAM in GiB
             config["available_memory_bytes"] = mem.available # Available RAM in bytes
             config["available_memory_gb"] = mem.available / (1024 ** 3) # Available RAM in GiB
             logger.debug("Memory (RAM) info gathered.")
        except Exception as e:
             logger.error(f"Failed to get Memory info: {e}", exc_info=True)
             config["memory_info_error"] = str(e)

        # --- PyTorch and CUDA/cuDNN Information ---
        try:
             config["pytorch_version"] = torch.__version__ # PyTorch version string
             config["is_cuda_available"] = torch.cuda.is_available() # Check if CUDA is usable by PyTorch

             if config["is_cuda_available"]:
                 logger.debug("CUDA is available via PyTorch. Gathering CUDA/GPU details...")
                 config["cuda_version"] = torch.version.cuda # CUDA runtime version PyTorch was compiled with
                 try:
                    config["cudnn_version"] = torch.backends.cudnn.version() # cuDNN version
                 except Exception as cudnn_err:
                      logger.warning(f"Could not retrieve cuDNN version: {cudnn_err}")
                      config["cudnn_version"] = "N/A"

                 config["gpu_count_torch"] = torch.cuda.device_count() # Number of GPUs visible to PyTorch
                 devices_torch = []
                 for i in range(config["gpu_count_torch"]):
                      try:
                          props = torch.cuda.get_device_properties(i)
                          devices_torch.append({
                              "id_torch": i, # PyTorch device index
                              "name_torch": props.name, # GPU Name (e.g., "NVIDIA GeForce RTX 3090")
                              "total_memory_mb_torch": props.total_memory / (1024**2), # Total memory in MiB
                              "multi_processor_count": props.multi_processor_count, # SM count
                              "major_minor": f"{props.major}.{props.minor}" # CUDA compute capability
                          })
                      except Exception as prop_err:
                           logger.error(f"Failed to get properties for GPU {i} via torch: {prop_err}", exc_info=True)
                           devices_torch.append({"id_torch": i, "error": str(prop_err)})
                 config["gpu_details_torch"] = devices_torch
                 logger.debug(f"Gathered details for {len(devices_torch)} GPUs via PyTorch.")
             else:
                  # Set defaults if CUDA is not available
                  logger.debug("CUDA not available via PyTorch.")
                  config["cuda_version"] = "N/A"
                  config["cudnn_version"] = "N/A"
                  config["gpu_count_torch"] = 0
                  config["gpu_details_torch"] = []

        except Exception as e:
             logger.error(f"Failed to get PyTorch/CUDA info: {e}", exc_info=True)
             config["pytorch_cuda_info_error"] = str(e)
             # Ensure basic keys exist even on partial failure
             if "is_cuda_available" not in config: config["is_cuda_available"] = False
             if "gpu_count_torch" not in config: config["gpu_count_torch"] = 0


        # --- GPU Information (via GPUtil - supplemental) ---
        # Provides driver version, UUID etc., which PyTorch doesn't easily expose.
        config["gpu_info_gputil"] = [] # Initialize as empty list
        config["gputil_error"] = None  # Initialize error as None
        if GPUtil is not None:
            if config.get("is_cuda_available", False):
                 # Attempt to use GPUtil only if CUDA is available to PyTorch
                 logger.debug("Attempting to gather GPU info via GPUtil...")
                 try:
                     gpus_gputil = GPUtil.getGPUs()
                     gpu_info_list_gputil = []
                     for gpu in gpus_gputil:
                         gpu_info_list_gputil.append({
                             "id": gpu.id, # GPUtil index (often matches PyTorch index, but not guaranteed)
                             "uuid": gpu.uuid, # Unique GPU identifier
                             "name": gpu.name, # GPU Name (should match torch)
                             "memory_total_mb": gpu.memoryTotal, # Total memory in MiB (from nvidia-smi)
                             "driver_version": gpu.driver, # NVIDIA driver version
                         })
                     config["gpu_info_gputil"] = gpu_info_list_gputil
                     logger.debug(f"Gathered details for {len(gpus_gputil)} GPUs via GPUtil.")
                     # Sanity check: compare GPU counts
                     if len(gpus_gputil) != config.get("gpu_count_torch", 0):
                          logger.warning(f"Mismatch in GPU count: PyTorch sees {config.get('gpu_count_torch')}, GPUtil sees {len(gpus_gputil)}. This might indicate visibility issues (e.g., CUDA_VISIBLE_DEVICES).")
                 except Exception as e:
                     # Catch errors specific to GPUtil (e.g., nvidia-smi not found or inaccessible)
                     err_msg = f"Failed to get GPU info via GPUtil: {e}"
                     logger.error(err_msg, exc_info=True)
                     config["gputil_error"] = str(e)
            else:
                 # CUDA not available, so GPUtil info isn't relevant in this context
                 msg = "CUDA not available, skipping GPUtil info gathering."
                 logger.debug(msg)
                 config["gputil_error"] = msg
        else:
             # GPUtil library was not imported successfully
             msg = "GPUtil library not installed, cannot gather supplemental GPU info."
             logger.debug(msg)
             config["gputil_error"] = msg


        logger.info("System configuration gathering complete.")
        self._config_cache = config # Store the gathered info in the cache
        return self._config_cache

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the cached system configuration dictionary.
        If the cache is empty, it triggers `_gather_config`.

        Returns:
            Dict[str, Any]: The complete system configuration dictionary.
        """
        # Return cache if it exists, otherwise gather and return
        return self._config_cache if self._config_cache is not None else self._gather_config()

    def get_summary(self) -> Dict[str, Any]:
         """
         Provides a concise summary of the most important system configuration details.

         Returns:
             Dict[str, Any]: A dictionary containing key system specs:
                             'os', 'python_version', 'cpu_cores', 'total_memory_gb',
                             'pytorch_version', 'cuda_available', 'cuda_version',
                             'cudnn_version', 'gpu_count', 'gpu_info' (list of dicts).
         """
         config = self.get_config() # Ensure config is gathered/retrieved
         summary = {
             "os": f"{config.get('os_platform', 'N/A')} {config.get('os_release', 'N/A')}",
             "python_version": config.get('python_version', 'N/A'),
             "cpu_cores": config.get('cpu_logical_cores', 'N/A'), # Report logical cores/threads
             "total_memory_gb": round(config.get('total_memory_gb', 0), 2), # Total RAM in GB
             "pytorch_version": config.get('pytorch_version', 'N/A'),
             "cuda_available": config.get('is_cuda_available', False),
             "cuda_version": config.get('cuda_version', 'N/A'), # CUDA version used by PyTorch
             "cudnn_version": config.get('cudnn_version', 'N/A'),
             "gpu_count": config.get('gpu_count_torch', 0), # Number of GPUs visible to PyTorch
             # Provide GPU details: Prefer GPUtil if available, fallback to PyTorch info
             "gpu_info": config.get("gpu_info_gputil") if config.get("gpu_info_gputil") else \
                         [{"id": d.get("id_torch"), "name": d.get("name_torch", "N/A"), "memory_total_mb": d.get("total_memory_mb_torch", "N/A")} for d in config.get("gpu_details_torch", [])]
         }
         # Add error fields to summary if they occurred during gathering
         for key, value in config.items():
             if key.endswith("_error") and value:
                 summary[key] = value

         logger.debug("Generated system configuration summary.")
         return summary

    def refresh(self):
        """
        Clears the configuration cache and forces a re-gathering of system information
        the next time `get_config()` or `get_summary()` is called.
        """
        logger.info("Refreshing system configuration cache.")
        self._config_cache = None # Clear the cache
        # Optionally, immediately re-gather: self._gather_config()
        # Current implementation re-gathers lazily on next get() call.