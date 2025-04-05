# TrainSense/analyzer.py
import logging
from typing import Dict, Any, List, Optional, Union

# Make sure SystemConfig is imported if type hinting is used extensively or for clarity
from .system_config import SystemConfig
from .utils import validate_positive_integer, validate_positive_float

logger = logging.getLogger(__name__)

class TrainingAnalyzer:
    DEFAULT_LOW_MEM_GPU_THRESHOLD_MB = 6 * 1024
    DEFAULT_HIGH_MEM_GPU_THRESHOLD_MB = 12 * 1024
    DEFAULT_LOW_MEM_BATCH_SIZE_LIMIT = 16
    DEFAULT_HIGH_MEM_BATCH_SIZE_LIMIT = 128
    DEFAULT_LARGE_MODEL_THRESHOLD_PARAMS = 100_000_000
    DEFAULT_SMALL_MODEL_THRESHOLD_PARAMS = 1_000_000
    DEFAULT_MAX_LR = 0.1
    DEFAULT_MIN_LR = 1e-5
    DEFAULT_MIN_EPOCHS = 10
    DEFAULT_MAX_EPOCHS = 300
    DEFAULT_SUGGESTED_MIN_EPOCHS = 50
    DEFAULT_SUGGESTED_MAX_EPOCHS = 150

    def __init__(self,
                 batch_size: int,
                 learning_rate: float,
                 epochs: int,
                 system_config: Optional[SystemConfig] = None, # Keep accepting the object
                 arch_info: Optional[Dict[str, Any]] = None):

        validate_positive_integer(batch_size, "Batch size")
        validate_positive_float(learning_rate, "Learning rate")
        validate_positive_integer(epochs, "Epochs")

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        # Store the SystemConfig object as before
        self.system_config = system_config
        self.arch_info = arch_info if arch_info else {}

        logger.info(f"TrainingAnalyzer initialized with batch_size={batch_size}, lr={learning_rate}, epochs={epochs}")

    def _get_avg_gpu_memory_mb(self) -> Optional[float]:
        """Calculates average GPU memory using the SystemConfig object's summary."""
        if self.system_config:
            # Access GPU info via the get_summary() method
            system_summary = self.system_config.get_summary()
            gpu_info_list = system_summary.get('gpu_info', []) # Get the list from summary

            if gpu_info_list and isinstance(gpu_info_list, list):
                # Extract memory, handling potential None or missing keys robustly
                # The summary provides 'memory_total_mb' key
                valid_mems = [
                    mem for gpu in gpu_info_list
                    if isinstance(mem := gpu.get("memory_total_mb"), (int, float))
                ]

                if valid_mems:
                    total_memory = sum(valid_mems)
                    return total_memory / len(valid_mems)
                else:
                    logger.warning("GPU info list found in summary, but no valid 'memory_total_mb' values.")
                    return None # No valid memory info found
            else:
                # logger.debug("No GPU info found in system config summary.") # Debug level might be better
                return None # No GPUs listed in summary
        # logger.debug("No system_config object provided to TrainingAnalyzer.") # Debug level
        return None # No system config provided

    def check_hyperparameters(self) -> List[str]:
        recommendations = []
        # This call now uses the corrected method
        avg_gpu_mem_mb = self._get_avg_gpu_memory_mb()
        total_params = self.arch_info.get("total_parameters", 0)

        # --- The rest of the method remains the same ---
        if avg_gpu_mem_mb is not None:
            logger.info(f"Average GPU memory detected: {avg_gpu_mem_mb:.0f} MB")
            if avg_gpu_mem_mb < self.DEFAULT_LOW_MEM_GPU_THRESHOLD_MB:
                if self.batch_size > self.DEFAULT_LOW_MEM_BATCH_SIZE_LIMIT:
                    recommendations.append(f"Low GPU memory ({avg_gpu_mem_mb:.0f} MB avg) detected. Consider batch size <= {self.DEFAULT_LOW_MEM_BATCH_SIZE_LIMIT} (current: {self.batch_size}).")
                else:
                     recommendations.append(f"Batch size ({self.batch_size}) seems appropriate for low GPU memory ({avg_gpu_mem_mb:.0f} MB avg).")
            elif avg_gpu_mem_mb >= self.DEFAULT_HIGH_MEM_GPU_THRESHOLD_MB:
                 if self.batch_size < self.DEFAULT_LOW_MEM_BATCH_SIZE_LIMIT * 2 and total_params > self.DEFAULT_SMALL_MODEL_THRESHOLD_PARAMS:
                     recommendations.append(f"High GPU memory ({avg_gpu_mem_mb:.0f} MB avg) available. Consider increasing batch size for better utilization (current: {self.batch_size}).")
                 elif self.batch_size > self.DEFAULT_HIGH_MEM_BATCH_SIZE_LIMIT:
                     recommendations.append(f"Batch size ({self.batch_size}) might be excessive even for high memory GPUs ({avg_gpu_mem_mb:.0f} MB avg). Recommended <= {self.DEFAULT_HIGH_MEM_BATCH_SIZE_LIMIT}.")
                 else:
                     recommendations.append(f"Batch size ({self.batch_size}) appears suitable for high GPU memory ({avg_gpu_mem_mb:.0f} MB avg).")
            else: # Moderate memory
                recommendations.append(f"Batch size ({self.batch_size}) seems reasonable for available GPU memory ({avg_gpu_mem_mb:.0f} MB avg).")

            # Check model size vs batch size on GPU
            if total_params > self.DEFAULT_LARGE_MODEL_THRESHOLD_PARAMS and self.batch_size > self.DEFAULT_LOW_MEM_BATCH_SIZE_LIMIT * 2:
                 recommendations.append(f"Large model ({total_params:,} params) detected. Current batch size ({self.batch_size}) might lead to memory issues on {avg_gpu_mem_mb:.0f} MB GPU. Monitor usage or reduce size.")
            elif total_params < self.DEFAULT_SMALL_MODEL_THRESHOLD_PARAMS and self.batch_size < self.DEFAULT_LOW_MEM_BATCH_SIZE_LIMIT:
                 recommendations.append(f"Small model ({total_params:,} params) detected. Consider increasing batch size (current: {self.batch_size}) for potentially faster training on GPU.")

        else: # No GPU info available
            recommendations.append("No valid GPU memory info available. Batch size recommendations are limited. Assuming CPU or unknown device.")
            if total_params > self.DEFAULT_LARGE_MODEL_THRESHOLD_PARAMS and self.batch_size > 32:
                 recommendations.append(f"Large model ({total_params:,} params) likely on CPU/unknown device; batch size {self.batch_size} might be slow. Consider reducing.")

        # Learning Rate Checks
        if self.learning_rate > self.DEFAULT_MAX_LR:
            recommendations.append(f"Learning rate ({self.learning_rate}) is potentially too high (> {self.DEFAULT_MAX_LR}). Risk of unstable training or divergence.")
        elif self.learning_rate < self.DEFAULT_MIN_LR:
            recommendations.append(f"Learning rate ({self.learning_rate}) is very low (< {self.DEFAULT_MIN_LR}). Training might be extremely slow.")
        else:
            recommendations.append(f"Learning rate ({self.learning_rate}) is within a typical range [{self.DEFAULT_MIN_LR}, {self.DEFAULT_MAX_LR}]. Fine-tune based on loss behavior.")

        # Epoch Checks
        if self.epochs < self.DEFAULT_MIN_EPOCHS:
            recommendations.append(f"Number of epochs ({self.epochs}) is low (< {self.DEFAULT_MIN_EPOCHS}). May lead to underfitting.")
        elif self.epochs > self.DEFAULT_MAX_EPOCHS:
            recommendations.append(f"Number of epochs ({self.epochs}) is high (> {self.DEFAULT_MAX_EPOCHS}). Increased risk of overfitting and long training time.")
        else:
            recommendations.append(f"Number of epochs ({self.epochs}) is within a reasonable range [{self.DEFAULT_MIN_EPOCHS}, {self.DEFAULT_MAX_EPOCHS}]. Monitor validation metrics.")

        if not self.arch_info:
             recommendations.append("No model architecture information provided. Recommendations are based only on hyperparameters and system config.")

        return recommendations

    def auto_adjust(self) -> Dict[str, Union[int, float]]:
        adjusted_params = {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs
        }
        # Use the corrected method here as well
        avg_gpu_mem_mb = self._get_avg_gpu_memory_mb()
        total_params = self.arch_info.get("total_parameters", 0)

        # --- Batch Size Adjustment ---
        if avg_gpu_mem_mb is not None: # GPU present
            original_bs = adjusted_params["batch_size"]
            if avg_gpu_mem_mb < self.DEFAULT_LOW_MEM_GPU_THRESHOLD_MB:
                adjusted_params["batch_size"] = min(original_bs, self.DEFAULT_LOW_MEM_BATCH_SIZE_LIMIT)
            elif avg_gpu_mem_mb >= self.DEFAULT_HIGH_MEM_GPU_THRESHOLD_MB:
                 adjusted_params["batch_size"] = min(original_bs, self.DEFAULT_HIGH_MEM_BATCH_SIZE_LIMIT)
                 # Suggest increase for small models only if current bs is low
                 if total_params < self.DEFAULT_SMALL_MODEL_THRESHOLD_PARAMS and original_bs < self.DEFAULT_LOW_MEM_BATCH_SIZE_LIMIT * 2:
                      adjusted_params["batch_size"] = max(original_bs, self.DEFAULT_LOW_MEM_BATCH_SIZE_LIMIT * 2)
            # Apply cap based on model size *after* initial GPU memory adjustment
            if total_params > self.DEFAULT_LARGE_MODEL_THRESHOLD_PARAMS:
                 adjusted_params["batch_size"] = min(adjusted_params["batch_size"], self.DEFAULT_LOW_MEM_BATCH_SIZE_LIMIT * 4)

        else: # CPU or unknown
             if total_params > self.DEFAULT_LARGE_MODEL_THRESHOLD_PARAMS:
                 adjusted_params["batch_size"] = min(self.batch_size, 32) # Cap large model batch size on CPU

        # --- Learning Rate Adjustment ---
        if self.learning_rate > self.DEFAULT_MAX_LR:
            adjusted_params["learning_rate"] = self.DEFAULT_MAX_LR / 2
        elif self.learning_rate < self.DEFAULT_MIN_LR:
            adjusted_params["learning_rate"] = self.DEFAULT_MIN_LR * 10 # Increase very low LR

        # --- Epoch Adjustment ---
        if self.epochs < self.DEFAULT_MIN_EPOCHS:
            adjusted_params["epochs"] = self.DEFAULT_SUGGESTED_MIN_EPOCHS
        elif self.epochs > self.DEFAULT_MAX_EPOCHS:
            adjusted_params["epochs"] = self.DEFAULT_SUGGESTED_MAX_EPOCHS

        if adjusted_params != {"batch_size": self.batch_size, "learning_rate": self.learning_rate, "epochs": self.epochs}:
             logger.info(f"Original params: batch={self.batch_size}, lr={self.learning_rate}, epochs={self.epochs}. Suggested adjustments: {adjusted_params}")
        else:
             logger.info("No automatic adjustments suggested based on current heuristics.")
        return adjusted_params

    def summary(self) -> Dict[str, Any]:
        s = {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs
        }
        if self.system_config:
            # Use the summary method here too for consistency
            s["system_summary"] = self.system_config.get_summary()
        if self.arch_info:
            # Ensure arch_info is included if available
            s["architecture_summary"] = self.arch_info # Or specific keys if needed
        return s