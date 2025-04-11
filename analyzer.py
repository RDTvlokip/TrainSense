# TrainSense/analyzer.py
import logging
from typing import Dict, Any, List, Optional, Union

# Import SystemConfig for type hinting and access to system information
from .system_config import SystemConfig
# Import validation utilities
from .utils import validate_positive_integer, validate_positive_float

# Get a logger instance specific to this module
logger = logging.getLogger(__name__)

class TrainingAnalyzer:
    """
    Analyzes training hyperparameters (batch size, learning rate, epochs)
    in the context of system resources (especially GPU memory) and model architecture.
    Provides recommendations and suggests potential adjustments.
    """
    # --- Default Thresholds ---
    # These constants define thresholds used for making recommendations.
    # GPU Memory Thresholds (MB)
    DEFAULT_LOW_MEM_GPU_THRESHOLD_MB = 6 * 1024  # GPUs below this are considered low memory
    DEFAULT_HIGH_MEM_GPU_THRESHOLD_MB = 12 * 1024 # GPUs above this are considered high memory
    # Batch Size Limits based on Memory
    DEFAULT_LOW_MEM_BATCH_SIZE_LIMIT = 16       # Suggested max batch size for low memory GPUs
    DEFAULT_HIGH_MEM_BATCH_SIZE_LIMIT = 128      # Suggested max batch size for high memory GPUs
    # Model Parameter Count Thresholds
    DEFAULT_LARGE_MODEL_THRESHOLD_PARAMS = 100_000_000 # Models above this are considered large
    DEFAULT_SMALL_MODEL_THRESHOLD_PARAMS = 1_000_000   # Models below this are considered small
    # Learning Rate Range
    DEFAULT_MAX_LR = 0.1                           # Warning if LR is higher than this
    DEFAULT_MIN_LR = 1e-5                          # Warning if LR is lower than this
    # Epoch Range
    DEFAULT_MIN_EPOCHS = 10                        # Warning if epochs are fewer than this
    DEFAULT_MAX_EPOCHS = 300                       # Warning if epochs are more than this
    # Suggested Epochs (for auto_adjust)
    DEFAULT_SUGGESTED_MIN_EPOCHS = 50
    DEFAULT_SUGGESTED_MAX_EPOCHS = 150
    # --------------------------

    def __init__(self,
                 batch_size: int,
                 learning_rate: float,
                 epochs: int,
                 system_config: Optional[SystemConfig] = None, # Accept a SystemConfig object
                 arch_info: Optional[Dict[str, Any]] = None):  # Accept model architecture info
        """
        Initializes the TrainingAnalyzer.

        Args:
            batch_size (int): The current batch size used for training.
            learning_rate (float): The current learning rate used.
            epochs (int): The current number of epochs planned for training.
            system_config (Optional[SystemConfig]): An instance of SystemConfig containing
                                                    system hardware details. Defaults to None.
            arch_info (Optional[Dict[str, Any]]): A dictionary containing model architecture
                                                  details (e.g., from ArchitectureAnalyzer). Defaults to None.
        """
        # Validate inputs
        validate_positive_integer(batch_size, "Batch size")
        validate_positive_float(learning_rate, "Learning rate")
        validate_positive_integer(epochs, "Epochs")

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        # Store the provided SystemConfig object
        self.system_config = system_config
        # Store the provided architecture info, default to empty dict if None
        self.arch_info = arch_info if arch_info else {}

        logger.info(f"TrainingAnalyzer initialized with batch_size={batch_size}, lr={learning_rate}, epochs={epochs}")
        if system_config:
            logger.debug("SystemConfig provided.")
        if arch_info:
            logger.debug("Architecture info provided.")


    def _get_avg_gpu_memory_mb(self) -> Optional[float]:
        """
        Calculates the average total GPU memory across all detected GPUs.
        Uses the summary provided by the SystemConfig object.

        Returns:
            Optional[float]: The average GPU memory in MB, or None if no GPUs are found
                             or SystemConfig was not provided or GPU info is unavailable/invalid.
        """
        # Check if a SystemConfig object was provided during initialization
        if self.system_config:
            # Get the summary dictionary from the SystemConfig object
            system_summary = self.system_config.get_summary()
            # Safely get the 'gpu_info' list from the summary (defaults to empty list)
            gpu_info_list = system_summary.get('gpu_info', [])

            # Check if gpu_info_list is actually a list and is not empty
            if gpu_info_list and isinstance(gpu_info_list, list):
                # Extract total memory values, ensuring they are numeric and handling missing keys robustly
                # The summary provides 'memory_total_mb' key from GPUtil or fallback.
                valid_mems = [
                    mem for gpu in gpu_info_list
                    # Use assignment expression (walrus operator) for brevity
                    if isinstance(mem := gpu.get("memory_total_mb"), (int, float)) and mem > 0 # Ensure it's a positive number
                ]

                # If we found valid memory entries
                if valid_mems:
                    total_memory = sum(valid_mems)
                    average_memory = total_memory / len(valid_mems)
                    logger.debug(f"Calculated average GPU memory: {average_memory:.0f} MB from {len(valid_mems)} GPUs.")
                    return average_memory
                else:
                    # Log a warning if the list was present but contained no valid memory info
                    logger.warning("GPU info list found in summary, but no valid 'memory_total_mb' values > 0.")
                    return None # No valid memory info found
            else:
                # Log if the summary had no GPU info (e.g., no GPUs detected by GPUtil/PyTorch)
                # logger.debug("No GPU info found in system config summary.") # Debug level might be better
                return None # No GPUs listed in summary
        # logger.debug("No system_config object provided to TrainingAnalyzer.") # Debug level
        return None # No system config provided


    def check_hyperparameters(self) -> List[str]:
        """
        Checks the current hyperparameters against system resources and model architecture.

        Returns:
            List[str]: A list of recommendations or observations.
        """
        recommendations = []
        # Get average GPU memory using the internal helper method
        avg_gpu_mem_mb = self._get_avg_gpu_memory_mb()
        # Get total parameters from architecture info, default to 0 if not available
        total_params = self.arch_info.get("total_parameters", 0)
        # Get primary architecture type
        primary_arch = self.arch_info.get("primary_architecture_type", "Unknown")

        logger.info("Checking hyperparameters against system and model context.")

        # --- Batch Size Recommendations based on GPU Memory ---
        if avg_gpu_mem_mb is not None:
            logger.info(f"Average GPU memory detected: {avg_gpu_mem_mb:.0f} MB")
            # Low Memory GPU Case
            if avg_gpu_mem_mb < self.DEFAULT_LOW_MEM_GPU_THRESHOLD_MB:
                if self.batch_size > self.DEFAULT_LOW_MEM_BATCH_SIZE_LIMIT:
                    recommendations.append(f"[Warning] Low GPU memory ({avg_gpu_mem_mb:.0f} MB avg) detected. Batch size {self.batch_size} might cause Out-of-Memory errors. Consider <= {self.DEFAULT_LOW_MEM_BATCH_SIZE_LIMIT}.")
                else:
                     recommendations.append(f"[Info] Batch size ({self.batch_size}) seems appropriate for low GPU memory ({avg_gpu_mem_mb:.0f} MB avg).")
            # High Memory GPU Case
            elif avg_gpu_mem_mb >= self.DEFAULT_HIGH_MEM_GPU_THRESHOLD_MB:
                 # Suggest increasing BS only if current BS is low AND model isn't tiny (already fast)
                 if self.batch_size < self.DEFAULT_LOW_MEM_BATCH_SIZE_LIMIT * 2 and total_params > self.DEFAULT_SMALL_MODEL_THRESHOLD_PARAMS:
                     recommendations.append(f"[Suggestion] High GPU memory ({avg_gpu_mem_mb:.0f} MB avg) available. Consider increasing batch size (current: {self.batch_size}) for potentially better utilization and faster training.")
                 elif self.batch_size > self.DEFAULT_HIGH_MEM_BATCH_SIZE_LIMIT:
                     recommendations.append(f"[Warning] Batch size ({self.batch_size}) might be excessive even for high memory GPUs ({avg_gpu_mem_mb:.0f} MB avg). Recommended <= {self.DEFAULT_HIGH_MEM_BATCH_SIZE_LIMIT}.")
                 else:
                     recommendations.append(f"[Info] Batch size ({self.batch_size}) appears suitable for high GPU memory ({avg_gpu_mem_mb:.0f} MB avg).")
            # Moderate Memory GPU Case
            else:
                recommendations.append(f"[Info] Batch size ({self.batch_size}) seems reasonable for available moderate GPU memory ({avg_gpu_mem_mb:.0f} MB avg).")

            # --- Check Model Size vs Batch Size on GPU ---
            if total_params > self.DEFAULT_LARGE_MODEL_THRESHOLD_PARAMS and self.batch_size > self.DEFAULT_LOW_MEM_BATCH_SIZE_LIMIT * 2:
                 recommendations.append(f"[Warning] Large model ({total_params:,} params) detected. Current batch size ({self.batch_size}) might strain GPU memory ({avg_gpu_mem_mb:.0f} MB avg). Monitor usage closely or consider reducing batch size/using gradient accumulation.")
            elif total_params < self.DEFAULT_SMALL_MODEL_THRESHOLD_PARAMS and self.batch_size < self.DEFAULT_LOW_MEM_BATCH_SIZE_LIMIT:
                 recommendations.append(f"[Suggestion] Small model ({total_params:,} params) detected. Consider increasing batch size (current: {self.batch_size}) for potentially faster training on your GPU ({avg_gpu_mem_mb:.0f} MB avg).")

        else: # No GPU info available (assume CPU or unknown)
            recommendations.append("[Info] No valid GPU memory info available. Batch size recommendations are limited. Assuming CPU or unknown device.")
            # Check for large model on potentially slow CPU setup
            if total_params > self.DEFAULT_LARGE_MODEL_THRESHOLD_PARAMS and self.batch_size > 32:
                 recommendations.append(f"[Warning] Large model ({total_params:,} params) likely on CPU/unknown device; batch size {self.batch_size} might lead to very slow training. Consider reducing.")

        # --- Learning Rate Checks ---
        if self.learning_rate > self.DEFAULT_MAX_LR:
            recommendations.append(f"[Warning] Learning rate ({self.learning_rate}) is potentially too high (> {self.DEFAULT_MAX_LR}). Risk of unstable training or divergence. Consider lowering.")
        elif self.learning_rate < self.DEFAULT_MIN_LR:
            recommendations.append(f"[Warning] Learning rate ({self.learning_rate}) is very low (< {self.DEFAULT_MIN_LR}). Training might be extremely slow. Ensure this is intended.")
        else:
            recommendations.append(f"[Info] Learning rate ({self.learning_rate}) is within a typical range [{self.DEFAULT_MIN_LR}, {self.DEFAULT_MAX_LR}]. Fine-tune based on observed loss behavior.")

        # --- Epoch Checks ---
        if self.epochs < self.DEFAULT_MIN_EPOCHS:
            recommendations.append(f"[Warning] Number of epochs ({self.epochs}) is low (< {self.DEFAULT_MIN_EPOCHS}). Model may underfit. Ensure sufficient training time.")
        elif self.epochs > self.DEFAULT_MAX_EPOCHS:
            recommendations.append(f"[Warning] Number of epochs ({self.epochs}) is high (> {self.DEFAULT_MAX_EPOCHS}). Increased risk of overfitting and long training duration. Monitor validation metrics closely.")
        else:
            recommendations.append(f"[Info] Number of epochs ({self.epochs}) is within a reasonable range [{self.DEFAULT_MIN_EPOCHS}, {self.DEFAULT_MAX_EPOCHS}]. Monitor validation metrics to determine optimal stopping point.")

        # --- Architecture-Specific Recommendations ---
        if primary_arch == "Transformer":
             recommendations.append("[Recommendation] Transformer architecture detected. Often benefits from AdamW optimizer and learning rate scheduling (e.g., linear warmup with decay). Initial LR might need careful tuning (e.g., 1e-5 to 5e-4 range).")
        elif primary_arch == "RNN":
             recommendations.append("[Recommendation] RNN/LSTM/GRU architecture detected. Prone to vanishing/exploding gradients; consider using gradient clipping. May benefit from lower learning rates (e.g., 1e-4 to 1e-3) or RMSprop/Adam optimizers.")
        elif primary_arch == "CNN":
             recommendations.append("[Recommendation] CNN architecture detected. Generally robust. Adam or SGD with momentum are common choices. Ensure appropriate normalization (e.g., BatchNorm) is used.")


        # Final check if architecture info was missing
        if not self.arch_info:
             recommendations.append("[Info] No model architecture information provided. Recommendations are based only on hyperparameters and system config (if available).")

        return recommendations


    def auto_adjust(self) -> Dict[str, Union[int, float]]:
        """
        Suggests adjusted hyperparameters based on simple heuristics.
        NOTE: This is a basic suggestion, not a guaranteed optimal set.

        Returns:
            Dict[str, Union[int, float]]: A dictionary containing potentially adjusted
                                          'batch_size', 'learning_rate', and 'epochs'.
        """
        logger.info("Attempting heuristic auto-adjustment of hyperparameters...")
        # Start with current parameters
        adjusted_params = {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs
        }
        # Get system/model info again
        avg_gpu_mem_mb = self._get_avg_gpu_memory_mb()
        total_params = self.arch_info.get("total_parameters", 0)

        original_bs = self.batch_size # Store original for comparison

        # --- Batch Size Adjustment ---
        if avg_gpu_mem_mb is not None: # GPU present
            logger.debug(f"Adjusting batch size based on GPU memory {avg_gpu_mem_mb:.0f}MB and model size {total_params:,}")
            # Apply memory constraints
            if avg_gpu_mem_mb < self.DEFAULT_LOW_MEM_GPU_THRESHOLD_MB:
                adjusted_params["batch_size"] = min(original_bs, self.DEFAULT_LOW_MEM_BATCH_SIZE_LIMIT)
            elif avg_gpu_mem_mb >= self.DEFAULT_HIGH_MEM_GPU_THRESHOLD_MB:
                 # Allow increase only if currently low AND model isn't tiny
                 if original_bs < self.DEFAULT_LOW_MEM_BATCH_SIZE_LIMIT * 2 and total_params > self.DEFAULT_SMALL_MODEL_THRESHOLD_PARAMS:
                      adjusted_params["batch_size"] = max(original_bs, self.DEFAULT_LOW_MEM_BATCH_SIZE_LIMIT * 2) # Suggest doubling
                 # Apply upper cap for high memory
                 adjusted_params["batch_size"] = min(adjusted_params["batch_size"], self.DEFAULT_HIGH_MEM_BATCH_SIZE_LIMIT)
            # Moderate memory - keep original unless adjusted by model size below

            # Apply model size constraint *after* memory adjustment
            if total_params > self.DEFAULT_LARGE_MODEL_THRESHOLD_PARAMS:
                 # Reduce BS for large models, ensure a minimum (e.g., 4)
                 adjusted_params["batch_size"] = max(4, min(adjusted_params["batch_size"], self.DEFAULT_LOW_MEM_BATCH_SIZE_LIMIT * 4))
            elif total_params < self.DEFAULT_SMALL_MODEL_THRESHOLD_PARAMS:
                 # Allow potential increase for small models, cap at a reasonable limit (e.g., 256)
                 adjusted_params["batch_size"] = min(256, max(adjusted_params["batch_size"], self.DEFAULT_LOW_MEM_BATCH_SIZE_LIMIT))


        else: # CPU or unknown
             logger.debug(f"Adjusting batch size based on CPU assumption and model size {total_params:,}")
             # Reduce BS significantly for large models on CPU
             if total_params > self.DEFAULT_LARGE_MODEL_THRESHOLD_PARAMS:
                 adjusted_params["batch_size"] = min(original_bs, 32) # Cap large model batch size on CPU


        # --- Learning Rate Adjustment ---
        if self.learning_rate > self.DEFAULT_MAX_LR:
            adjusted_params["learning_rate"] = self.DEFAULT_MAX_LR / 2 # Halve very high LR
            logger.debug(f"Adjusting high learning rate {self.learning_rate} -> {adjusted_params['learning_rate']:.2e}")
        elif self.learning_rate < self.DEFAULT_MIN_LR:
            adjusted_params["learning_rate"] = self.DEFAULT_MIN_LR * 10 # Increase very low LR
            logger.debug(f"Adjusting low learning rate {self.learning_rate} -> {adjusted_params['learning_rate']:.2e}")


        # --- Epoch Adjustment ---
        if self.epochs < self.DEFAULT_MIN_EPOCHS:
            adjusted_params["epochs"] = self.DEFAULT_SUGGESTED_MIN_EPOCHS
            logger.debug(f"Adjusting low epochs {self.epochs} -> {adjusted_params['epochs']}")
        elif self.epochs > self.DEFAULT_MAX_EPOCHS:
            adjusted_params["epochs"] = self.DEFAULT_SUGGESTED_MAX_EPOCHS
            logger.debug(f"Adjusting high epochs {self.epochs} -> {adjusted_params['epochs']}")


        # Log if adjustments were made
        original_params = {"batch_size": self.batch_size, "learning_rate": self.learning_rate, "epochs": self.epochs}
        if adjusted_params != original_params:
             logger.info(f"Original params: {original_params}. Suggested adjustments: {adjusted_params}")
        else:
             logger.info("No automatic adjustments suggested based on current heuristics.")

        return adjusted_params


    def summary(self) -> Dict[str, Any]:
        """
        Provides a summary of the current settings and system/architecture context.

        Returns:
            Dict[str, Any]: A dictionary containing the current hyperparameters and
                            summaries of the system and model architecture, if available.
        """
        s = {
            "current_hyperparameters": {
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs
            },
            "system_summary": None,
            "architecture_summary": None
        }
        # Include system summary if available
        if self.system_config:
            s["system_summary"] = self.system_config.get_summary()
        # Include architecture summary if available
        if self.arch_info:
            # Select key info or include the whole dict
            s["architecture_summary"] = {
                 "total_parameters": self.arch_info.get("total_parameters"),
                 "primary_architecture_type": self.arch_info.get("primary_architecture_type"),
                 "complexity_category": self.arch_info.get("complexity_category"),
            } # Or simply: self.arch_info

        logger.debug("Generated TrainingAnalyzer summary.")
        return s