# TrainSense/ultra_optimizer.py
import logging
from typing import Dict, Any, Optional

# Import OptimizerHelper to leverage its suggestions
from .optimizer import OptimizerHelper

# Get a logger instance specific to this module
logger = logging.getLogger(__name__)

class UltraOptimizer:
    """
    Provides heuristic suggestions for initial training hyperparameters
    (batch size, learning rate, epochs, optimizer, scheduler) based on
    dataset statistics, model architecture information, and system configuration.

    Note: These are *heuristic* starting points, not guaranteed optimal values.
    They aim to provide a reasonable baseline configuration.
    """

    # --- Heuristic Thresholds ---
    # GPU Memory (in GB) for Batch Size Tiers
    MIN_MEM_GB_FOR_HIGH_BATCH = 16    # GPUs >= this might support batch_size 64+
    HIGH_MEM_GB_FOR_VERY_HIGH_BATCH = 32 # GPUs >= this might support batch_size 128+
    # Model Size
    LARGE_MODEL_PARAMS_THRESHOLD = OptimizerHelper.PARAM_THRESHOLD_LARGE # Use consistent threshold (e.g., 50M)
    # Dataset Size
    LARGE_DATASET_SIZE_THRESHOLD = 1_000_000 # Datasets > this might need fewer epochs
    SMALL_DATASET_SIZE_THRESHOLD = 10_000   # Datasets < this might need more epochs
    # --------------------------

    def __init__(self,
                 training_data_stats: Dict[str, Any],
                 model_arch_stats: Dict[str, Any],
                 system_config_summary: Dict[str, Any]):
        """
        Initializes the UltraOptimizer with context information.

        Args:
            training_data_stats (Dict[str, Any]): Dictionary containing statistics about the training data
                                                  (e.g., {'data_size': 100000}).
            model_arch_stats (Dict[str, Any]): Dictionary containing model architecture analysis results
                                               (e.g., from ArchitectureAnalyzer.analyze()). Should include
                                               'total_parameters' and 'primary_architecture_type'.
            system_config_summary (Dict[str, Any]): A summary dictionary of the system configuration
                                                    (e.g., from SystemConfig.get_summary()). Should include
                                                    'total_memory_gb', 'gpu_info' (list), 'gpu_count'.

        Raises:
            TypeError: If input arguments are not dictionaries.
        """
        # Basic type validation for inputs
        if not isinstance(training_data_stats, dict):
            raise TypeError("training_data_stats must be a dictionary.")
        if not isinstance(model_arch_stats, dict):
            raise TypeError("model_arch_stats must be a dictionary.")
        if not isinstance(system_config_summary, dict):
             raise TypeError("system_config_summary must be a dictionary.")

        self.training_data_stats = training_data_stats
        self.model_arch_stats = model_arch_stats
        self.system_config_summary = system_config_summary
        logger.info("UltraOptimizer initialized.")
        # Log the received stats at debug level for verification
        logger.debug(f"Received Data Stats: {self.training_data_stats}")
        logger.debug(f"Received Model Stats: {self.model_arch_stats}")
        logger.debug(f"Received System Stats: {self.system_config_summary}")


    def compute_heuristic_hyperparams(self) -> Dict[str, Any]:
        """
        Computes suggested initial hyperparameters based on the provided context.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'hyperparameters' (Dict): The suggested hyperparameter values
                  ('batch_size', 'learning_rate', 'epochs', 'optimizer_name', 'scheduler_name').
                - 'reasoning' (Dict): Explanations for each suggested value.
        """
        logger.info("Computing heuristic hyperparameter suggestions...")
        params: Dict[str, Any] = {}      # To store suggested hyperparameter values
        recommendations: Dict[str, str] = {} # To store the reasoning behind suggestions

        # --- 1. Batch Size Suggestion ---
        # Based primarily on available GPU memory and secondarily on model size.
        gpu_info = self.system_config_summary.get("gpu_info", [])
        gpu_count = self.system_config_summary.get("gpu_count", 0)
        avg_gpu_mem_mb: Optional[float] = None

        if gpu_count > 0 and isinstance(gpu_info, list) and gpu_info:
             # Calculate average GPU memory if available
             valid_mems = [
                 mem for gpu in gpu_info
                 # Ensure 'memory_total_mb' exists and is a number
                 if isinstance(mem := gpu.get('memory_total_mb'), (int, float)) and mem > 0
             ]
             if valid_mems:
                 total_gpu_mem_mb = sum(valid_mems)
                 avg_gpu_mem_mb = total_gpu_mem_mb / len(valid_mems)
                 logger.debug(f"Average GPU Memory: {avg_gpu_mem_mb:.0f} MB across {len(valid_mems)} GPUs.")
             else:
                 logger.warning("GPU detected, but could not extract valid 'memory_total_mb' from gpu_info.")
        else:
             logger.debug("No GPUs detected or gpu_info unavailable/invalid in system summary.")

        # Get model size
        model_params = self.model_arch_stats.get("total_parameters", 0)
        if not isinstance(model_params, int) or model_params <= 0:
            logger.warning(f"Invalid 'total_parameters' ({model_params}). Using 0 for batch size logic.")
            model_params = 0

        # Set default batch size and adjust based on context
        batch_size = 32 # Default starting point (reasonable for moderate GPU/CPU)
        bs_reason = "Default starting point."

        if avg_gpu_mem_mb: # GPU is present and memory info is valid
            avg_gpu_mem_gb = avg_gpu_mem_mb / 1024
            if avg_gpu_mem_gb < 8: # Low GPU memory (< 8GB)
                 batch_size = 16
                 bs_reason = f"Low avg GPU memory ({avg_gpu_mem_gb:.1f}GB)."
            elif avg_gpu_mem_gb < self.MIN_MEM_GB_FOR_HIGH_BATCH: # Moderate GPU memory (8-16GB)
                 batch_size = 32
                 bs_reason = f"Moderate avg GPU memory ({avg_gpu_mem_gb:.1f}GB)."
            elif avg_gpu_mem_gb < self.HIGH_MEM_GB_FOR_VERY_HIGH_BATCH: # High GPU memory (16-32GB)
                 batch_size = 64
                 bs_reason = f"High avg GPU memory ({avg_gpu_mem_gb:.1f}GB)."
            else: # Very high GPU memory (32GB+)
                 batch_size = 128
                 bs_reason = f"Very high avg GPU memory ({avg_gpu_mem_gb:.1f}GB)."

            # Adjust batch size based on model size (on GPU)
            if model_params > self.LARGE_MODEL_PARAMS_THRESHOLD:
                reduction_factor = 2 # Halve for large models
                new_bs = max(4, batch_size // reduction_factor) # Ensure minimum BS (e.g., 4)
                if new_bs < batch_size:
                     bs_reason += f" Reduced to {new_bs} due to large model size ({model_params:,} params)."
                     batch_size = new_bs
            elif model_params > 0 and model_params < OptimizerHelper.PARAM_THRESHOLD_SMALL and avg_gpu_mem_gb > 8:
                 # Potential increase for small models on decent GPUs, cap reasonably
                 increase_factor = 2
                 new_bs = min(256, batch_size * increase_factor) # Cap at 256
                 if new_bs > batch_size:
                      bs_reason += f" Increased to {new_bs} due to small model size ({model_params:,} params) and sufficient GPU RAM."
                      batch_size = new_bs

        else: # Assuming CPU or unable to get GPU memory
            total_ram_gb = self.system_config_summary.get("total_memory_gb")
            reason_cpu = "Assuming CPU."
            if total_ram_gb is not None and isinstance(total_ram_gb, (int, float)):
                 if total_ram_gb < 16: # Low system RAM
                     batch_size = 16
                     reason_cpu = f"Low system RAM ({total_ram_gb:.1f}GB) detected."
                 else: # Moderate/High system RAM
                     batch_size = 32 # Keep default
                     reason_cpu = f"Moderate/High system RAM ({total_ram_gb:.1f}GB) detected."
            else: # RAM info missing
                 batch_size = 32
                 reason_cpu = "System RAM info unavailable, assuming moderate CPU setup."

            bs_reason = f"{reason_cpu} Initial BS: {batch_size}."

            # Adjust batch size based on model size (on CPU - more conservative)
            if model_params > self.LARGE_MODEL_PARAMS_THRESHOLD:
                 reduction_factor = 4 # Reduce more aggressively on CPU for large models
                 new_bs = max(4, batch_size // reduction_factor)
                 if new_bs < batch_size:
                     bs_reason += f" Reduced to {new_bs} due to large model size ({model_params:,} params) on CPU."
                     batch_size = new_bs

        params["batch_size"] = batch_size
        recommendations["batch_size"] = bs_reason


        # --- 2. Learning Rate Suggestion ---
        # Leverage OptimizerHelper based on architecture and size
        arch_type = self.model_arch_stats.get("primary_architecture_type", "Unknown")
        suggested_lr = OptimizerHelper.suggest_initial_learning_rate(arch_type, model_params)
        params["learning_rate"] = suggested_lr
        recommendations["learning_rate"] = f"Suggested initial LR ({suggested_lr:.1e}) based on architecture ('{arch_type}') and size ({model_params:,} params). Requires tuning."


        # --- 3. Epochs Suggestion ---
        # Based on dataset size heuristic
        data_size = self.training_data_stats.get("data_size", 0)
        # Validate data_size type
        if not isinstance(data_size, (int, float)) or data_size < 0:
            logger.warning(f"Invalid 'data_size' ({data_size}) in training_data_stats. Using 0 for epoch suggestion.")
            data_size = 0

        if data_size == 0:
             params["epochs"] = 50 # Default if size is unknown
             recommendations["epochs"] = f"Dataset size unknown. Suggesting default epochs ({params['epochs']}). Monitor validation loss."
        elif data_size > self.LARGE_DATASET_SIZE_THRESHOLD:
             params["epochs"] = 30 # Fewer epochs often sufficient for very large datasets
             recommendations["epochs"] = f"Large dataset detected ({data_size:,} samples). Suggesting fewer epochs ({params['epochs']}). Monitor validation loss early."
        elif data_size < self.SMALL_DATASET_SIZE_THRESHOLD:
             params["epochs"] = 100 # More epochs might be needed for small datasets to converge/avoid underfitting
             recommendations["epochs"] = f"Small dataset detected ({data_size:,} samples). Suggesting more epochs ({params['epochs']}). Watch for overfitting."
        else: # Moderate dataset size
             params["epochs"] = 50 # Default moderate number
             recommendations["epochs"] = f"Dataset size ({data_size:,} samples) is moderate. Suggesting standard epochs ({params['epochs']})."


        # --- 4. Optimizer Suggestion ---
        # Leverage OptimizerHelper
        layer_count = self.model_arch_stats.get("layer_count", 0) # Get layer count if available
        suggested_optimizer_full = OptimizerHelper.suggest_optimizer(model_params, layer_count, arch_type)
        # Extract the base optimizer name (e.g., "AdamW" from "AdamW (...)")
        base_optimizer = suggested_optimizer_full.split(" ")[0].split("/")[0].strip() # Handle "AdamW", "RMSprop or Adam" -> "RMSprop"
        params["optimizer_name"] = base_optimizer
        recommendations["optimizer_name"] = f"Suggested: {suggested_optimizer_full}"


        # --- 5. Scheduler Suggestion ---
        # Leverage OptimizerHelper based on the suggested optimizer
        suggested_scheduler_full = OptimizerHelper.suggest_learning_rate_scheduler(base_optimizer)
        # Extract the primary scheduler name
        base_scheduler = suggested_scheduler_full.split(" ")[0].split("/")[0].strip() # Handle "StepLR/MultiStepLR" -> "StepLR"
        params["scheduler_name"] = base_scheduler
        recommendations["scheduler_name"] = f"Suggested: {suggested_scheduler_full}"


        # --- Log and Return Results ---
        logger.info(f"Computed heuristic hyperparameters: {params}")
        logger.debug(f"Reasoning: {recommendations}")

        # Return both the suggested parameters and the reasoning
        return {"hyperparameters": params, "reasoning": recommendations}