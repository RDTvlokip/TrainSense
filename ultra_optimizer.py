# TrainSense/ultra_optimizer.py
import logging
from typing import Dict, Any, Optional

# Added import
from .optimizer import OptimizerHelper

logger = logging.getLogger(__name__)

class UltraOptimizer:
    MIN_MEM_GB_FOR_HIGH_BATCH = 16
    HIGH_MEM_GB_FOR_VERY_HIGH_BATCH = 32
    LARGE_MODEL_PARAMS_THRESHOLD = 50_000_000
    LARGE_DATASET_SIZE_THRESHOLD = 1_000_000

    def __init__(self,
                 training_data_stats: Dict[str, Any],
                 model_arch_stats: Dict[str, Any],
                 system_config_summary: Dict[str, Any]):

        # Basic validation
        if not isinstance(training_data_stats, dict):
            raise TypeError("training_data_stats must be a dictionary.")
        if not isinstance(model_arch_stats, dict):
            raise TypeError("model_arch_stats must be a dictionary.")
        if not isinstance(system_config_summary, dict):
             raise TypeError("system_config_summary must be a dictionary.")


        self.training_data_stats = training_data_stats
        self.model_arch_stats = model_arch_stats
        self.system_config_summary = system_config_summary
        logger.info("UltraOptimizer initialized with data, model, and system stats.")
        logger.debug(f"Data Stats: {training_data_stats}")
        logger.debug(f"Model Stats: {model_arch_stats}")
        logger.debug(f"System Stats: {system_config_summary}")


    def compute_heuristic_hyperparams(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        recommendations: Dict[str, str] = {}

        # --- Batch Size ---
        total_mem_gb = self.system_config_summary.get("total_memory_gb")
        gpu_info = self.system_config_summary.get("gpu_info", [])
        avg_gpu_mem_mb = None
        if gpu_info and isinstance(gpu_info, list) and len(gpu_info) > 0:
             # Ensure memory values are numbers and handle potential N/A or missing keys
             valid_mems = [gpu.get('memory_total_mb', 0) for gpu in gpu_info if isinstance(gpu.get('memory_total_mb'), (int, float))]
             if valid_mems:
                 total_gpu_mem = sum(valid_mems)
                 avg_gpu_mem_mb = total_gpu_mem / len(valid_mems)


        model_params = self.model_arch_stats.get("total_parameters", 0)
        batch_size = 32 # Default starting point

        if avg_gpu_mem_mb: # GPU present and memory info valid
            if avg_gpu_mem_mb < 6 * 1024:
                 batch_size = 16
                 recommendations["batch_size"] = f"Low GPU memory ({avg_gpu_mem_mb:.0f}MB avg). Starting with small batch size."
            elif avg_gpu_mem_mb < self.MIN_MEM_GB_FOR_HIGH_BATCH * 1024:
                 batch_size = 32
                 recommendations["batch_size"] = f"Moderate GPU memory ({avg_gpu_mem_mb:.0f}MB avg). Starting with medium batch size."
            elif avg_gpu_mem_mb < self.HIGH_MEM_GB_FOR_VERY_HIGH_BATCH * 1024:
                 batch_size = 64
                 recommendations["batch_size"] = f"High GPU memory ({avg_gpu_mem_mb:.0f}MB avg). Starting with larger batch size."
            else:
                 batch_size = 128
                 recommendations["batch_size"] = f"Very high GPU memory ({avg_gpu_mem_mb:.0f}MB avg). Starting with very large batch size."

            # Adjust based on model size
            if model_params > self.LARGE_MODEL_PARAMS_THRESHOLD:
                # Use // for integer division, ensure minimum batch size (e.g., 2 or 4)
                batch_size = max(4, batch_size // 2)
                recommendations["batch_size"] += f" Reduced due to large model size ({model_params:,} params)."
            elif model_params < 5_000_000 and avg_gpu_mem_mb > 8 * 1024 :
                 # Cap max batch size (e.g., 256 or 512)
                 batch_size = min(256, batch_size * 2)
                 recommendations["batch_size"] += f" Increased due to small model size ({model_params:,} params) and sufficient GPU memory."

        else: # CPU or no GPU info or invalid GPU info
             reason = "Assuming CPU or moderate RAM"
             if total_mem_gb is not None and isinstance(total_mem_gb, (int, float)):
                 if total_mem_gb < 8:
                     batch_size = 16
                     reason = "Low system RAM detected"
                 else:
                     batch_size = 32 # Keep default for moderate/high RAM on CPU
             else:
                 batch_size = 32 # Default if RAM info is missing/invalid
                 reason = "System RAM info unavailable, assuming moderate"

             recommendations["batch_size"] = f"{reason}. Starting with batch size {batch_size}."

             if model_params > self.LARGE_MODEL_PARAMS_THRESHOLD * 2: # More stringent for CPU
                  batch_size = max(4, batch_size // 2)
                  recommendations["batch_size"] += f" Reduced further due to very large model size ({model_params:,} params) likely on CPU."

        params["batch_size"] = batch_size


        # --- Learning Rate ---
        arch_type = self.model_arch_stats.get("primary_architecture_type", "Unknown")
        # Leverage OptimizerHelper for initial suggestion
        suggested_lr = OptimizerHelper.suggest_initial_learning_rate(arch_type, model_params)
        params["learning_rate"] = suggested_lr
        recommendations["learning_rate"] = f"Suggested initial LR ({suggested_lr:.1e}) based on architecture ({arch_type}) and size ({model_params:,} params)."


        # --- Epochs ---
        # Ensure data_size is a number, default to 0 if missing/invalid
        data_size = self.training_data_stats.get("data_size", 0)
        if not isinstance(data_size, (int, float)):
            logger.warning(f"Invalid data_size type ({type(data_size)}), defaulting to 0 for epoch calculation.")
            data_size = 0

        if data_size > self.LARGE_DATASET_SIZE_THRESHOLD:
            params["epochs"] = 50 # Fewer epochs often needed for large datasets
            recommendations["epochs"] = f"Large dataset detected ({data_size:,} samples). Suggesting moderate epochs ({params['epochs']})."
        elif data_size < 10000 and data_size > 0:
             params["epochs"] = 100 # More epochs might be needed for small datasets
             recommendations["epochs"] = f"Small dataset detected ({data_size:,} samples). Suggesting more epochs ({params['epochs']})."
        else:
             params["epochs"] = 75 # Default middle ground
             recommendations["epochs"] = f"Dataset size moderate or unknown ({data_size:,} samples). Suggesting standard number of epochs ({params['epochs']})."


        # --- Optimizer ---
        suggested_optimizer_full = OptimizerHelper.suggest_optimizer(
             model_params,
             self.model_arch_stats.get("layer_count", 0),
             arch_type
        )
        # Extract the base optimizer name (e.g., "AdamW" from "AdamW (...)")
        base_optimizer = suggested_optimizer_full.split(" ")[0]
        params["optimizer_name"] = base_optimizer
        recommendations["optimizer_name"] = f"Suggested: {suggested_optimizer_full}."

        # --- Scheduler ---
        suggested_scheduler_full = OptimizerHelper.suggest_learning_rate_scheduler(base_optimizer)
        # Just the name, e.g. CosineAnnealingLR
        base_scheduler = suggested_scheduler_full.split(" ")[0].split("/")[0] # Take first if multiple options like StepLR/MultiStepLR
        params["scheduler_name"] = base_scheduler
        recommendations["scheduler_name"] = f"Suggested: {suggested_scheduler_full}."


        logger.info(f"Computed heuristic hyperparameters: {params}")
        logger.debug(f"Recommendations reasoning: {recommendations}")

        # Return both params and the reasoning behind them
        return {"hyperparameters": params, "reasoning": recommendations}