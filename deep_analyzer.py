# TrainSense/deep_analyzer.py
import logging
from typing import Dict, Any, List, Optional, Iterable, Union # Added Iterable, Union
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader # For type hinting

# Import other TrainSense components
from .analyzer import TrainingAnalyzer
from .arch_analyzer import ArchitectureAnalyzer
from .model_profiler import ModelProfiler
from .system_diagnostics import SystemDiagnostics
from .optimizer import OptimizerHelper
from .gradient_analyzer import GradientAnalyzer # Import GradientAnalyzer
from .utils import format_bytes, format_time     # Import formatting utilities

# Get a logger instance specific to this module
logger = logging.getLogger(__name__)

class DeepAnalyzer:
    """
    Orchestrates various analysis components (Training, Architecture, Profiler, System, Gradient)
    to generate a comprehensive diagnostic report and aggregated recommendations.
    """
    # --- Thresholds for Recommendations ---
    HIGH_CPU_USAGE_THRESHOLD = 85.0         # System-wide CPU usage percentage
    HIGH_MEM_USAGE_THRESHOLD_PERCENT = 85.0 # System-wide RAM usage percentage
    PROFILING_HIGH_MEM_MB_THRESHOLD = 2 * 1024 # Peak memory usage (MB) during profiling considered high
    HIGH_DATA_LOAD_PERCENT_THRESHOLD = 30.0 # If >30% of training step time is data loading -> Warning
    LOW_GRAD_NORM_THRESHOLD = 1e-6          # Global gradient norm below this might indicate vanishing gradients
    HIGH_GRAD_NORM_THRESHOLD = 1e3          # Global gradient norm above this might indicate exploding gradients
    GPU_UTILIZATION_LOW_THRESHOLD = 50.0    # Percentage below which GPU util might be considered low
    CPU_BOTTLENECK_GPU_UTIL_THRESHOLD = 75.0 # If GPU util < this AND CPU util is high -> potential CPU bottleneck
    # ------------------------------------

    def __init__(self,
                 training_analyzer: TrainingAnalyzer,
                 arch_analyzer: ArchitectureAnalyzer,
                 model_profiler: ModelProfiler,
                 system_diag: SystemDiagnostics,
                 gradient_analyzer: Optional[GradientAnalyzer] = None): # Make gradient_analyzer optional
        """
        Initializes the DeepAnalyzer with instances of other analyzer components.

        Args:
            training_analyzer (TrainingAnalyzer): Analyzer for training hyperparameters.
            arch_analyzer (ArchitectureAnalyzer): Analyzer for model architecture.
            model_profiler (ModelProfiler): Profiler for model inference and training steps.
            system_diag (SystemDiagnostics): Component for dynamic system resource diagnostics.
            gradient_analyzer (Optional[GradientAnalyzer]): Analyzer for gradient statistics.
                                                           Defaults to None.
        """
        self.training_analyzer = training_analyzer
        self.arch_analyzer = arch_analyzer
        self.model_profiler = model_profiler
        self.system_diag = system_diag
        # Store the optional GradientAnalyzer
        self.gradient_analyzer = gradient_analyzer
        logger.info("DeepAnalyzer initialized.")
        if gradient_analyzer:
            logger.info("GradientAnalyzer provided and included in potential analysis.")
        else:
            logger.info("GradientAnalyzer not provided. Gradient analysis will be skipped.")


    def comprehensive_report(self,
                             profile_inference: bool = True,
                             profile_training: bool = False, # Default OFF for training profile (needs more setup)
                             gradient_analysis: bool = False, # Default OFF for gradient analysis (needs backward pass)
                             inference_input_shape: Optional[tuple] = None,
                             # Required arguments if profile_training is True
                             training_data_loader: Optional[Iterable] = None,
                             criterion: Optional[nn.Module] = None,
                             optimizer: Optional[Optimizer] = None,
                             # --- Optional parameters for profiling ---
                             profile_iterations: int = 50, # Iterations for inference profiling
                             train_profile_iterations: int = 10, # Iterations for training step profiling
                             profiler_warmup: int = 5, # Warmup iterations for profiling
                             profiler_use_torch: bool = True, # Use torch.profiler for detailed breakdown
                             profiler_sort_by: str = "self_cpu_time_total", # How to sort profiler table
                             profiler_row_limit: int = 15 # Rows in profiler table summary
                            ) -> Dict[str, Any]:
        """
        Generates a comprehensive diagnostic report by running selected analyses.

        Args:
            profile_inference (bool): Whether to run inference profiling. Defaults to True.
            profile_training (bool): Whether to run training step profiling. Requires
                                     `training_data_loader`, `criterion`, and `optimizer`. Defaults to False.
            gradient_analysis (bool): Whether to run gradient analysis. Requires `gradient_analyzer`
                                      to be provided during init AND `model.backward()` to have been
                                      called just before this report. Defaults to False.
            inference_input_shape (Optional[tuple]): Input shape for inference profiling (e.g., (1, 3, 224, 224)).
                                                     If None, attempts to use estimated shape from arch_analyzer.
            training_data_loader (Optional[Iterable]): DataLoader or iterable for training profiling.
            criterion (Optional[nn.Module]): Loss function for training profiling.
            optimizer (Optional[Optimizer]): Optimizer for training profiling.
            profile_iterations (int): Number of iterations for inference profiling.
            train_profile_iterations (int): Number of iterations for training step profiling.
            profiler_warmup (int): Number of warmup iterations before profiling starts.
            profiler_use_torch (bool): Use `torch.profiler` for detailed op analysis.
            profiler_sort_by (str): Key to sort `torch.profiler` results by.
            profiler_row_limit (int): Number of rows to show in the `torch.profiler` summary table.

        Returns:
            Dict[str, Any]: A dictionary containing the results of all performed analyses
                            and aggregated recommendations. Structure includes keys like:
                            'hyperparameter_analysis', 'architecture_analysis', 'system_diagnostics',
                            'inference_profiling', 'training_step_profiling', 'gradient_analysis',
                            'overall_recommendations'. Each section may contain an 'error' key if failed.
        """
        logger.info(f"Generating comprehensive report (Profile Inference: {profile_inference}, "
                    f"Profile Training: {profile_training}, Analyze Gradients: {gradient_analysis})")
        report: Dict[str, Any] = {} # Initialize the main report dictionary
        profiler_error_occurred = False # Flag to track if any profiling section fails

        # --- 1. Hyperparameter Analysis (Always run) ---
        logger.info("Running hyperparameter analysis...")
        try:
            # Fetch recommendations and suggested adjustments from TrainingAnalyzer
            hp_recommendations = self.training_analyzer.check_hyperparameters()
            hp_adjustments = self.training_analyzer.auto_adjust()
            report["hyperparameter_analysis"] = {
                "current_values": {
                    "batch_size": self.training_analyzer.batch_size,
                    "learning_rate": self.training_analyzer.learning_rate,
                    "epochs": self.training_analyzer.epochs,
                },
                "recommendations": hp_recommendations,
                "suggested_adjustments": hp_adjustments
            }
            logger.info("Hyperparameter analysis completed.")
        except Exception as e:
            logger.error(f"Hyperparameter analysis failed: {e}", exc_info=True)
            report["hyperparameter_analysis"] = {"error": f"Failed to analyze hyperparameters: {e}"}


        # --- 2. Architecture Analysis (Always run) ---
        logger.info("Running architecture analysis...")
        try:
            # Get the full analysis dictionary from ArchitectureAnalyzer
            arch_info = self.arch_analyzer.analyze()
            report["architecture_analysis"] = arch_info
            logger.info("Architecture analysis completed.")
        except Exception as e:
            logger.error(f"Architecture analysis failed: {e}", exc_info=True)
            report["architecture_analysis"] = {"error": f"Failed to analyze architecture: {e}"}
            arch_info = {} # Ensure arch_info exists as an empty dict for later steps


        # --- 3. System Diagnostics (Always run) ---
        logger.info("Running system diagnostics...")
        try:
            # Get current system diagnostic snapshot
            sys_info = self.system_diag.diagnostics()
            report["system_diagnostics"] = sys_info
            logger.info("System diagnostics completed.")
        except Exception as e:
             logger.error(f"System diagnostics failed: {e}", exc_info=True)
             report["system_diagnostics"] = {"error": f"Failed to gather system diagnostics: {e}"}


        # --- 4. Inference Profiling (Optional) ---
        if profile_inference:
            logger.info("Running inference profiling...")
            input_shape = inference_input_shape
            # Try to use estimated shape if specific shape not provided
            if input_shape is None:
                 input_shape = arch_info.get("estimated_input_shape")
                 if input_shape is None:
                      logger.warning("Cannot perform inference profiling: Input shape not provided and could not be estimated from architecture.")
                      report["inference_profiling"] = {"error": "Input shape not provided and could not be estimated."}
                      profiler_error_occurred = True
                 else:
                      logger.info(f"Using estimated input shape for inference profiling: {input_shape}")
                      # Modify estimated shape for typical inference (batch size 1)
                      # Ensure it remains a tuple
                      input_shape = (1,) + tuple(input_shape[1:])
                      logger.info(f"Adjusted estimated shape for profiling to: {input_shape}")

            # Proceed only if we have a valid input shape and no prior error
            if input_shape and not profiler_error_occurred:
                try:
                    inf_results = self.model_profiler.profile_model(
                         input_shape=input_shape,
                         iterations=profile_iterations,
                         warmup=profiler_warmup,
                         use_torch_profiler=profiler_use_torch,
                         profiler_sort_by=profiler_sort_by,
                         profiler_row_limit=profiler_row_limit
                         # Pass other relevant profiler args if needed
                    )
                    report["inference_profiling"] = inf_results
                    logger.info("Inference profiling completed.")
                    if inf_results.get("error"): profiler_error_occurred = True # Check for errors reported by the profiler itself
                except Exception as e:
                    logger.error(f"Inference profiling failed with exception: {e}", exc_info=True)
                    report["inference_profiling"] = {"error": f"Inference profiling crashed: {e}"}
                    profiler_error_occurred = True


        # --- 5. Training Step Profiling (Optional) ---
        if profile_training:
            logger.info("Running training step profiling...")
            # Check if all necessary components are provided
            if not all([training_data_loader, criterion, optimizer]):
                err_msg = "Training profiling requested but required arguments (data_loader, criterion, optimizer) were not provided."
                logger.error(err_msg)
                report["training_step_profiling"] = {"error": err_msg}
                profiler_error_occurred = True
            else:
                 # Proceed with training step profiling
                 try:
                    train_results = self.model_profiler.profile_training_step(
                        data_loader=training_data_loader,
                        criterion=criterion,
                        optimizer=optimizer,
                        iterations=train_profile_iterations,
                        warmup=profiler_warmup,
                        use_torch_profiler=profiler_use_torch,
                        profiler_sort_by=profiler_sort_by,
                        profiler_row_limit=profiler_row_limit
                        # Pass other relevant profiler args if needed
                    )
                    report["training_step_profiling"] = train_results
                    logger.info("Training step profiling completed.")
                    if train_results.get("error"): profiler_error_occurred = True # Check for errors reported by the profiler
                 except Exception as e:
                    logger.error(f"Training step profiling failed with exception: {e}", exc_info=True)
                    report["training_step_profiling"] = {"error": f"Training step profiling crashed: {e}"}
                    profiler_error_occurred = True


        # --- 6. Gradient Analysis (Optional) ---
        if gradient_analysis:
            logger.info("Running gradient analysis...")
            if self.gradient_analyzer is None:
                err_msg = "Gradient analysis requested, but GradientAnalyzer was not provided during DeepAnalyzer initialization."
                logger.error(err_msg)
                report["gradient_analysis"] = {"error": err_msg}
            else:
                try:
                    # --- IMPORTANT REMINDER ---
                    # Assuming model.backward() was called *just before* this report generation.
                    # The analyzer itself doesn't perform the backward pass.
                    logger.warning("Gradient analysis relies on `model.backward()` having been called recently. Ensure gradients exist.")
                    # Get the summary from the GradientAnalyzer
                    grad_summary = self.gradient_analyzer.summary() # Assumes L2 norm by default
                    report["gradient_analysis"] = grad_summary

                    # Check for specific issues reported by the analyzer
                    if grad_summary.get("error"):
                         logger.warning(f"Gradient analysis summary reported an error: {grad_summary['error']}")
                    elif grad_summary.get("num_params_with_grads", 0) == 0 and not grad_summary.get("error"):
                         warn_msg = "Gradient analysis ran but found no gradients. Ensure `model.backward()` was called before generating this report."
                         logger.warning(warn_msg)
                         # Add a specific warning to the report
                         report["gradient_analysis"]["warning"] = warn_msg
                    else:
                        logger.info(f"Gradient analysis completed. Found grads for {grad_summary.get('num_params_with_grads')} params.")

                except Exception as e:
                    logger.error(f"Gradient analysis failed with exception: {e}", exc_info=True)
                    report["gradient_analysis"] = {"error": f"Gradient analysis crashed: {e}"}


        # --- 7. Aggregate Recommendations ---
        logger.info("Aggregating recommendations from all analyses...")
        try:
            report["overall_recommendations"] = self._aggregate_recommendations(report)
            logger.info(f"Generated {len(report['overall_recommendations'])} overall recommendations.")
        except Exception as e:
             logger.error(f"Failed to aggregate recommendations: {e}", exc_info=True)
             report["overall_recommendations"] = ["Error: Failed to generate aggregated recommendations."]

        logger.info("Comprehensive report generation complete.")
        return report


    def _aggregate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """
        Combines recommendations from different analysis sections into a single list.

        Args:
            report (Dict[str, Any]): The comprehensive report dictionary containing results
                                     from various analyses.

        Returns:
            List[str]: A list of unique, sorted recommendation strings.
        """
        recommendations = []
        profiling_rec_added = False # Flag to avoid redundant messages about profiling results

        # --- Hyperparameter Recommendations ---
        hp_analysis = report.get("hyperparameter_analysis", {})
        if "error" not in hp_analysis:
            recommendations.extend(hp_analysis.get("recommendations", []))
        else:
            recommendations.append(f"[Error] Hyperparameter analysis failed: {hp_analysis['error']}")

        # --- Architecture Recommendations ---
        arch_analysis = report.get("architecture_analysis", {})
        if "error" not in arch_analysis:
            # Add the main recommendation string from arch_analyzer
            recommendations.extend(arch_analysis.get("recommendations", ["No specific architecture recommendations available."])) # Use extend for list

            # Suggest Optimizer based on architecture info
            total_params = arch_analysis.get("total_parameters", 0)
            layer_count = arch_analysis.get("layer_count", 0)
            primary_arch = arch_analysis.get("primary_architecture_type", "Unknown")
            optimizer_rec = OptimizerHelper.suggest_optimizer(total_params, layer_count, primary_arch)
            recommendations.append(f"[Optimization] Based on model ({primary_arch}, {total_params:,} params), suggested optimizer: {optimizer_rec}.")
        else:
            recommendations.append(f"[Error] Architecture analysis failed: {arch_analysis['error']}")


        # --- Inference Profiling Recommendations ---
        inf_profiling_data = report.get("inference_profiling", {})
        if "error" not in inf_profiling_data and inf_profiling_data:
            avg_time_ms = inf_profiling_data.get('avg_total_time_ms', 0)
            throughput = inf_profiling_data.get('throughput_samples_per_sec', 0)
            if avg_time_ms > 0:
                 recommendations.append(f"[Inference Perf] Avg Inference Time: {avg_time_ms:.2f} ms | Throughput: {throughput:.1f} samples/sec.")
                 profiling_rec_added = True
            else:
                 recommendations.append("[Inference Perf] Inference time is zero or invalid; verify profiling setup.")

            # Memory Usage (Inference)
            mem_usage_mb = inf_profiling_data.get("max_memory_allocated_mb", 0)
            if mem_usage_mb and mem_usage_mb > self.PROFILING_HIGH_MEM_MB_THRESHOLD:
                recommendations.append(f"[Resource Usage] High peak GPU memory ({mem_usage_mb:.1f} MB) during inference profiling. Consider optimization (quantization, pruning) or larger GPU if this is a bottleneck.")

            # Device Utilization (Inference) from Profiler Data
            prof_data_inf = inf_profiling_data.get("profiler_data", {})
            cpu_time_pct_inf = prof_data_inf.get('avg_cpu_time_percent', None)
            gpu_time_pct_inf = prof_data_inf.get('avg_gpu_time_percent', None)
            if cpu_time_pct_inf is not None and gpu_time_pct_inf is not None:
                 recommendations.append(f"[Inference Perf] Device Utilization (Profiler): CPU {cpu_time_pct_inf:.1f}%, GPU {gpu_time_pct_inf:.1f}%.")
                 # Check for potential bottlenecks based on utilization
                 if gpu_time_pct_inf < self.GPU_UTILIZATION_LOW_THRESHOLD and avg_time_ms > 5: # Low GPU usage and non-trivial time
                     recommendations.append("[Bottleneck?] Low GPU utilization during inference might indicate I/O bottlenecks, small batch sizes, or CPU-bound operations.")
                 elif gpu_time_pct_inf < self.CPU_BOTTLENECK_GPU_UTIL_THRESHOLD and cpu_time_pct_inf > self.GPU_UTILIZATION_LOW_THRESHOLD:
                      # GPU not fully utilized, but CPU is somewhat busy
                      recommendations.append("[Bottleneck?] Moderate GPU utilization with higher CPU usage might suggest a CPU bottleneck (e.g., data pre-processing) or suboptimal kernel launch.")

        elif "error" in inf_profiling_data:
             recommendations.append(f"[Error] Inference profiling failed: {inf_profiling_data['error']}")


        # --- Training Step Profiling Recommendations ---
        train_profiling_data = report.get("training_step_profiling", {})
        if "error" not in train_profiling_data and train_profiling_data:
             avg_step_time_ms = train_profiling_data.get('avg_step_time_ms', 0)
             if avg_step_time_ms > 0:
                 recommendations.append(f"[Training Perf] Avg Full Step Time: {avg_step_time_ms:.2f} ms.")
                 profiling_rec_added = True

                 # Data Loading Bottleneck Check
                 data_load_perc = train_profiling_data.get('percent_time_data_total_load', 0)
                 if data_load_perc > self.HIGH_DATA_LOAD_PERCENT_THRESHOLD:
                      data_fetch_perc = train_profiling_data.get('percent_time_data_fetch', 0)
                      data_prep_perc = train_profiling_data.get('percent_time_data_prep', 0)
                      recommendations.append(f"[Bottleneck?] High time ({data_load_perc:.1f}%) spent in DataLoader (Fetch: {data_fetch_perc:.1f}%, Prep: {data_prep_perc:.1f}%). Consider: increasing `num_workers` in DataLoader, using `pin_memory=True`, optimizing data augmentations/transforms, checking disk I/O performance, or pre-fetching data.")

                 # Forward/Backward Pass Time Check
                 forward_perc = train_profiling_data.get('percent_time_forward', 0)
                 backward_perc = train_profiling_data.get('percent_time_backward', 0)
                 if backward_perc > 60: # If backward is dominant
                      recommendations.append(f"[Training Perf] Backward pass ({backward_perc:.1f}%) dominates step time. Expected for large models, but verify if specific ops within backward are slow using torch.profiler details. Consider gradient accumulation if step time is too high.")
                 elif forward_perc > 60: # If forward is dominant
                      recommendations.append(f"[Training Perf] Forward pass ({forward_perc:.1f}%) dominates step time. Check profiler details for slow layers/operations within the forward pass.")

             else:
                  recommendations.append("[Training Perf] Step time is zero or invalid; verify profiling setup.")

             # Memory Usage (Training)
             train_mem_mb = train_profiling_data.get("max_memory_allocated_mb", 0)
             if train_mem_mb and train_mem_mb > self.PROFILING_HIGH_MEM_MB_THRESHOLD:
                 recommendations.append(f"[Resource Usage] High peak GPU memory ({train_mem_mb:.1f} MB) during training step profiling. If facing OOMs, consider: reducing batch size, using gradient accumulation, enabling mixed precision (AMP), optimizing model layers (e.g., activation checkpointing), or using a larger GPU.")

             # Device Utilization (Training) from Profiler Data
             prof_data_train = train_profiling_data.get("profiler_data", {})
             cpu_time_pct_train = prof_data_train.get('avg_cpu_time_percent', None)
             gpu_time_pct_train = prof_data_train.get('avg_gpu_time_percent', None)
             if cpu_time_pct_train is not None and gpu_time_pct_train is not None:
                  recommendations.append(f"[Training Perf] Device Utilization (Profiler): CPU {cpu_time_pct_train:.1f}%, GPU {gpu_time_pct_train:.1f}%.")
                  # Similar bottleneck checks as inference
                  if gpu_time_pct_train < self.GPU_UTILIZATION_LOW_THRESHOLD and avg_step_time_ms > 10:
                      recommendations.append("[Bottleneck?] Low GPU utilization during training suggests potential bottlenecks in data loading, CPU processing, or small workload size.")
                  elif gpu_time_pct_train < self.CPU_BOTTLENECK_GPU_UTIL_THRESHOLD and cpu_time_pct_train > self.GPU_UTILIZATION_LOW_THRESHOLD:
                       recommendations.append("[Bottleneck?] Moderate GPU utilization with higher CPU usage might indicate a CPU bottleneck (data loading/prep) or inefficient GPU kernel usage.")

        elif "error" in train_profiling_data:
             recommendations.append(f"[Error] Training step profiling failed: {train_profiling_data['error']}")


        # --- Gradient Analysis Recommendations ---
        grad_analysis_data = report.get("gradient_analysis", {})
        if "error" not in grad_analysis_data and grad_analysis_data:
             # Check for preliminary warnings (e.g., no gradients found)
             if grad_analysis_data.get("warning"):
                  recommendations.append(f"[Warning][Gradients] {grad_analysis_data['warning']}")

             # Check for NaN/Inf gradients (Critical issue)
             num_nan = grad_analysis_data.get("num_params_nan_grad", 0)
             num_inf = grad_analysis_data.get("num_params_inf_grad", 0)
             if num_nan > 0 or num_inf > 0:
                 recommendations.append(f"[CRITICAL][Gradients] Found {num_nan} NaN and {num_inf} Inf gradients! Training is likely unstable or diverging. Check: learning rate (too high?), data normalization/scaling, numerical stability of operations (e.g., log(0), sqrt(<0)), AMP usage (ensure GradScaler is used correctly).")

             # Check gradient norm magnitude (only if no NaN/Inf)
             global_norm = grad_analysis_data.get("global_grad_norm_L2")
             if global_norm is not None and not (num_nan > 0 or num_inf > 0):
                 recommendations.append(f"[Gradients] Global Gradient Norm (L2): {global_norm:.3e}") # Report the norm
                 # Check for exploding gradients
                 if global_norm > self.HIGH_GRAD_NORM_THRESHOLD:
                     recommendations.append(f"[Warning][Gradients] High global gradient norm ({global_norm:.2e}) detected. Risk of exploding gradients. Consider using gradient clipping (e.g., `torch.nn.utils.clip_grad_norm_`).")
                 # Check for vanishing gradients
                 elif global_norm < self.LOW_GRAD_NORM_THRESHOLD:
                     recommendations.append(f"[Warning][Gradients] Low global gradient norm ({global_norm:.2e}) detected. Potential vanishing gradient issue, hindering learning. Check: activation functions (avoid saturated sigmoids/tanh in deep nets), weight initialization, normalization layers, or overall architecture depth.")
             elif global_norm is None and not (num_nan > 0 or num_inf > 0) and "warning" not in grad_analysis_data:
                 recommendations.append("[Warning][Gradients] Global gradient norm could not be calculated (check logs).")

             # Check Grad/Param ratio if available
             avg_ratio = grad_analysis_data.get("avg_grad_param_norm_ratio")
             if avg_ratio is not None:
                 recommendations.append(f"[Gradients] Average Gradient/Parameter Norm Ratio (L2): {avg_ratio:.2e}. Ratios << 1 or >> 1 might warrant investigation depending on the layer/optimizer.")


        elif "error" in grad_analysis_data:
             recommendations.append(f"[Error] Gradient analysis failed: {grad_analysis_data['error']}")


        # --- System Diagnostics Recommendations ---
        sys_diag_data = report.get("system_diagnostics", {})
        if "error" not in sys_diag_data and sys_diag_data:
            cpu_usage = sys_diag_data.get("cpu_usage_percent", 0)
            mem_usage_perc = sys_diag_data.get("memory_usage_percent", 0)

            # High system-wide CPU usage
            if cpu_usage > self.HIGH_CPU_USAGE_THRESHOLD:
                recommendations.append(f"[Resource Usage] High overall system CPU usage ({cpu_usage:.1f}%) detected. If training is slow and GPU util is low, check for CPU-bound data loading/preprocessing or other demanding processes on the system.")
            # High system-wide RAM usage
            if mem_usage_perc > self.HIGH_MEM_USAGE_THRESHOLD_PERCENT:
                mem_total_gb = sys_diag_data.get("memory_total_bytes", 0) / (1024**3)
                recommendations.append(f"[Resource Usage] High overall system RAM usage ({mem_usage_perc:.1f}% of {mem_total_gb:.1f} GB). This might slow down disk caching or other processes. Consider closing unused applications or increasing system RAM if it impacts training I/O.")
        elif "error" in sys_diag_data:
            recommendations.append(f"[Error] System diagnostics failed: {sys_diag_data['error']}")


        # --- Final Cleanup ---
        # Remove duplicates and filter out None/empty strings
        unique_recs = sorted(list(set(filter(None, recommendations))))
        return unique_recs