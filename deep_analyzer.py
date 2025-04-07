# TrainSense/deep_analyzer.py
import logging
from typing import Dict, Any, List, Optional, Iterable # Added Iterable
# Added necessary imports
from torch.optim import Optimizer
import torch.nn as nn
from torch.utils.data import DataLoader

from .analyzer import TrainingAnalyzer
from .arch_analyzer import ArchitectureAnalyzer
from .model_profiler import ModelProfiler
from .system_diagnostics import SystemDiagnostics
from .optimizer import OptimizerHelper
# Added import
from .gradient_analyzer import GradientAnalyzer
from .utils import format_bytes

logger = logging.getLogger(__name__)

class DeepAnalyzer:
    HIGH_CPU_USAGE_THRESHOLD = 85.0
    HIGH_MEM_USAGE_THRESHOLD_PERCENT = 85.0
    PROFILING_HIGH_MEM_MB_THRESHOLD = 2 * 1024 # 2 GB
    # --- New Thresholds ---
    HIGH_DATA_LOAD_PERCENT_THRESHOLD = 30.0 # If >30% time in data load -> Warning
    LOW_GRAD_NORM_THRESHOLD = 1e-6
    HIGH_GRAD_NORM_THRESHOLD = 1e3
    # ---------------------

    def __init__(self,
                 training_analyzer: TrainingAnalyzer,
                 arch_analyzer: ArchitectureAnalyzer,
                 model_profiler: ModelProfiler,
                 system_diag: SystemDiagnostics,
                 # --- Optional Gradient Analyzer ---
                 gradient_analyzer: Optional[GradientAnalyzer] = None): # Make optional
        self.training_analyzer = training_analyzer
        self.arch_analyzer = arch_analyzer
        self.model_profiler = model_profiler
        self.system_diag = system_diag
        # --- Store Gradient Analyzer ---
        self.gradient_analyzer = gradient_analyzer
        logger.info("DeepAnalyzer initialized.")
        if gradient_analyzer:
            logger.info("GradientAnalyzer provided.")


    # --- MODIFIED comprehensive_report signature ---
    def comprehensive_report(self,
                             profile_inference: bool = True,
                             profile_training: bool = False, # Default off
                             gradient_analysis: bool = False, # Default off
                             inference_input_shape: Optional[tuple] = None,
                             # Required if profile_training is True
                             training_data_loader: Optional[Iterable] = None,
                             criterion: Optional[nn.Module] = None,
                             optimizer: Optional[Optimizer] = None,
                             # Other profiling params (can be passed down)
                             profile_iterations: int = 50, # For inference
                             train_profile_iterations: int = 10, # For training
                             **profiler_kwargs # Pass other kwargs like warmup, sort_by etc.
                            ) -> Dict[str, Any]:
        logger.info(f"Generating comprehensive report (Profile Inference: {profile_inference}, "
                    f"Profile Training: {profile_training}, Analyze Gradients: {gradient_analysis})")
        report: Dict[str, Any] = {}
        profiler_error_occurred = False # Flag to track if profiling fails

        # --- 1. Hyperparameter Analysis (Always run) ---
        try:
            report["hyperparameter_analysis"] = {
                "current_values": { "batch_size": self.training_analyzer.batch_size, "learning_rate": self.training_analyzer.learning_rate, "epochs": self.training_analyzer.epochs, },
                "recommendations": self.training_analyzer.check_hyperparameters(),
                "suggested_adjustments": self.training_analyzer.auto_adjust()
            }
        except Exception as e:
            logger.error(f"Hyperparameter analysis failed: {e}", exc_info=True)
            report["hyperparameter_analysis"] = {"error": f"Failed: {e}"}


        # --- 2. Architecture Analysis (Always run) ---
        logger.info("Running architecture analysis.")
        try:
            arch_info = self.arch_analyzer.analyze()
            report["architecture_analysis"] = arch_info
        except Exception as e:
            logger.error(f"Architecture analysis failed: {e}", exc_info=True)
            report["architecture_analysis"] = {"error": f"Failed: {e}"}
            arch_info = {} # Ensure arch_info exists for later use

        # --- 3. System Diagnostics (Always run) ---
        logger.info("Running system diagnostics.")
        try:
            sys_info = self.system_diag.diagnostics()
            report["system_diagnostics"] = sys_info
        except Exception as e:
             logger.error(f"System diagnostics failed: {e}", exc_info=True)
             report["system_diagnostics"] = {"error": f"Diagnostics failed: {str(e)}"}


        # --- 4. Inference Profiling (Optional) ---
        if profile_inference:
            logger.info("Running inference profiling.")
            input_shape = inference_input_shape
            if input_shape is None:
                 input_shape = arch_info.get("estimated_input_shape")
                 if input_shape is None:
                      logger.warning("Cannot determine input shape for inference profiling. Skipping.")
                      report["inference_profiling"] = {"error": "Input shape not provided and could not be estimated."}
                      profiler_error_occurred = True
                 else:
                      logger.info(f"Using estimated input shape for inference profiling: {input_shape}")
                      # Adjust batch size for inference profiling (usually 1 or small batch)
                      # Here we use batch size 1 as default if using estimated shape
                      input_shape = (1,) + input_shape[1:]
                      logger.info(f"Adjusted inference profiling shape to: {input_shape}")

            if input_shape and not profiler_error_occurred:
                try:
                    inf_results = self.model_profiler.profile_model(
                         input_shape=input_shape,
                         iterations=profile_iterations,
                         **profiler_kwargs # Pass down other args like warmup, use_torch_profiler, sort_by
                    )
                    report["inference_profiling"] = inf_results
                    if inf_results.get("error"): profiler_error_occurred = True
                except Exception as e:
                    logger.error(f"Inference profiling failed: {e}", exc_info=True)
                    report["inference_profiling"] = {"error": f"Failed: {str(e)}"}
                    profiler_error_occurred = True


        # --- 5. Training Step Profiling (Optional) ---
        if profile_training:
            logger.info("Running training step profiling.")
            if not all([training_data_loader, criterion, optimizer]):
                logger.error("Training profiling requested but data_loader, criterion, or optimizer not provided.")
                report["training_step_profiling"] = {"error": "Missing required arguments (data_loader, criterion, optimizer)."}
                profiler_error_occurred = True
            else:
                 try:
                    train_results = self.model_profiler.profile_training_step(
                        data_loader=training_data_loader,
                        criterion=criterion,
                        optimizer=optimizer,
                        iterations=train_profile_iterations,
                        **profiler_kwargs # Pass down other args
                    )
                    report["training_step_profiling"] = train_results
                    if train_results.get("error"): profiler_error_occurred = True
                 except Exception as e:
                    logger.error(f"Training step profiling failed: {e}", exc_info=True)
                    report["training_step_profiling"] = {"error": f"Failed: {str(e)}"}
                    profiler_error_occurred = True


        # --- 6. Gradient Analysis (Optional) ---
        if gradient_analysis:
            logger.info("Running gradient analysis.")
            if self.gradient_analyzer is None:
                logger.error("Gradient analysis requested but GradientAnalyzer was not provided during DeepAnalyzer initialization.")
                report["gradient_analysis"] = {"error": "GradientAnalyzer not initialized."}
            else:
                try:
                    # IMPORTANT: Assumes user has run backward() just before calling this report!
                    grad_summary = self.gradient_analyzer.summary()
                    report["gradient_analysis"] = grad_summary
                    if grad_summary.get("error"):
                         logger.warning(f"Gradient analysis summary reported an error: {grad_summary['error']}")
                    elif grad_summary.get("num_params_with_grads", 0) == 0:
                         logger.warning("Gradient analysis ran but found no gradients. Ensure model.backward() was called.")
                         # Add a note to the report as well
                         report["gradient_analysis"]["warning"] = "No gradients found in parameters. Ensure model.backward() was called before generating the report."

                except Exception as e:
                    logger.error(f"Gradient analysis failed: {e}", exc_info=True)
                    report["gradient_analysis"] = {"error": f"Failed: {str(e)}"}


        # --- 7. Aggregate Recommendations ---
        logger.info("Aggregating recommendations.")
        report["overall_recommendations"] = self._aggregate_recommendations(report)

        logger.info("Comprehensive report generation complete.")
        return report

    # --- MODIFIED _aggregate_recommendations ---
    def _aggregate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        recommendations = []
        profiling_rec_added = False # Flag to avoid redundant profiling messages

        # Hyperparameters
        if report.get("hyperparameter_analysis") and "error" not in report["hyperparameter_analysis"]:
            recommendations.extend(report["hyperparameter_analysis"].get("recommendations", []))

        # Architecture
        if report.get("architecture_analysis") and "error" not in report["architecture_analysis"]:
            recommendations.append(report["architecture_analysis"].get("recommendation", "No specific architecture recommendation available."))
            # Add optimizer suggestion here based on arch info
            arch_info = report["architecture_analysis"]
            total_params = arch_info.get("total_parameters", 0)
            optimizer_rec = OptimizerHelper.suggest_optimizer(total_params, arch_info.get("layer_count", 0), arch_info.get("primary_architecture_type", "Unknown"))
            recommendations.append(f"Based on model size/type, suggested optimizer: {optimizer_rec}.")

        # Inference Profiling Results
        inf_profiling_data = report.get("inference_profiling", {})
        if "error" not in inf_profiling_data and inf_profiling_data:
            mem_usage_mb = inf_profiling_data.get("max_memory_allocated_mb", 0)
            if mem_usage_mb and mem_usage_mb > self.PROFILING_HIGH_MEM_MB_THRESHOLD:
                recommendations.append(f"[Inference Profiling] High peak memory usage ({mem_usage_mb:.1f} MB) detected. Consider model optimization or resource adjustments if VRAM is limited.")

            avg_time = inf_profiling_data.get('avg_total_time_ms', 0)
            throughput = inf_profiling_data.get('throughput_samples_per_sec', 0)
            if avg_time > 0:
                 recommendations.append(f"[Inference Profiling] Avg Inference Time: {avg_time:.2f} ms | Throughput: {throughput:.2f} samples/sec.")
                 profiling_rec_added = True
            else:
                 recommendations.append("[Inference Profiling] Results suggest very fast or potentially zero execution time; verify setup.")

            cpu_time_pct = inf_profiling_data.get('avg_cpu_time_percent', None)
            gpu_time_pct = inf_profiling_data.get('avg_gpu_time_percent', None)
            if cpu_time_pct is not None and gpu_time_pct is not None:
                 recommendations.append(f"[Inference Profiling] Device Utilization: CPU {cpu_time_pct:.1f}%, GPU {gpu_time_pct:.1f}%.")
                 if gpu_time_pct < 50 and cpu_time_pct < 50 and avg_time > 5:
                     recommendations.append("[Inference Profiling] Low CPU & GPU utilization might indicate I/O bottlenecks or inefficient operations.")
                 elif gpu_time_pct < 75 and cpu_time_pct > 50 and self.training_analyzer.system_config and self.training_analyzer.system_config.get_summary().get('gpu_count', 0) > 0:
                      recommendations.append("[Inference Profiling] Relatively low GPU utilization with higher CPU usage might suggest a CPU bottleneck (e.g., data processing).")

        elif "error" in inf_profiling_data:
             recommendations.append(f"[Inference Profiling] Could not be completed: {inf_profiling_data['error']}")

        # Training Step Profiling Results
        train_profiling_data = report.get("training_step_profiling", {})
        if "error" not in train_profiling_data and train_profiling_data:
             avg_step_time = train_profiling_data.get('avg_step_time_ms', 0)
             if avg_step_time > 0:
                 recommendations.append(f"[Training Profiling] Avg Full Step Time: {avg_step_time:.2f} ms.")
                 profiling_rec_added = True
                 # Data loading bottleneck check
                 data_load_perc = train_profiling_data.get('percent_time_data_total_load', 0)
                 if data_load_perc > self.HIGH_DATA_LOAD_PERCENT_THRESHOLD:
                      recommendations.append(f"[Training Profiling] High time ({data_load_perc:.1f}%) spent in DataLoader. Consider increasing num_workers, optimizing transforms, or checking I/O performance.")
                 # Backward pass time check
                 backward_perc = train_profiling_data.get('percent_time_backward', 0)
                 if backward_perc > 60: # If > 60% is backward, might be expected but worth noting
                      recommendations.append(f"[Training Profiling] Backward pass takes a significant portion of time ({backward_perc:.1f}%). Expected for large models, but check profiler details for specific layer bottlenecks if step time is high.")
             else:
                  recommendations.append("[Training Profiling] Results suggest very fast or potentially zero step time; verify setup.")

             train_mem_mb = train_profiling_data.get("max_memory_allocated_mb", 0)
             if train_mem_mb and train_mem_mb > self.PROFILING_HIGH_MEM_MB_THRESHOLD:
                 recommendations.append(f"[Training Profiling] High peak memory usage ({train_mem_mb:.1f} MB) detected. Consider reducing batch size, gradient accumulation, mixed precision, or model optimization if VRAM is limited.")

             # Add profiler utilization if available
             prof_data = train_profiling_data.get('profiler_data', {})
             cpu_time_pct_train = prof_data.get('profiler_avg_cpu_time_percent', None)
             gpu_time_pct_train = prof_data.get('profiler_avg_gpu_time_percent', None)
             if cpu_time_pct_train is not None and gpu_time_pct_train is not None:
                  recommendations.append(f"[Training Profiling] Device Utilization: CPU {cpu_time_pct_train:.1f}%, GPU {gpu_time_pct_train:.1f}%.")
                  # Add checks similar to inference based on these percentages


        elif "error" in train_profiling_data:
             recommendations.append(f"[Training Profiling] Could not be completed: {train_profiling_data['error']}")

        # Gradient Analysis Results
        grad_analysis_data = report.get("gradient_analysis", {})
        if "error" not in grad_analysis_data and grad_analysis_data:
             if grad_analysis_data.get("warning"): # Check for warning like 'no gradients found'
                  recommendations.append(f"[Gradient Analysis] Warning: {grad_analysis_data['warning']}")

             num_nan = grad_analysis_data.get("num_params_nan_grad", 0)
             num_inf = grad_analysis_data.get("num_params_inf_grad", 0)
             if num_nan > 0 or num_inf > 0:
                 recommendations.append(f"[Gradient Analysis] CRITICAL: Found {num_nan} NaN and {num_inf} Inf gradients! Training likely unstable. Check learning rate, data, operations, or use AMP GradScaler correctly.")

             global_norm = grad_analysis_data.get("global_grad_norm_L2")
             if global_norm is not None:
                 if not (num_nan > 0 or num_inf > 0): # Only check thresholds if no NaN/Inf
                    if global_norm > self.HIGH_GRAD_NORM_THRESHOLD:
                        recommendations.append(f"[Gradient Analysis] High global gradient norm ({global_norm:.2e}) detected. Consider gradient clipping to prevent potential explosion.")
                    elif global_norm < self.LOW_GRAD_NORM_THRESHOLD:
                        recommendations.append(f"[Gradient Analysis] Low global gradient norm ({global_norm:.2e}) detected. Potential vanishing gradient issue. Check initialization, activation functions, normalization, or architecture.")
                 recommendations.append(f"[Gradient Analysis] Global Gradient Norm (L2): {global_norm:.3e}") # Always report norm

        elif "error" in grad_analysis_data:
             recommendations.append(f"[Gradient Analysis] Could not be completed: {grad_analysis_data['error']}")


        # System Diagnostics
        sys_diag_data = report.get("system_diagnostics", {})
        if "error" not in sys_diag_data and sys_diag_data:
            cpu_usage = sys_diag_data.get("cpu_usage_percent", 0)
            mem_usage = sys_diag_data.get("memory_usage_percent", 0)
            if cpu_usage > self.HIGH_CPU_USAGE_THRESHOLD:
                recommendations.append(f"[System] High overall CPU usage ({cpu_usage:.1f}%) detected system-wide. Check for other demanding processes or optimize CPU-bound tasks (like data loading if not offloaded).")
            if mem_usage > self.HIGH_MEM_USAGE_THRESHOLD_PERCENT:
                recommendations.append(f"[System] High overall system memory usage ({mem_usage:.1f}%) detected. Consider closing unused applications or increasing system RAM.")
        elif "error" in sys_diag_data:
            recommendations.append(f"[System Diagnostics] Failed: {sys_diag_data['error']}")


        # Final cleanup - remove duplicates and sort for consistency
        unique_recs = sorted(list(set(filter(None, recommendations))))
        return unique_recs