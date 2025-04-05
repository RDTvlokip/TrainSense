# TrainSense/deep_analyzer.py
import logging
from typing import Dict, Any, List

from .analyzer import TrainingAnalyzer
from .arch_analyzer import ArchitectureAnalyzer
from .model_profiler import ModelProfiler
from .system_diagnostics import SystemDiagnostics
from .optimizer import OptimizerHelper
from .utils import format_bytes

logger = logging.getLogger(__name__)

class DeepAnalyzer:
    HIGH_CPU_USAGE_THRESHOLD = 85.0
    HIGH_MEM_USAGE_THRESHOLD_PERCENT = 85.0
    PROFILING_HIGH_MEM_MB_THRESHOLD = 2 * 1024 # 2 GB

    def __init__(self,
                 training_analyzer: TrainingAnalyzer,
                 arch_analyzer: ArchitectureAnalyzer,
                 model_profiler: ModelProfiler,
                 system_diag: SystemDiagnostics):
        self.training_analyzer = training_analyzer
        self.arch_analyzer = arch_analyzer
        self.model_profiler = model_profiler
        self.system_diag = system_diag
        logger.info("DeepAnalyzer initialized.")

    def comprehensive_report(self, profile_input_shape: Any = None, profile_iterations: int = 100) -> Dict[str, Any]:
        logger.info("Generating comprehensive report.")
        report: Dict[str, Any] = {}

        report["hyperparameter_analysis"] = {
            "current_values": {
                "batch_size": self.training_analyzer.batch_size,
                "learning_rate": self.training_analyzer.learning_rate,
                "epochs": self.training_analyzer.epochs,
            },
            "recommendations": self.training_analyzer.check_hyperparameters(),
            "suggested_adjustments": self.training_analyzer.auto_adjust()
        }

        logger.info("Running architecture analysis.")
        arch_info = self.arch_analyzer.analyze()
        report["architecture_analysis"] = arch_info

        logger.info("Running model profiling.")
        try:
            # Use estimated shape from arch_analyzer if not provided, else raise error if still None
            if profile_input_shape is None:
                 profile_input_shape = arch_info.get("estimated_input_shape")
                 if profile_input_shape is None:
                      logger.warning("Cannot determine input shape for profiling. Profiling skipped.")
                      report["model_profiling"] = {"error": "Input shape not provided and could not be estimated."}
                 else:
                      logger.info(f"Using estimated input shape for profiling: {profile_input_shape}")

            if profile_input_shape: # Proceed only if shape is determined
                 profile_results = self.model_profiler.profile_model(
                     input_shape=profile_input_shape,
                     iterations=profile_iterations
                 )
                 report["model_profiling"] = profile_results
            else:
                 # Handles the case where shape was None and warning was logged.
                 # Redundant assignment but clarifies the structure.
                 if "model_profiling" not in report:
                    report["model_profiling"] = {"error": "Profiling skipped due to missing input shape."}

        except Exception as e:
            logger.error(f"Model profiling failed: {e}", exc_info=True)
            report["model_profiling"] = {"error": f"Profiling failed: {str(e)}"}

        logger.info("Running system diagnostics.")
        try:
            sys_info = self.system_diag.diagnostics()
            report["system_diagnostics"] = sys_info
        except Exception as e:
             logger.error(f"System diagnostics failed: {e}", exc_info=True)
             report["system_diagnostics"] = {"error": f"Diagnostics failed: {str(e)}"}

        logger.info("Aggregating recommendations.")
        report["overall_recommendations"] = self._aggregate_recommendations(report)

        logger.info("Comprehensive report generation complete.")
        return report

    def _aggregate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        recommendations = []

        # Hyperparameters
        if report.get("hyperparameter_analysis"):
            recommendations.extend(report["hyperparameter_analysis"].get("recommendations", []))

        # Architecture
        if report.get("architecture_analysis"):
            recommendations.append(report["architecture_analysis"].get("recommendation", "No specific architecture recommendation available."))

        # Profiling
        profiling_data = report.get("model_profiling", {})
        if "error" not in profiling_data and profiling_data:
            mem_usage_mb = profiling_data.get("max_memory_allocated_mb", 0)
            if mem_usage_mb > self.PROFILING_HIGH_MEM_MB_THRESHOLD:
                recommendations.append(f"High peak memory usage ({mem_usage_mb:.1f} MB) detected during profiling. Consider model optimization, gradient accumulation, or mixed precision if VRAM is limited.")

            avg_time = profiling_data.get('avg_total_time_ms', 0)
            throughput = profiling_data.get('throughput_samples_per_sec', 0)
            if avg_time > 0:
                 recommendations.append(f"Profiling results: Avg Inference Time: {avg_time:.2f} ms | Throughput: {throughput:.2f} samples/sec.")
            else:
                 recommendations.append("Profiling results suggest very fast or potentially zero execution time; verify profiling setup.")

            cpu_time_pct = profiling_data.get('avg_cpu_time_percent', None)
            gpu_time_pct = profiling_data.get('avg_gpu_time_percent', None)
            if cpu_time_pct is not None and gpu_time_pct is not None:
                 recommendations.append(f"Device Utilization (Profiling): CPU {cpu_time_pct:.1f}%, GPU {gpu_time_pct:.1f}%.")
                 if gpu_time_pct < 50 and cpu_time_pct < 50 and avg_time > 5: # Avoid flagging tiny models
                     recommendations.append("Low CPU and GPU utilization during profiling might indicate I/O bottlenecks or inefficient data loading/preprocessing.")
                 elif gpu_time_pct < 75 and cpu_time_pct > 50 and self.training_analyzer.system_config and self.training_analyzer.system_config.gpu_info:
                      recommendations.append("Relatively low GPU utilization with higher CPU usage might suggest the CPU is a bottleneck (e.g., data processing).")

        elif "error" in profiling_data:
             recommendations.append(f"Model profiling could not be completed: {profiling_data['error']}")


        # System Diagnostics
        sys_diag_data = report.get("system_diagnostics", {})
        if "error" not in sys_diag_data and sys_diag_data:
            cpu_usage = sys_diag_data.get("cpu_usage_percent", 0)
            mem_usage = sys_diag_data.get("memory_usage_percent", 0)

            if cpu_usage > self.HIGH_CPU_USAGE_THRESHOLD:
                recommendations.append(f"High overall CPU usage ({cpu_usage:.1f}%) detected system-wide. Check for other demanding processes or optimize data loading/preprocessing.")
            if mem_usage > self.HIGH_MEM_USAGE_THRESHOLD_PERCENT:
                recommendations.append(f"High overall system memory usage ({mem_usage:.1f}%) detected. Consider closing unused applications or increasing system RAM.")
        elif "error" in sys_diag_data:
            recommendations.append(f"System diagnostics failed: {sys_diag_data['error']}")


        # Cross-Component Checks
        arch_info = report.get("architecture_analysis", {})
        total_params = arch_info.get("total_parameters", 0)
        optimizer_rec = OptimizerHelper.suggest_optimizer(total_params, arch_info.get("layer_count", 0), arch_info.get("primary_architecture_type", "Unknown"))
        recommendations.append(f"Based on model size/type, suggested optimizer: {optimizer_rec}.")


        # Remove duplicates and filter empty strings
        unique_recs = sorted(list(set(filter(None, recommendations))))

        return unique_recs