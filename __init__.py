# TrainSense/__init__.py

# --- Update Version ---
__version__ = "0.4.0" # Version reflects these changes
# --------------------

# Core Analyzers & Monitors
from .analyzer import TrainingAnalyzer
from .arch_analyzer import ArchitectureAnalyzer
from .deep_analyzer import DeepAnalyzer
from .gpu_monitor import GPUMonitor
from .model_profiler import ModelProfiler
from .optimizer import OptimizerHelper
from .ultra_optimizer import UltraOptimizer
from .system_config import SystemConfig
from .system_diagnostics import SystemDiagnostics
from .gradient_analyzer import GradientAnalyzer

# NEW Integrations & Monitoring
from .integrations import TrainStepMonitorHook, TrainSenseTRLCallback # Add new integration classes
from .monitoring import RealTimeMonitor # Add new monitoring class

# Utilities & Visualization
from .logger import TrainLogger, get_trainsense_logger
from .visualizer import plot_training_step_breakdown, plot_gradient_histogram # Add new plot func
from .utils import print_section, validate_positive_integer, validate_positive_float, format_bytes, format_time

# Optional: Define __all__ for explicit public API
__all__ = [
    "TrainingAnalyzer", "ArchitectureAnalyzer", "DeepAnalyzer", "GPUMonitor",
    "ModelProfiler", "OptimizerHelper", "UltraOptimizer", "SystemConfig",
    "SystemDiagnostics", "GradientAnalyzer", "RealTimeMonitor",
    "TrainStepMonitorHook", "TrainSenseTRLCallback", # If ready for public use
    "TrainLogger", "get_trainsense_logger",
    "plot_training_step_breakdown", "plot_gradient_histogram",
    "print_section", "validate_positive_integer", "validate_positive_float",
    "format_bytes", "format_time",
    "__version__"
]