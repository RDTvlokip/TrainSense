# TrainSense/__init__.py

# --- Update Version ---
# Defines the current version of the TrainSense package.
# This is read by setup.py during the build process.
__version__ = "0.5.1" # Version updated for this release
# --------------------

# --- Core Analyzers & Monitors ---
# These classes form the core functionality for analysis and monitoring.
from .analyzer import TrainingAnalyzer         # Analyzes training hyperparameters based on system/model context.
from .arch_analyzer import ArchitectureAnalyzer # Analyzes the model's architecture (layers, params).
from .deep_analyzer import DeepAnalyzer         # Combines multiple analyzers for a comprehensive report.
from .gpu_monitor import GPUMonitor             # Monitors GPU status and usage (requires GPUtil).
from .model_profiler import ModelProfiler       # Profiles model inference and training steps.
from .optimizer import OptimizerHelper          # Provides helper functions and suggestions for optimizers/schedulers.
from .ultra_optimizer import UltraOptimizer     # Suggests initial hyperparameters based on heuristics.
from .system_config import SystemConfig         # Gathers static system configuration details.
from .system_diagnostics import SystemDiagnostics # Gathers dynamic system resource usage.
from .gradient_analyzer import GradientAnalyzer # Analyzes gradient statistics (NEW/Enhanced focus).

# --- NEW Integrations & Monitoring ---
# Classes and functions for integrating TrainSense with other libraries or providing real-time monitoring.
from .integrations import TrainStepMonitorHook # Example: Hook for step-by-step monitoring (if implemented).
from .integrations import TrainSenseTRLCallback # Example: Callback for Hugging Face TRL library (if implemented).
from .monitoring import RealTimeMonitor         # Provides real-time system/GPU monitoring in a separate thread.

# --- Utilities & Visualization ---
# Helper functions and plotting utilities.
from .logger import TrainLogger, get_trainsense_logger # Configurable logging setup.
from .visualizer import plot_training_step_breakdown, plot_gradient_histogram # Plotting functions (added gradient histogram).
from .utils import (                               # General utility functions.
    print_section,
    validate_positive_integer,
    validate_positive_float,
    format_bytes,
    format_time
)

# --- Optional: Define __all__ for explicit public API ---
# Lists the names that are considered part of the public API of the package.
# Helps tools like linters and IDEs, and controls `from TrainSense import *`.
__all__ = [
    # Core Classes
    "TrainingAnalyzer",
    "ArchitectureAnalyzer",
    "DeepAnalyzer",
    "GPUMonitor",
    "ModelProfiler",
    "OptimizerHelper",
    "UltraOptimizer",
    "SystemConfig",
    "SystemDiagnostics",
    "GradientAnalyzer",
    "RealTimeMonitor",

    # Integration Classes (Add uncommented ones as they become stable)
    "TrainStepMonitorHook",
    "TrainSenseTRLCallback",

    # Logging
    "TrainLogger",
    "get_trainsense_logger",

    # Visualization Functions
    "plot_training_step_breakdown",
    "plot_gradient_histogram", # New plotting function

    # Utility Functions
    "print_section",
    "validate_positive_integer",
    "validate_positive_float",
    "format_bytes",
    "format_time",

    # Package Version
    "__version__"
]