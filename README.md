# TrainSense v0.5.0: Analyze, Profile, Diagnose, and Optimize your PyTorch Training

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Version](https://img.shields.io/badge/pytorch-1.8+-red.svg)](https://pytorch.org/)
<!-- Add PyPI version badge once published: [![PyPI version](https://badge.fury.io/py/trainsense.svg)](https://badge.fury.io/py/trainsense) -->
<!-- Add other badges for tests, coverage etc. as CI/CD is set up -->

**TrainSense is a Python toolkit designed to provide deep insights into your PyTorch model training environment and performance.** It empowers you to understand your system, analyze your model architecture, evaluate hyperparameters, profile performance bottlenecks (including full training steps **and** real-time resource usage), and crucially, **diagnose gradient health**. Ultimately, TrainSense helps you optimize your entire deep learning workflow.

Whether you're struggling with slow training, mysterious NaN losses, vanishing/exploding gradients, inefficient GPU utilization, or simply want a clearer picture of your training dynamics, TrainSense v0.5.0 offers a powerful and integrated suite of tools.

**(Link to GitHub Repository: [https://github.com/RDTvlokip/TrainSense](https://github.com/RDTvlokip/TrainSense))**

---

## Table of Contents

*   [Key Features (v0.5.0)](#key-features-v050)
*   [What's New in v0.5.0](#whats-new-in-v050)
*   [Installation](#installation)
*   [Core Concepts](#core-concepts)
*   [Getting Started: Quick Example](#getting-started-quick-example)
*   [Detailed Usage Examples](#detailed-usage-examples)
    *   [1. System Configuration (`SystemConfig`)](#1-checking-system-configuration)
    *   [2. Architecture Analysis (`ArchitectureAnalyzer`)](#2-analyzing-your-models-architecture)
    *   [3. Hyperparameter Recommendations (`TrainingAnalyzer`)](#3-getting-hyperparameter-recommendations)
    *   [4. Inference Performance Profiling (`ModelProfiler.profile_model`)](#4-profiling-model-inference-performance)
    *   [5. Training Step Profiling (`ModelProfiler.profile_training_step`)](#5-profiling-a-full-training-step)
    *   [6. **Gradient Analysis (`GradientAnalyzer`)**](#6-analyzing-gradients-new--enhanced)
    *   [7. GPU Monitoring (`GPUMonitor`)](#7-monitoring-gpu-status)
    *   [8. Optimizer & Scheduler Suggestions (`OptimizerHelper`)](#8-getting-optimizer-and-scheduler-suggestions)
    *   [9. Heuristic Hyperparameters (`UltraOptimizer`)](#9-generating-heuristic-hyperparameters-ultraoptimizer)
    *   [10. **Comprehensive Reporting (`DeepAnalyzer`)**](#10-using-the-comprehensive-reporter-deepanalyzer)
    *   [11. Plotting Training Breakdown (`visualizer`)](#11-plotting-training-breakdown-optional)
    *   [12. **Plotting Gradient Histogram (`GradientAnalyzer`)**](#12-plotting-gradient-histogram-optional)
    *   [13. **Real-Time Monitoring (`RealTimeMonitor`)**](#13-real-time-monitoring)
    *   [14. Logging (`TrainLogger`)](#14-using-the-logger)
*   [Interpreting the Output](#interpreting-the-output)
*   [Contributing](#contributing)
*   [License](#license)

---

## Key Features (v0.5.0)

*   **System Analysis:** `SystemConfig` (Detect Hardware/Software), `SystemDiagnostics` (Check current usage).
*   **Model Architecture Insight:** `ArchitectureAnalyzer` (Parameters, Layers, Inferred Type, Complexity, Recommendations).
*   **Hyperparameter Sanity Checks:** `TrainingAnalyzer` (Contextual checks for Batch Size, LR, Epochs & heuristic suggestions).
*   **Advanced Performance Profiling:** `ModelProfiler` (Profiles Inference & **Full Training Steps** with `torch.profiler` support, data loading breakdown).
*   ✨ **Deep Gradient Diagnostics (New!):** `GradientAnalyzer` meticulously analyzes gradients *after* `backward()`:
    *   Calculates per-parameter stats (Norm, Mean, Std, Min, Max).
    *   **Detects NaN/Inf gradients** per parameter.
    *   Computes **global gradient norm** (L2 default).
    *   Provides aggregated summary statistics.
    *   Optionally plots gradient norm distribution histograms (`matplotlib` required).
*   **GPU Monitoring:** `GPUMonitor` (Real-time Load, Memory Usage/Util, Temp via `GPUtil`).
*   **Training Optimization Guidance:** `OptimizerHelper` (Suggests Optimizers/Schedulers based on context), `UltraOptimizer` (Generates heuristic starting parameters).
*   ✨ **Comprehensive & Integrated Reporting:** `DeepAnalyzer` orchestrates analyses (including **gradient analysis results**) into a detailed dictionary report. Aggregated recommendations now factor in gradient health signals.
*   **Visualization (Optional):** Functions to plot Training Step Time Breakdown and Gradient Norm Histograms (requires `matplotlib`).
*   ✨ **Real-Time Resource Monitoring (New!):** `RealTimeMonitor` class runs in a background thread to track system (CPU/RAM) and GPU usage **during** specific code sections (like training loops).

## What's New in v0.5.0

Version 0.5.0 marks a significant enhancement in diagnostic capabilities, focusing on the critical aspects of gradient health and real-time resource usage.

*   ✨ **`GradientAnalyzer` Module (Major Feature):** Introduces dedicated, in-depth analysis of model gradients after the backward pass. Key for debugging unstable training (NaNs, explosion, vanishing).
*   ✨ **`RealTimeMonitor` Module (Major Feature):** Provides the ability to monitor CPU, RAM, and GPU utilization dynamically in a separate thread *during* training or other long-running operations.
*   **Enhanced `DeepAnalyzer`:** Now incorporates results from `GradientAnalyzer` into its comprehensive report and `overall_recommendations`, providing more insightful diagnostics (e.g., warning about high/low/NaN gradient norms).
*   **Gradient Histogram Plotting:** `GradientAnalyzer` includes a `plot_gradient_norm_histogram` method (requires `matplotlib`) for visualizing the distribution of gradient magnitudes across layers.
*   **Improved Architecture/Optimizer Suggestions:** Recommendations from `TrainingAnalyzer` and `OptimizerHelper` now better consider the inferred model architecture type.
*   **Code Quality & Robustness:** Added extensive English comments across core modules, improved error handling, and refined logging.

## Installation

Using a virtual environment is highly recommended.

1.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Linux/macOS: source venv/bin/activate
    # On Windows: venv\Scripts\activate
    ```

2.  **Install PyTorch:** Follow official instructions for your system/CUDA version: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) (TrainSense v0.5.0 requires >= 1.8.0)

3.  **Install TrainSense v0.5.0:**
    *(From local source - replace with `pip install trainsense==0.5.0` if published)*

    *   **Core Installation:**
        ```bash
        pip install .
        # Or for development (editable install):
        # pip install -e .
        ```
    *   **With Optional Plotting:** (Needed for gradient histograms & breakdown plots)
        ```bash
        pip install .[plotting]
        # Or: pip install -e .[plotting]
        ```
    *   **With All Optional Features (excluding Dev):**
        ```bash
        pip install .[all]
        # Or: pip install -e .[all]
        ```
        *(Note: 'all' in v0.5.0 typically includes plotting, html, trl)*
    *   **For Development (includes all features + test tools):**
        ```bash
        pip install -e .[dev]
        ```
    Core dependencies (`psutil`, `torch`, `GPUtil`) are installed automatically. Optional ones (`matplotlib`, `numpy`, `jinja2`, `transformers`) are managed via these extras.

## Core Concepts

TrainSense v0.5.0 provides insights across several stages:

1.  **Environment Setup (`SystemConfig`, `SystemDiagnostics`, `GPUMonitor`):** Understand your hardware and current system load.
2.  **Model Structure (`ArchitectureAnalyzer`):** Analyze your model's complexity, layers, and type.
3.  **Training Plan (`TrainingAnalyzer`, `OptimizerHelper`, `UltraOptimizer`):** Get recommendations on your chosen hyperparameters and sensible starting points.
4.  **Execution Performance (`ModelProfiler`):** Profile inference speed and detailed training step timings, identifying bottlenecks.
5.  **Learning Dynamics & Stability (`GradientAnalyzer`):** **Crucially, analyze the health of gradients after backpropagation.**
6.  **Live Resource Tracking (`RealTimeMonitor`):** Observe how resources are used *during* key operations.
7.  **Synthesis & Reporting (`DeepAnalyzer`, `visualizer`):** Combine all insights into a comprehensive dictionary report with actionable recommendations.

## Getting Started: Quick Example

This example demonstrates initializing key components and generating a report including gradient analysis.

```python
import torch
import torch.nn as nn
import logging
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("TrainSenseExample")

# --- Import TrainSense ---
try:
    from TrainSense import (SystemConfig, ArchitectureAnalyzer, ModelProfiler,
                          DeepAnalyzer, TrainingAnalyzer, SystemDiagnostics,
                          GradientAnalyzer, # <-- New in 0.5.0
                          print_section, get_trainsense_logger)
    from TrainSense.gradient_analyzer import MATPLOTLIB_AVAILABLE as PLOTTING_AVAILABLE
    # logger = get_trainsense_logger() # Optionally use TrainSense logger
except ImportError as e:
    logger.error(f"Failed to import TrainSense: {e}")
    sys.exit(1)

# --- Define Model & Setup ---
model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 5))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
batch_size, lr, epochs = 16, 1e-3, 3 # Short run
input_shape = (batch_size, 64)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = Adam(model.parameters(), lr=lr)
# Create minimal dummy data and loader
dummy_X = torch.randn(batch_size * 2, 64); dummy_y = torch.randint(0, 5, (batch_size * 2,), dtype=torch.long)
dummy_loader = DataLoader(TensorDataset(dummy_X, dummy_y), batch_size=batch_size)
logger.info(f"Using device: {device}")

# --- Instantiate TrainSense ---
try:
    logger.info("Initializing TrainSense components...")
    sys_config = SystemConfig()
    sys_diag = SystemDiagnostics()
    arch_analyzer = ArchitectureAnalyzer(model)
    arch_info = arch_analyzer.analyze() # Get arch info early
    model_profiler = ModelProfiler(model, device=device)
    training_analyzer = TrainingAnalyzer(batch_size, lr, epochs, system_config=sys_config, arch_info=arch_info)
    grad_analyzer = GradientAnalyzer(model) # Initialize GradientAnalyzer
    deep_analyzer = DeepAnalyzer(training_analyzer, arch_analyzer, model_profiler, sys_diag, grad_analyzer)
    logger.info("TrainSense Initialized.")

    # --- Run a Training Step & Backward Pass (Required for Gradient Analysis) ---
    print_section("Setup: Running One Training Step + Backward Pass")
    model.train()
    optimizer.zero_grad()
    try:
        inputs, targets = next(iter(dummy_loader))
        outputs = model(inputs.to(device))
        loss = criterion(outputs, targets.to(device))
        loss.backward() # <<-- Populate .grad attributes
        optimizer.step() # Optional, but realistic
        logger.info(f"Backward pass complete (Loss: {loss.item():.4f}). Gradients available.")
        GRADIENTS_AVAILABLE = True
    except Exception as e:
         logger.error(f"Failed to run training step: {e}", exc_info=True)
         GRADIENTS_AVAILABLE = False
    model.eval()

    # --- Generate Comprehensive Report ---
    # Enable gradient_analysis if the backward pass succeeded
    print_section("Running Comprehensive Analysis")
    report = deep_analyzer.comprehensive_report(
        profile_inference=True,          # Include inference profile
        profile_training=False,          # Skip detailed training profile for brevity here
        gradient_analysis=GRADIENTS_AVAILABLE, # <<-- Enable gradient analysis
        inference_input_shape=(1, 64)  # Use BS=1 for inference profile
    )
    logger.info("Comprehensive Analysis Complete.")

    # --- Display Key Findings from Report ---
    print("\n>>> Overall Recommendations:")
    recommendations = report.get("overall_recommendations", ["N/A - Check Logs"])
    if recommendations:
        for i, rec in enumerate(recommendations): print(f"  [{i+1}] {rec}")

    # Display Gradient Summary if analysis ran
    grad_analysis_summary = report.get("gradient_analysis", {})
    if not grad_analysis_summary.get("error") and "global_grad_norm_L2" in grad_analysis_summary:
         print("\n>>> Gradient Analysis Summary:")
         norm_l2 = grad_analysis_summary.get('global_grad_norm_L2')
         norm_str = f"{norm_l2:.3e}" if norm_l2 is not None else "N/A"
         print(f"  Global Norm L2: {norm_str}")
         print(f"  NaN/Inf Grads Found: {grad_analysis_summary.get('num_params_nan_grad', 0)} / {grad_analysis_summary.get('num_params_inf_grad', 0)}")
         # Optionally plot histogram if available
         if PLOTTING_AVAILABLE and GRADIENTS_AVAILABLE:
             plot_path = "quick_start_grad_histogram.png"
             if grad_analyzer.plot_gradient_norm_histogram(save_path=plot_path, show_plot=False):
                 print(f"  Gradient histogram saved to: {plot_path}")

except ImportError:
    logger.error("TrainSense import failed. Please check installation.")
except Exception as e:
    logger.exception("Error during TrainSense quick start example") # Log full traceback
    print(f"\nERROR encountered: {e}")

```

## Detailed Usage Examples

*(Examples for `SystemConfig`, `ArchitectureAnalyzer`, `TrainingAnalyzer`, `ModelProfiler`, `GPUMonitor`, `OptimizerHelper`, `UltraOptimizer`, `Plotting` remain largely the same as v0.4.0 conceptually, focusing on individual component usage. Key updates are shown below)*

### 6. Analyzing Gradients (New & Enhanced!)

This is a core feature of v0.5.0. **Requires running `model.backward()` *before* analysis.**

```python
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from TrainSense import GradientAnalyzer, print_section

# --- Setup Model, Data, Optimizer ---
model = nn.Linear(64, 10).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = Adam(model.parameters())
dummy_X = torch.randn(32, 64); dummy_y = torch.randint(0, 10, (32,));
loader = DataLoader(TensorDataset(dummy_X, dummy_y), batch_size=32)

# --- Instantiate Analyzer ---
grad_analyzer = GradientAnalyzer(model)
print_section("Gradient Analysis")

# --- Run Backward Pass ---
model.train()
optimizer.zero_grad()
inputs, targets = next(iter(loader))
outputs = model(inputs.to(device))
loss = criterion(outputs, targets.to(device))
loss.backward() # <<< IMPORTANT: Populate gradients
model.eval()
print("Ran backward pass.")

# --- Get Gradient Summary ---
summary = grad_analyzer.summary() # Calculates global norm, aggregates stats
print("\n--- Gradient Summary ---")
print(f"Global Gradient Norm (L2): {summary.get('global_grad_norm_L2', 'N/A'):.3e}")
print(f"Params with Grads: {summary.get('num_params_with_grads', 'N/A')}")
print(f"NaN Gradients: {summary.get('num_params_nan_grad', 'N/A')}")
print(f"Inf Gradients: {summary.get('num_params_inf_grad', 'N/A')}")
print(f"Avg Grad Norm (L2): {summary.get('avg_grad_norm', 'N/A'):.3e}")
print(f"Max Grad Norm Layer: {summary.get('layer_with_max_grad_norm', 'N/A')}")

# --- Get Detailed Per-Parameter Stats (Optional) ---
# detailed_stats = grad_analyzer.analyze_gradients()
# print("\n--- Detailed Stats (First Param Example) ---")
# first_param_name = list(detailed_stats.keys())[0]
# print(f"{first_param_name}: {detailed_stats[first_param_name]}")

# --- Plot Gradient Histogram (Optional) ---
# Requires matplotlib: pip install trainsense[plotting]
try:
    from TrainSense.gradient_analyzer import MATPLOTLIB_AVAILABLE
    if MATPLOTLIB_AVAILABLE:
        plot_path = "gradient_norm_histogram.png"
        success = grad_analyzer.plot_gradient_norm_histogram(save_path=plot_path, show_plot=False)
        if success:
            print(f"\nGradient histogram saved to: {plot_path}")
        else:
            print("\nFailed to generate gradient histogram (check logs/gradient status).")
    else:
        print("\nGradient histogram plotting skipped: matplotlib/numpy not available.")
except ImportError:
     print("\nGradient histogram plotting skipped: matplotlib/numpy not available.")

```

### 10. Using the Comprehensive Reporter (`DeepAnalyzer`)

`DeepAnalyzer` now leverages `GradientAnalyzer` results for better recommendations.

```python
from TrainSense import DeepAnalyzer # ... plus other components ...
# (Assume all components initialized: training_analyzer, arch_analyzer, model_profiler, sys_diag, grad_analyzer)
# (Assume backward pass run if gradient_analysis=True)

deep_analyzer = DeepAnalyzer(training_analyzer, arch_analyzer, model_profiler, sys_diag, grad_analyzer)
print_section("Comprehensive Report (with Gradient Analysis)")

# --- Run backward pass if analyzing gradients ---
# ... (code similar to Gradient Analysis example) ...
# ---------------------------------------------

report = deep_analyzer.comprehensive_report(
    profile_inference=True,
    profile_training=False, # Set to True if you need training step profile
    gradient_analysis=True, # <<< Enable gradient analysis
    inference_input_shape=(1, 64) # Provide shape if profiling inference
    # Pass loader/criterion/optimizer if profile_training=True
)

print("Report generated (dictionary). Access keys like 'gradient_analysis'.")
print("\n--- Overall Recommendations ---")
# Recommendations now consider gradient health!
for i, rec in enumerate(report.get("overall_recommendations", ["N/A"])):
    print(f"  [{i+1}] {rec}")

# Access gradient summary directly from report
grad_summary = report.get("gradient_analysis", {})
print("\n--- Gradient Summary from Report ---")
print(f"Global Norm L2: {grad_summary.get('global_grad_norm_L2', 'N/A'):.2e}, NaN/Inf Grads: {grad_summary.get('num_params_nan_grad', 0)}/{grad_summary.get('num_params_inf_grad', 0)}")

# Note: v0.5.0 does NOT have the save_html_path argument. HTML reports were added later.
```

### 11. Plotting Training Breakdown (Optional)
*(Usage unchanged, depends on `ModelProfiler` results)*
```python
from TrainSense.visualizer import plot_training_step_breakdown
# (Assume 'train_profile_results' dictionary exists from ModelProfiler.profile_training_step)
if train_profile_results and "error" not in train_profile_results:
    plot_training_step_breakdown(train_profile_results, save_path="training_breakdown.png", show_plot=False)
```

### 12. Plotting Gradient Histogram (Optional)
*(Now a method of `GradientAnalyzer`, see Example #6)*

### 13. Real-Time Monitoring (New!)

Monitor resource usage during a specific code block (like a training loop).

```python
from TrainSense import RealTimeMonitor
import time

# Initialize the monitor (checks e.g., every 1 second)
monitor = RealTimeMonitor(interval_sec=1.0, monitor_gpu=True) # Set monitor_gpu=False if no GPU/GPUtil

print("Starting monitored section...")
# Use as a context manager (automatically starts/stops)
with monitor:
    # Simulate some work (e.g., your training loop)
    print("  Work started...")
    time.sleep(3.5) # Simulate work for ~3.5 seconds
    print("  Work finished.")
print("Monitored section complete.")

# Get the collected history
history = monitor.get_history()
print(f"\nCollected {len(history)} monitoring snapshots:")
for i, snapshot in enumerate(history[-3:]): # Print last 3 snapshots
    print(f"  Snapshot {i+1}:")
    print(f"    Timestamp: {snapshot.get('timestamp')}")
    print(f"    CPU Usage %: {snapshot.get('cpu_usage_percent'):.1f}")
    print(f"    Memory Usage %: {snapshot.get('memory_usage_percent'):.1f}")
    gpu_status = snapshot.get('gpu_status')
    if isinstance(gpu_status, list) and gpu_status: # Check if list and not empty
        print(f"    GPU Avg Load %: {sum(g.get('load', 0) for g in gpu_status) / len(gpu_status):.1f}")
        print(f"    GPU Avg Mem Util %: {sum(g.get('memory_utilization_percent', 0) for g in gpu_status) / len(gpu_status):.1f}")
    elif isinstance(gpu_status, str):
        print(f"    GPU Status: {gpu_status}")

# You can now process/save the 'history' list (list of dictionaries)
```

### 14. Using the Logger
*(Usage unchanged)*
```python
from TrainSense import get_trainsense_logger
# Get the configured logger (initializes with defaults if first time)
logger = get_trainsense_logger()
logger.info("This is an info message from TrainSense logger.")
logger.warning("This is a warning message.")
```

## Interpreting the Output

*   **Comprehensive Report Dictionary:** Examine the `overall_recommendations` list first. Then, dive into specific sections like `gradient_analysis`, `training_step_profiling`, `inference_profiling` based on the recommendations or your area of concern. Look for `error` keys within sections.
*   **Gradient Analysis:**
    *   **High `Global Grad Norm` (> 1e2, 1e3+):** Risk of exploding gradients. Training might become unstable (loss increases). Consider gradient clipping.
    *   **Low `Global Grad Norm` (< 1e-6, 1e-7):** Risk of vanishing gradients. Learning slows down or stops. Check activations (ReLU preferred over sigmoid/tanh in deep nets), initialization, normalization layers.
    *   **`NaN/Inf Grads Found > 0`:** **Critical!** Stop training immediately. Debug data loading (check for NaNs in input/targets), check for numerically unstable operations (e.g., log(0), sqrt(<0), large exponents), reduce learning rate, ensure correct mixed precision usage (use `GradScaler`).
    *   **Histogram:** A healthy distribution often looks somewhat bell-shaped or log-normal. A distribution heavily skewed towards zero indicates potential vanishing, while very large outliers suggest explosion.
*   **Training Step Profiling:** High `% Data Load` suggests I/O or preprocessing bottlenecks. High `% Backward Pass` is often normal but check specific ops if step time is excessive.
*   **Real-Time Monitoring:** Look for sustained high CPU/RAM usage, or periods of low GPU utilization during expected heavy computation (might indicate CPU/IO bottlenecks).

## Contributing

Contributions are welcome! Please refer to the contribution guidelines (link to be added) or open an issue/pull request on the [GitHub repository](https://github.com/RDTvlokip/TrainSense).

## License

This project is licensed under the MIT License. See the LICENSE file for details.
