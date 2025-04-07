# TrainSense v0.3.0: Analyze, Profile, and Optimize your PyTorch Training Workflow

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- Badges for CI/CD, PyPI version, etc., can be added later -->

TrainSense is a Python toolkit designed to provide deep insights into your PyTorch model training environment and performance. It helps you understand your system's capabilities, analyze your model's architecture, evaluate hyperparameter choices, profile execution bottlenecks (including full training steps), diagnose gradient issues, and ultimately optimize your deep learning workflow.

Whether you're debugging slow training, trying to maximize GPU utilization, investigating vanishing/exploding gradients, or simply want a clearer picture of your setup, TrainSense offers a suite of tools to assist you.

**(GitHub Repo link coming very soon!)**

---

## Table of Contents

*   [Key Features](#key-features)
*   [What's New in v0.3.0](#whats-new-in-v030)
*   [Installation](#installation)
*   [Core Concepts](#core-concepts)
*   [Getting Started: Quick Example](#getting-started-quick-example)
*   [Detailed Usage Examples](#detailed-usage-examples)
    *   [1. System Configuration (`SystemConfig`)](#1-checking-system-configuration)
    *   [2. Architecture Analysis (`ArchitectureAnalyzer`)](#2-analyzing-your-models-architecture)
    *   [3. Hyperparameter Recommendations (`TrainingAnalyzer`)](#3-getting-hyperparameter-recommendations)
    *   [4. Inference Performance Profiling (`ModelProfiler.profile_model`)](#4-profiling-model-inference-performance)
    *   [5. Training Step Profiling (`ModelProfiler.profile_training_step`)](#5-profiling-a-full-training-step-new-in-v030)
    *   [6. Gradient Analysis (`GradientAnalyzer`)](#6-analyzing-gradients-new-in-v030)
    *   [7. GPU Monitoring (`GPUMonitor`)](#7-monitoring-gpu-status)
    *   [8. Optimizer & Scheduler Suggestions (`OptimizerHelper`)](#8-getting-optimizer-and-scheduler-suggestions)
    *   [9. Heuristic Hyperparameters (`UltraOptimizer`)](#9-generating-heuristic-hyperparameters-ultraoptimizer)
    *   [10. Comprehensive Reporting (`DeepAnalyzer`)](#10-using-the-comprehensive-reporter-deepanalyzer)
    *   [11. Plotting Training Breakdown (`visualizer`)](#11-plotting-training-breakdown-optional-new-in-v030)
    *   [12. Logging (`TrainLogger`)](#12-using-the-logger)
*   [Interpreting the Output](#interpreting-the-output)
*   [Contributing](#contributing)
*   [License](#license)

---

## Key Features

*   **System Analysis:**
    *   **`SystemConfig`:** Detects hardware (CPU, RAM, GPU), OS, Python, PyTorch, CUDA, and cuDNN versions.
    *   **`SystemDiagnostics`:** Monitors real-time system resource usage (CPU, Memory, Disk, Network).
*   **Model Architecture Insight:**
    *   **`ArchitectureAnalyzer`:** Counts parameters (total/trainable), layers, analyzes layer types, estimates input shape, infers architecture type (CNN, RNN, Transformer...), and provides complexity assessment and recommendations.
*   **Hyperparameter Sanity Checks:**
    *   **`TrainingAnalyzer`:** Evaluates batch size, learning rate, and epochs based on system resources and model complexity. Provides recommendations and suggests automatic adjustments.
*   **Advanced Performance Profiling:**
    *   **`ModelProfiler`:**
        *   Measures **inference speed** (latency, throughput).
        *   **(New!)** Profiles a **full training step** (data loading/prep, forward, loss, backward, optimizer step) to identify bottlenecks specific to training.
        *   Integrates `torch.profiler` for detailed operator-level CPU/GPU time and memory usage analysis.
*   **Gradient Diagnostics (New!):**
    *   **`GradientAnalyzer`:** Inspects gradient statistics (norms, mean, std, NaN/Inf counts) per parameter after a backward pass to help diagnose vanishing/exploding gradients or other training stability issues.
*   **GPU Monitoring:**
    *   **`GPUMonitor`:** Provides real-time, detailed GPU status including load, memory utilization (used, total), and temperature (requires `GPUtil`).
*   **Training Optimization Guidance:**
    *   **`OptimizerHelper`:** Suggests suitable optimizers (Adam, AdamW, SGD) and learning rate schedulers based on model characteristics. Recommends initial learning rates.
    *   **`UltraOptimizer`:** Generates a full set of heuristic hyperparameters (batch size, LR, epochs, optimizer, scheduler) as a starting point, based on system, model, and basic data stats.
*   **Consolidated Reporting:**
    *   **`DeepAnalyzer`:** Orchestrates analysis modules to generate a comprehensive report with aggregated insights and recommendations. Can optionally include results from training step profiling and gradient analysis.
*   **Visualization (New!):**
    *   **`visualizer.plot_training_step_breakdown`:** Generates a bar chart showing the time distribution across different phases of the training step (requires `matplotlib`).
*   **Flexible Logging:**
    *   **`TrainLogger`:** Configurable logging to console and rotating files.

## What's New in v0.3.0

*   🚀 **Training Step Profiling (`ModelProfiler.profile_training_step`):** Go beyond inference! Profile the entire forward-backward-optimizer sequence, including detailed data loading breakdown (fetch vs. prep), to understand where time is *really* spent during training.
*   🩺 **Gradient Analysis (`GradientAnalyzer`):** A new dedicated tool to check the health of your gradients after `loss.backward()`. Calculate norms, check for NaN/Inf values, and get summaries to quickly spot potential training instabilities like vanishing or exploding gradients.
*   📊 **Basic Visualization (`visualizer.plot_training_step_breakdown`):** Optionally generate a bar chart visualizing the training step time breakdown (requires `matplotlib`).
*   ⚙️ **Integrated Reporting (`DeepAnalyzer`):** The comprehensive report can now optionally trigger and include results from training step profiling and gradient analysis for a more holistic view.

## Installation

It's highly recommended to use a virtual environment.

1.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Linux/macOS
    source venv/bin/activate
    # On Windows
    # venv\Scripts\activate
    ```

2.  **Install PyTorch:** TrainSense depends on PyTorch. Install the version suitable for your system (especially CUDA version) by following the official instructions: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

3.  **Install Dependencies & TrainSense:**
    *(Assuming you have the code locally. Replace with `pip install trainsense` if published on PyPI)*
    ```bash
    # Install core dependencies (if requirements.txt is up-to-date)
    # pip install -r requirements.txt

    # Install TrainSense
    pip install .

    # Or for development (recommended):
    # pip install -e .

    # To install with plotting capabilities:
    pip install .[plotting]
    # Or if using development mode:
    # pip install -e .[plotting]
    ```
    The core dependencies (`psutil`, `torch`, `GPUtil`) are listed in `setup.py`'s `install_requires`. `matplotlib` is an optional dependency defined in `extras_require['plotting']`.

## Core Concepts

TrainSense aims to provide a holistic view by examining different facets of your training setup:

1.  **System Context (`SystemConfig`, `SystemDiagnostics`, `GPUMonitor`):** Understand the environment (hardware/software). *Can my GPU handle this batch size? Is my CPU bottlenecking data loading?*
2.  **Model Introspection (`ArchitectureAnalyzer`):** Look inside the model. *How complex is it? What layers are used? Might this architecture benefit from AdamW?*
3.  **Hyperparameter Evaluation (`TrainingAnalyzer`, `OptimizerHelper`, `UltraOptimizer`):** Assess training parameters. *Is my learning rate too high? Are enough epochs planned? Is SGD appropriate here?*
4.  **Performance Measurement (`ModelProfiler`):** Measure execution. *How fast is inference? Where is time spent during a *training* step (data fetch, data prep, forward, backward, optimizer)? How much memory is needed?*
5.  **Training Stability (`GradientAnalyzer`):** Check the learning process itself. *Are my gradients vanishing? Are they exploding? Are there NaNs?*
6.  **Synthesis (`DeepAnalyzer`):** Combine insights from various analyses into actionable recommendations.

## Getting Started: Quick Example

```python
import torch
import torch.nn as nn
import logging
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Import Key TrainSense Components ---
# Assuming TrainSense is installed or in PYTHONPATH
from TrainSense import (SystemConfig, ArchitectureAnalyzer, ModelProfiler,
                      DeepAnalyzer, TrainingAnalyzer, SystemDiagnostics,
                      GradientAnalyzer, OptimizerHelper, GPUMonitor, print_section,
                      plot_training_step_breakdown) # Import plotting function

# --- Define Your Model & Setup ---
model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10)) # Example model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
batch_size, lr, epochs = 32, 0.001, 10
# IMPORTANT: Define the correct input shape for your model for one batch!
input_shape = (batch_size, 128)
criterion = nn.CrossEntropyLoss().to(device) # Move criterion to device
optimizer = Adam(model.parameters(), lr=lr)

# Create dummy data for profiling/gradient analysis examples
dummy_X = torch.randn(*input_shape, device='cpu')
dummy_y = torch.randint(0, 10, (input_shape[0],), device='cpu', dtype=torch.long)
dummy_dataset = TensorDataset(dummy_X, dummy_y)
dummy_loader = DataLoader(dummy_dataset, batch_size=batch_size, num_workers=0)

# --- Instantiate TrainSense Components ---
print_section("Initializing TrainSense Components")
try:
    sys_config = SystemConfig()
    sys_diag = SystemDiagnostics()
    arch_analyzer = ArchitectureAnalyzer(model)
    arch_info = arch_analyzer.analyze() # Analyze first for context
    model_profiler = ModelProfiler(model, device=device)
    training_analyzer = TrainingAnalyzer(batch_size, lr, epochs, system_config=sys_config, arch_info=arch_info)
    grad_analyzer = GradientAnalyzer(model) # Needs model

    # DeepAnalyzer combines multiple analyses
    # Pass the gradient analyzer instance here
    deep_analyzer = DeepAnalyzer(training_analyzer, arch_analyzer, model_profiler, sys_diag, gradient_analyzer=grad_analyzer)
    print("TrainSense Components Initialized.")

    # --- Run Backward Pass (Needed for Gradient Analysis!) ---
    print_section("Setup: Running Backward Pass")
    GRADIENTS_AVAILABLE = False
    try:
        model.train()
        optimizer.zero_grad()
        inputs, targets = next(iter(dummy_loader)) # Get one batch
        outputs = model(inputs.to(device))
        loss = criterion(outputs, targets.to(device))
        loss.backward()
        print(f"Ran one backward pass (Loss: {loss.item():.3f})")
        GRADIENTS_AVAILABLE = True
    except Exception as e:
        logging.error(f"Failed to run backward pass for gradient analysis: {e}", exc_info=True)
        print(f"!! Failed to run backward pass: {e}")
    finally:
        model.eval() # Set back to eval mode

    # --- Run Comprehensive Analysis (including new v0.3.0 features) ---
    print_section("Running Comprehensive Analysis (v0.3.0)")
    report = {} # Initialize report dict
    try:
        report = deep_analyzer.comprehensive_report(
            profile_inference=True,             # Enable inference profiling
            profile_training=True,              # <<-- Enable training profiling
            gradient_analysis=GRADIENTS_AVAILABLE, # <<-- Enable gradient analysis (if backward succeeded)
            inference_input_shape=input_shape,  # Shape for inference
            training_data_loader=dummy_loader,  # Loader for training profile
            criterion=criterion,                # Criterion for training profile
            optimizer=optimizer,                # Optimizer for training profile
            profile_iterations=50,              # Iterations for inference profile
            train_profile_iterations=10         # Iterations for training profile
        )
        print("Comprehensive Analysis Complete.")

        # --- Display Key Findings ---
        print("\n>>> Overall Recommendations:")
        for rec in report.get("overall_recommendations", ["No recommendations available."]):
            print(f"- {rec}")

        # Optional: Display specific results directly
        train_profile = report.get("training_step_profiling", {})
        if "error" not in train_profile and train_profile:
            print("\n>>> Training Step Timing Breakdown (%):")
            print(f"  DataFetch={train_profile.get('percent_time_data_fetch', 0):.1f}, DataPrep={train_profile.get('percent_time_data_prep', 0):.1f}, Fwd={train_profile.get('percent_time_forward', 0):.1f}, Loss={train_profile.get('percent_time_loss', 0):.1f}, Bwd={train_profile.get('percent_time_backward', 0):.1f}, Opt={train_profile.get('percent_time_optimizer', 0):.1f}")

        grad_analysis = report.get("gradient_analysis", {})
        if "error" not in grad_analysis and grad_analysis:
             print("\n>>> Gradient Analysis Summary:")
             print(f"  Global Norm L2: {grad_analysis.get('global_grad_norm_L2', 'N/A'):.2e}")
             print(f"  NaN/Inf Grads: {grad_analysis.get('num_params_nan_grad', 0)} / {grad_analysis.get('num_params_inf_grad', 0)}")

    except Exception as e:
        logging.exception("Error during comprehensive report generation")
        print(f"\nERROR generating comprehensive report: {e}")

    # --- Generate Plot (Optional) ---
    print_section("Optional: Generate Training Step Plot")
    if report and "training_step_profiling" in report and "error" not in report["training_step_profiling"]:
        print("Attempting plot generation (requires matplotlib: `pip install trainsense[plotting]`)")
        try:
            if not os.path.exists("logs"): os.makedirs("logs") # Ensure log dir exists
            plot_generated = plot_training_step_breakdown(
                profile_results=report["training_step_profiling"],
                save_path="logs/training_step_plot.png",
                show_plot=False # Avoid blocking execution
            )
            if plot_generated: print("Plot saved to logs/training_step_plot.png")
            else: print("Plot generation failed. Is matplotlib installed?")
        except Exception as plot_err:
            print(f"Plotting error: {plot_err}")
    else:
        print("Skipping plot: Training profile data not available or has errors.")


except Exception as e:
    logging.exception("An error occurred during the TrainSense example.")
    print(f"\n--- OVERALL ERROR --- \nAn error occurred: {e}")

```

## Detailed Usage Examples

*(These examples show how to use each component individually)*

### 1. Checking System Configuration

```python
from TrainSense import SystemConfig, print_section

sys_config = SystemConfig()
summary = sys_config.get_summary() # Concise overview
print_section("System Summary")
for key, value in summary.items(): print(f"- {key.replace('_', ' ').title()}: {value}")
# full_config = sys_config.get_config() # More details
```

### 2. Analyzing Your Model's Architecture

```python
import torch.nn as nn
from TrainSense import ArchitectureAnalyzer, print_section

model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
arch_analyzer = ArchitectureAnalyzer(model)
analysis = arch_analyzer.analyze()
print_section("Architecture Analysis")
print(f"- Params (Total/Trainable): {analysis.get('total_parameters', 0):,} / {analysis.get('trainable_parameters', 0):,}")
print(f"- Layer Count: {analysis.get('layer_count', 'N/A')}")
print(f"- Primary Architecture: {analysis.get('primary_architecture_type', 'N/A')}")
print(f"- Complexity: {analysis.get('complexity_category', 'N/A')}")
print(f"- Estimated Input: {analysis.get('estimated_input_shape', 'N/A')}")
print(f"- Recommendation: {analysis.get('recommendation', 'N/A')}")
# ... print layer types ...
```

### 3. Getting Hyperparameter Recommendations

```python
from TrainSense import TrainingAnalyzer, SystemConfig, ArchitectureAnalyzer, print_section
import torch.nn as nn

model = nn.Linear(10, 2)
sys_config = SystemConfig()
arch_analyzer = ArchitectureAnalyzer(model); arch_info = arch_analyzer.analyze()
analyzer = TrainingAnalyzer(batch_size=512, learning_rate=0.1, epochs=5, system_config=sys_config, arch_info=arch_info)

print_section("Hyperparameter Checks")
recommendations = analyzer.check_hyperparameters()
print("Recommendations:"); [print(f"- {r}") for r in recommendations]
adjustments = analyzer.auto_adjust()
print("\nSuggested Adjustments:"); # ... print adjustments ...
```

### 4. Profiling Model Inference Performance

```python
import torch, torch.nn as nn
from TrainSense import ModelProfiler, print_section

model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_shape = (32, 64) # IMPORTANT: Set correct shape!

profiler = ModelProfiler(model, device=device)
print_section("Model Inference Profiling")
try:
    results = profiler.profile_model(input_shape, iterations=100, use_torch_profiler=True)
    if "error" in results: print(f"Error: {results['error']}")
    else:
        print(f"- Avg Time: {results.get('avg_total_time_ms', 0):.3f} ms")
        print(f"- Throughput: {results.get('throughput_samples_per_sec', 0):.1f} samples/sec")
        print(f"- Peak Memory: {results.get('max_memory_allocated_formatted', 'N/A')}")
        # ... print profiler stats ...
except Exception as e: print(f"Profiling failed: {e}")
```

### 5. Profiling a Full Training Step (New in v0.3.0!)

```python
import torch, torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from TrainSense import ModelProfiler, print_section

# --- Setup: Model, Device, Criterion, Optimizer, Loader ---
# (Define model, device, criterion, optimizer, dummy_loader as in Quick Start)
# ---
model = nn.Sequential(nn.Linear(64, 10)).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = Adam(model.parameters(), lr=0.001)
dummy_X = torch.randn(32, 64); dummy_y = torch.randint(0, 10, (32,), dtype=torch.long)
dummy_loader = DataLoader(TensorDataset(dummy_X, dummy_y), batch_size=32)
# ---

model_profiler = ModelProfiler(model, device=device)
print_section("Training Step Profiling")
try:
    results = model_profiler.profile_training_step(
        dummy_loader, criterion, optimizer, iterations=20, use_torch_profiler=True)

    if "error" in results: print(f"Error: {results['error']}")
    else:
        print(f"- Avg Step Time: {results.get('avg_step_time_ms', 0):.2f} ms")
        print(f"- Breakdown (%): DataFetch={results.get('percent_time_data_fetch', 0):.1f}, DataPrep={results.get('percent_time_data_prep', 0):.1f}, Fwd={results.get('percent_time_forward', 0):.1f}, Loss={results.get('percent_time_loss', 0):.1f}, Bwd={results.get('percent_time_backward', 0):.1f}, Opt={results.get('percent_time_optimizer', 0):.1f}")
        print(f"- Peak Memory: {results.get('max_memory_allocated_formatted', 'N/A')}")
        # ... optionally print detailed profiler table ...
except Exception as e: print(f"Training profiling failed: {e}")
```

### 6. Analyzing Gradients (New in v0.3.0!)

```python
import torch, torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from TrainSense import GradientAnalyzer, print_section

# --- Setup: Model, Device, Criterion, Optimizer, Loader ---
# (Define model, device, criterion, optimizer, dummy_loader as above)
# ---
model = nn.Sequential(nn.Linear(64, 10)).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = Adam(model.parameters(), lr=0.001)
dummy_X = torch.randn(32, 64); dummy_y = torch.randint(0, 10, (32,), dtype=torch.long)
dummy_loader = DataLoader(TensorDataset(dummy_X, dummy_y), batch_size=32)
# ---

grad_analyzer = GradientAnalyzer(model)
print_section("Gradient Analysis")
try:
    # --- CRITICAL: Run backward pass first! ---
    model.train()
    optimizer.zero_grad()
    inputs, targets = next(iter(dummy_loader))
    outputs = model(inputs.to(device))
    loss = criterion(outputs, targets.to(device))
    loss.backward()
    print(f"Ran backward pass (Loss: {loss.item():.4f}).")
    # -------------------------------------------

    grad_summary = grad_analyzer.summary()
    print("\n--- Gradient Summary ---")
    if "error" in grad_summary: print(f"Error: {grad_summary['error']}")
    else:
        print(f"- Global Grad Norm (L2): {grad_summary.get('global_grad_norm_L2', 'N/A'):.3e}")
        print(f"- Avg/Max Grad Norm: {grad_summary.get('avg_grad_norm', 'N/A'):.3e} / {grad_summary.get('max_grad_norm', 'N/A'):.3e}")
        print(f"- Layer w/ Max Norm: {grad_summary.get('layer_with_max_grad_norm', 'N/A')}")
        print(f"- NaN/Inf Grads: {grad_summary.get('num_params_nan_grad', 0)} / {grad_summary.get('num_params_inf_grad', 0)}")
except Exception as e: print(f"Gradient analysis failed: {e}")
finally: model.eval()
```

### 7. Monitoring GPU Status

```python
from TrainSense import GPUMonitor, print_section
try:
    gpu_monitor = GPUMonitor()
    print_section("GPU Status")
    if gpu_monitor.is_available():
        status_list = gpu_monitor.get_gpu_status()
        if status_list:
            for gpu in status_list: # Process each GPU
                print(f"GPU {gpu.get('id')}: {gpu.get('name')} | Load: {gpu.get('load', 0):.1f}% | Mem: {gpu.get('memory_used_mb', 0):.0f}/{gpu.get('memory_total_mb', 0):.0f}MB ({gpu.get('memory_utilization_percent', 0):.1f}%) | Temp: {gpu.get('temperature_celsius', 'N/A')}C")
        else: print("- No GPUs detected by GPUtil.")
    else: print("- GPUtil library unavailable.")
except Exception as e: print(f"GPU monitoring error: {e}")
```

### 8. Getting Optimizer and Scheduler Suggestions

```python
import torch.nn as nn
from TrainSense import OptimizerHelper, ArchitectureAnalyzer, print_section

model = nn.LSTM(10, 20, 2) # Example RNN
arch_analyzer = ArchitectureAnalyzer(model); arch_info = arch_analyzer.analyze()
print_section("Optimizer/Scheduler Suggestions")
suggested_opt = OptimizerHelper.suggest_optimizer(arch_info['total_parameters'], arch_info['layer_count'], arch_info['primary_architecture_type'])
print(f"- Optimizer: {suggested_optimizer}")
base_opt_name = suggested_opt.split(" ")[0]
print(f"- Scheduler for {base_opt_name}: {OptimizerHelper.suggest_learning_rate_scheduler(base_opt_name)}")
print(f"- Initial LR Suggestion: {OptimizerHelper.suggest_initial_learning_rate(arch_info['primary_architecture_type'], arch_info['total_parameters']):.1e}")
```

### 9. Generating Heuristic Hyperparameters (`UltraOptimizer`)

```python
from TrainSense import UltraOptimizer, SystemConfig, ArchitectureAnalyzer, print_section
import torch.nn as nn

model = nn.Linear(512, 10) # Simple MLP
sys_config = SystemConfig(); config_summary = sys_config.get_summary()
arch_analyzer = ArchitectureAnalyzer(model); arch_info = arch_analyzer.analyze()
data_stats = {"data_size": 150000} # Example data size

ultra_optimizer = UltraOptimizer(data_stats, arch_info, config_summary)
print_section("Heuristic Parameter Set (UltraOptimizer)")
result = ultra_optimizer.compute_heuristic_hyperparams()
print("Generated Params:", result.get("hyperparameters", {}))
# print("Reasoning:", result.get("reasoning", {})) # Optional: print reasons
```

### 10. Using the Comprehensive Reporter (`DeepAnalyzer`)

```python
# (Assume all necessary components like sys_config, arch_analyzer, model_profiler,
# training_analyzer, sys_diag, grad_analyzer, dummy_loader, criterion, optimizer
# are initialized as shown in the Quick Start example)
# Also assume a backward pass was run if gradient_analysis=True

from TrainSense import DeepAnalyzer, print_section

print_section("Comprehensive Report Generation")
deep_analyzer = DeepAnalyzer(training_analyzer, arch_analyzer, model_profiler, sys_diag, grad_analyzer)

try:
    report = deep_analyzer.comprehensive_report(
        profile_inference=True,
        profile_training=True,
        gradient_analysis=GRADIENTS_AVAILABLE, # Use flag from backward pass attempt
        inference_input_shape=input_shape,     # Defined earlier
        training_data_loader=dummy_loader,     # Defined earlier
        criterion=criterion,                   # Defined earlier
        optimizer=optimizer                    # Defined earlier
    )
    print("\n--- Overall Recommendations from DeepAnalyzer ---")
    for rec in report.get("overall_recommendations", []): print(f"- {rec}")
    # Access specific report sections: report['system_diagnostics'], report['training_step_profiling'], etc.

except Exception as e:
    print(f"Failed to generate comprehensive report: {e}")

```

### 11. Plotting Training Breakdown (Optional, New in v0.3.0!)

```python
from TrainSense import plot_training_step_breakdown
import os
# (Assume 'report' dictionary from DeepAnalyzer contains 'training_step_profiling' results)

print_section("Generate Training Step Plot")
if report and "training_step_profiling" in report and "error" not in report["training_step_profiling"]:
    print("Attempting plot (requires matplotlib: `pip install trainsense[plotting]`)")
    if not os.path.exists("logs"): os.makedirs("logs")
    try:
        success = plot_training_step_breakdown(
            report["training_step_profiling"],
            save_path="logs/training_breakdown.png",
            show_plot=False # Don't block execution in scripts
        )
        if success: print("Plot saved to logs/training_breakdown.png")
        else: print("Plot generation failed (matplotlib installed?)")
    except Exception as e: print(f"Plotting error: {e}")
else: print("Skipping plot: Training profile data unavailable.")
```

### 12. Using the Logger

```python
import logging
# from TrainSense.logger import TrainLogger # Or use standard logging

# Configure standard logging (run early in your script)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("MyScript") # Use any name

# Log messages
logger.info("Starting analysis...")
logger.warning("Potential issue detected.")
try: 1/0
except ZeroDivisionError: logger.error("Something went wrong!", exc_info=True)
```

## Interpreting the Output

*   **Training Step Profiling:**
    *   **High `% Data Fetch/Prep`:** Your bottleneck is likely I/O or data preprocessing. Increase `num_workers` in `DataLoader`, optimize transforms, check disk speed, pre-fetch data.
    *   **High `% Backward Pass`:** Expected for complex models, but very high values might indicate inefficient layers or large activation memory. Consider activation checkpointing for very large models.
    *   **High `% Optimizer Step`:** Can happen with complex optimizers (like AdamW with many parameters) or if using techniques like gradient clipping extensively. Usually less of a bottleneck than backward or data loading.
*   **Gradient Analysis:**
    *   **High `Global Grad Norm` / `Max Grad Norm`:** Potential for exploding gradients. Consider gradient clipping (`torch.nn.utils.clip_grad_norm_`).
    *   **Very Low `Global Grad Norm` / `Avg Grad Norm` (approaching zero):** Potential for vanishing gradients, especially in deep networks or RNNs. Check initialization, consider different activation functions (ReLU variants), use normalization layers (BatchNorm, LayerNorm), or architectures like ResNets/LSTMs/GRUs.
    *   **`NaN/Inf Grads Found > 0`:** Serious problem! Training will likely diverge. Common causes: learning rate too high, numerical instability (e.g., log(0), division by zero), issues with mixed precision (`amp`), bad data. Reduce learning rate significantly, check data pipelines, enable anomaly detection (`torch.autograd.set_detect_anomaly(True)` - **slows training!**).
*   **Other Common Patterns:**
    *   **High CPU Usage / Low GPU Utilization (General):** Often points to data loading issues (see training profiler), but could also be excessive Python logic between GPU calls.
    *   **High GPU Memory Usage (`max_memory_allocated`):** Your model or batch size might be too large for the GPU VRAM. Consider reducing batch size, using gradient accumulation, mixed-precision training (`torch.cuda.amp`), or model optimization techniques (pruning, quantization).
    *   **Inference Profiler Bottlenecks:** Look at the `profiler_top_ops_summary`. If specific operations take disproportionate time, investigate optimization.

## Contributing

Contributions are welcome! Please feel free to open an issue to discuss potential features or bug fixes, or submit a pull request. (Consider adding more specific guidelines later, e.g., code style, testing requirements).

## License

This project is licensed under the MIT License. See the LICENSE file (if available in the repository) for details.
