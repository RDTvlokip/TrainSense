# TrainSense v0.4.0: Analyze, Profile, and Optimize your PyTorch Training Workflow

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- Badges for CI/CD, PyPI version, etc., can be added later -->

TrainSense is a Python toolkit designed to provide deep insights into your PyTorch model training environment and performance. It helps you understand your system's capabilities, analyze your model's architecture, evaluate hyperparameter choices, profile execution bottlenecks (including full training steps), diagnose gradient issues, generate comprehensive reports, and ultimately optimize your deep learning workflow.

Whether you're debugging slow training, trying to maximize GPU utilization, investigating vanishing/exploding gradients, or simply want a clearer picture of your setup, TrainSense offers a suite of tools to assist you.

---

## Table of Contents

*   [Key Features](#key-features)
*   [What's New in v0.4.0](#whats-new-in-v040)
*   [Installation](#installation)
*   [Core Concepts](#core-concepts)
*   [Getting Started: Quick Example](#getting-started-quick-example)
*   [Detailed Usage Examples](#detailed-usage-examples)
    *   [1. System Configuration (`SystemConfig`)](#1-checking-system-configuration)
    *   [2. Architecture Analysis (`ArchitectureAnalyzer`)](#2-analyzing-your-models-architecture)
    *   [3. Hyperparameter Recommendations (`TrainingAnalyzer`)](#3-getting-hyperparameter-recommendations)
    *   [4. Inference Performance Profiling (`ModelProfiler.profile_model`)](#4-profiling-model-inference-performance)
    *   [5. Training Step Profiling (`ModelProfiler.profile_training_step`)](#5-profiling-a-full-training-step)
    *   [6. Gradient Analysis (`GradientAnalyzer`)](#6-analyzing-gradients)
    *   [7. GPU Monitoring (`GPUMonitor`)](#7-monitoring-gpu-status)
    *   [8. Optimizer & Scheduler Suggestions (`OptimizerHelper`)](#8-getting-optimizer-and-scheduler-suggestions)
    *   [9. Heuristic Hyperparameters (`UltraOptimizer`)](#9-generating-heuristic-hyperparameters-ultraoptimizer)
    *   [10. Comprehensive Reporting (`DeepAnalyzer`)](#10-using-the-comprehensive-reporter-deepanalyzer)
    *   [11. Generating HTML Reports (New!)](#11-generating-html-reports-new)
    *   [12. Plotting Training Breakdown (`visualizer`)](#12-plotting-training-breakdown-optional)
    *   [13. Plotting Gradient Histogram (`GradientAnalyzer`)](#13-plotting-gradient-histogram-optional)
    *   [14. Real-Time Monitoring (`RealTimeMonitor`)](#14-real-time-monitoring-experimental)
    *   [15. Integration Hooks/Callbacks (`integrations`)](#15-integration-hookscallbacks-experimental)
    *   [16. Logging (`TrainLogger`)](#16-using-the-logger)
*   [Interpreting the Output](#interpreting-the-output)
*   [Contributing](#contributing)
*   [License](#license)

---

## Key Features

*   **System Analysis:** `SystemConfig` (Hardware/Software Detection), `SystemDiagnostics` (Real-time Usage).
*   **Model Architecture Insight:** `ArchitectureAnalyzer` (Params, Layers, Type, Complexity, Recommendations).
*   **Hyperparameter Sanity Checks:** `TrainingAnalyzer` (Batch Size, LR, Epochs checks & suggestions).
*   **Advanced Performance Profiling:** `ModelProfiler` (Inference & **Full Training Step** profiling with `torch.profiler` integration, detailed data loading breakdown).
*   **Gradient Diagnostics:** `GradientAnalyzer` (Per-parameter stats, Norms, NaN/Inf detection, Summary, Optional Histogram Plotting).
*   **GPU Monitoring:** `GPUMonitor` (Real-time Load, Memory, Temp via `GPUtil`).
*   **Training Optimization Guidance:** `OptimizerHelper` (Optimizer/Scheduler suggestions), `UltraOptimizer` (Heuristic parameter generation).
*   **Comprehensive & Integrated Reporting:** `DeepAnalyzer` orchestrates analyses (including optional Training/Gradient analysis) into a single dictionary report **and optionally generates standalone HTML reports (New!)**. Aggregated recommendations consider multiple factors.
*   **Visualization:** Optional plotting functions for Training Step Breakdown and Gradient Norm Histograms (requires `matplotlib`).
*   **Real-Time Monitoring:** `RealTimeMonitor` class to track system/GPU usage in a background thread during specific code sections (Experimental).
*   **Integration Utilities:** `TrainStepMonitorHook` (PyTorch Hooks) & `TrainSenseTRLCallback` (for Hugging Face `Trainer` / `SFTTrainer`) provide ways to integrate monitoring into training loops (Experimental).
*   **Flexible Logging:** `TrainLogger` for configurable logging.

## What's New in v0.4.0

*   ✨ **Integrated Comprehensive Reports (`DeepAnalyzer`):** `DeepAnalyzer.comprehensive_report` can now more easily include results from Training Step Profiling and Gradient Analysis alongside other metrics, providing a truly holistic view when enabled. Recommendations are enhanced based on this broader data.
*   📄 **HTML Report Generation (New!):** Export the detailed comprehensive report from `DeepAnalyzer` into a self-contained, shareable HTML file, including plots if generated (requires `jinja2`, install via `pip install trainsense[html]`).
*   🤗 **Basic TRL Integration (`TrainSenseTRLCallback`):** Added an experimental callback for Hugging Face `Trainer` / `SFTTrainer`. Attach it during training to automatically log gradient summaries and basic step timing (requires `transformers`).
*   ⏱️ **Refined Training Profiling:** `ModelProfiler.profile_training_step` provides clearer separation and reporting of data fetching vs. data preparation times.
*   📈 **Improved Plotting:** Enhancements to the gradient histogram and training breakdown plots for clarity and robustness.
*   🔧 **Usability & Fixes:** General code cleanup, improved error handling, and minor refinements across modules. Updated documentation and examples.

## Installation

It's highly recommended to use a virtual environment.

1.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Linux/macOS: source venv/bin/activate
    # On Windows: venv\Scripts\activate
    ```

2.  **Install PyTorch:** Follow official instructions for your system/CUDA version: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

3.  **Install TrainSense:**
    *(Assuming you have the code locally. Replace with `pip install trainsense` if published on PyPI)*

    *   **Core Installation:**
        ```bash
        pip install .
        # Or for development (recommended):
        # pip install -e .
        ```
    *   **With Optional Features:**
        ```bash
        # For plotting capabilities (matplotlib)
        pip install .[plotting]

        # For HTML report generation (jinja2)
        pip install .[html]

        # For both plotting and HTML reports
        pip install .[plotting,html]

        # For development (includes plotting, html, testing tools)
        pip install -e .[dev]
        ```
    The required dependencies (`psutil`, `torch`, `GPUtil`) are installed automatically. Optional ones (`matplotlib`, `jinja2`, `transformers`) are managed via `extras_require`.

## Core Concepts

TrainSense provides a multi-faceted analysis pipeline:

1.  **Environment Setup (`SystemConfig`, `SystemDiagnostics`, `GPUMonitor`):** What are we working with?
2.  **Model Structure (`ArchitectureAnalyzer`):** What are we training? How complex is it?
3.  **Training Plan (`TrainingAnalyzer`, `OptimizerHelper`, `UltraOptimizer`):** Are the chosen settings sensible? What are good starting points?
4.  **Execution Performance (`ModelProfiler`):** How fast does it run (inference *and* training steps)? Where are the bottlenecks? How much memory does it use?
5.  **Learning Dynamics (`GradientAnalyzer`):** Is the learning process stable? Are gradients healthy?
6.  **Synthesis & Reporting (`DeepAnalyzer`, `visualizer`):** Combine all insights into actionable recommendations and easily digestible reports (Dictionary or HTML).
7.  **Integration (`integrations`):** Tools to embed monitoring directly into training loops.

## Getting Started: Quick Example

```python
import torch
import torch.nn as nn
import logging
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

from TrainSense import (SystemConfig, ArchitectureAnalyzer, ModelProfiler,
                      DeepAnalyzer, TrainingAnalyzer, SystemDiagnostics,
                      GradientAnalyzer, print_section) # Import main components

# --- Define Model & Setup ---
model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 5))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
batch_size, lr, epochs = 16, 1e-3, 5
input_shape = (batch_size, 64)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = Adam(model.parameters(), lr=lr)
dummy_X = torch.randn(*input_shape); dummy_y = torch.randint(0, 5, (batch_size,), dtype=torch.long)
dummy_loader = DataLoader(TensorDataset(dummy_X, dummy_y), batch_size=batch_size)

# --- Instantiate TrainSense ---
try:
    sys_config = SystemConfig()
    sys_diag = SystemDiagnostics()
    arch_analyzer = ArchitectureAnalyzer(model)
    arch_info = arch_analyzer.analyze()
    model_profiler = ModelProfiler(model, device=device)
    training_analyzer = TrainingAnalyzer(batch_size, lr, epochs, system_config=sys_config, arch_info=arch_info)
    grad_analyzer = GradientAnalyzer(model)
    deep_analyzer = DeepAnalyzer(training_analyzer, arch_analyzer, model_profiler, sys_diag, grad_analyzer)
    print("TrainSense Initialized.")

    # --- Run Backward Pass (Required for Gradient Analysis) ---
    print_section("Setup: Running Backward Pass")
    model.train()
    optimizer.zero_grad()
    inputs, targets = next(iter(dummy_loader))
    outputs = model(inputs.to(device))
    loss = criterion(outputs, targets.to(device))
    loss.backward()
    print(f"Backward pass complete (Loss: {loss.item():.4f}). Gradients available.")
    model.eval()
    GRADIENTS_AVAILABLE = True

    # --- Generate Comprehensive Report (including optional analyses) ---
    print_section("Running Comprehensive Analysis & HTML Report")
    report = deep_analyzer.comprehensive_report(
        profile_inference=True,
        profile_training=True,           # <<-- Enable training profiling
        gradient_analysis=GRADIENTS_AVAILABLE, # <<-- Enable gradient analysis
        inference_input_shape=input_shape,
        training_data_loader=dummy_loader,
        criterion=criterion,
        optimizer=optimizer,
        profile_iterations=20, train_profile_iterations=5, # Fewer iters for example
        # --- NEW: Save to HTML ---
        save_html_path="trainsense_report.html" # Requires jinja2: pip install trainsense[html]
    )
    print("Comprehensive Analysis Complete.")
    if os.path.exists("trainsense_report.html"):
        print(">> HTML Report saved to trainsense_report.html <<")

    # --- Display Key Findings ---
    print("\n>>> Overall Recommendations:")
    for rec in report.get("overall_recommendations", ["N/A"]): print(f"- {rec}")

    train_profile = report.get("training_step_profiling", {})
    if "error" not in train_profile and train_profile:
        print("\n>>> Training Step Timing Breakdown (%):")
        print(f"  DataLoad={train_profile.get('percent_time_data_total_load', 0):.1f}, Fwd={train_profile.get('percent_time_forward', 0):.1f}, Bwd={train_profile.get('percent_time_backward', 0):.1f}, Opt={train_profile.get('percent_time_optimizer', 0):.1f}")

    grad_analysis = report.get("gradient_analysis", {})
    if "error" not in grad_analysis and grad_analysis:
         print("\n>>> Gradient Analysis Summary:")
         print(f"  Global Norm L2: {grad_analysis.get('global_grad_norm_L2', 'N/A'):.2e}, NaN/Inf Grads: {grad_analysis.get('num_params_nan_grad', 0)}/{grad_analysis.get('num_params_inf_grad', 0)}")

except Exception as e:
    logging.exception("Error during TrainSense quick start example")
    print(f"\nERROR: {e}")

```

## Detailed Usage Examples

*(These examples show how to use each component individually. Assumes basic setup like model, device, data loader etc., are defined as needed.)*

### 1. Checking System Configuration
*(Same as v0.3.0)*
```python
from TrainSense import SystemConfig, print_section
sys_config = SystemConfig(); summary = sys_config.get_summary()
print_section("System Summary"); [print(f"- {k}: {v}") for k,v in summary.items()]
```

### 2. Analyzing Your Model's Architecture
*(Same as v0.3.0)*
```python
import torch.nn as nn
from TrainSense import ArchitectureAnalyzer, print_section
model = nn.Linear(100,10); arch_analyzer = ArchitectureAnalyzer(model)
analysis = arch_analyzer.analyze()
print_section("Architecture Analysis"); print(f"- Params: {analysis.get('total_parameters', 0):,}, Type: {analysis.get('primary_architecture_type')}") # etc.
```

### 3. Getting Hyperparameter Recommendations
*(Same as v0.3.0)*
```python
from TrainSense import TrainingAnalyzer, SystemConfig, ArchitectureAnalyzer, print_section
# (Define model, sys_config, get arch_info)
model = nn.Linear(10,2); sys_config = SystemConfig(); arch_analyzer = ArchitectureAnalyzer(model); arch_info = arch_analyzer.analyze()
analyzer = TrainingAnalyzer(512, 0.1, 5, system_config=sys_config, arch_info=arch_info)
print_section("Hyperparameter Checks"); [print(f"- {r}") for r in analyzer.check_hyperparameters()]
# print("Adjustments:", analyzer.auto_adjust())
```

### 4. Profiling Model Inference Performance
*(Same as v0.3.0)*
```python
import torch, torch.nn as nn
from TrainSense import ModelProfiler, print_section
model = nn.Linear(64, 10).to(device); input_shape = (32, 64)
profiler = ModelProfiler(model, device=device)
print_section("Model Inference Profiling")
results = profiler.profile_model(input_shape, iterations=50, use_torch_profiler=True)
# (Process results dictionary)
```

### 5. Profiling a Full Training Step
*(Same as v0.3.0, but remember the refined data loading breakdown)*
```python
import torch, torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from TrainSense import ModelProfiler, print_section
# (Define model, device, criterion, optimizer, dummy_loader)
model = nn.Linear(64,10).to(device); criterion=nn.CrossEntropyLoss().to(device); optimizer=Adam(model.parameters());
dummy_X=torch.randn(100,64); dummy_y=torch.randint(0,10,(100,)); loader=DataLoader(TensorDataset(dummy_X,dummy_y), batch_size=32)

model_profiler = ModelProfiler(model, device=device)
print_section("Training Step Profiling")
results = model_profiler.profile_training_step(loader, criterion, optimizer, iterations=10, use_torch_profiler=True)
# (Process results, check breakdown: percent_time_data_fetch, percent_time_data_prep etc.)
```

### 6. Analyzing Gradients
*(Same as v0.3.0, ensure backward pass is run first)*
```python
import torch, torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from TrainSense import GradientAnalyzer, print_section
# (Define model, device, criterion, optimizer, dummy_loader)
model=nn.Linear(64,10).to(device); criterion=nn.CrossEntropyLoss().to(device); optimizer=Adam(model.parameters())
dummy_X=torch.randn(32,64); dummy_y=torch.randint(0,10,(32,)); loader=DataLoader(TensorDataset(dummy_X,dummy_y), batch_size=32)

grad_analyzer = GradientAnalyzer(model)
print_section("Gradient Analysis")
# --- Run backward pass ---
model.train(); optimizer.zero_grad()
inputs, targets = next(iter(loader)); outputs = model(inputs.to(device))
loss = criterion(outputs, targets.to(device)); loss.backward(); model.eval()
print("Ran backward pass.")
# --- Analyze ---
summary = grad_analyzer.summary()
print("Summary:", summary)
# grad_stats = grad_analyzer.analyze_gradients() # For per-layer details
```

### 7. Monitoring GPU Status
*(Same as v0.3.0)*
```python
from TrainSense import GPUMonitor, print_section
# (Handle potential GPUtil unavailability)
gpu_monitor = GPUMonitor(); print_section("GPU Status")
if gpu_monitor.is_available(): [print(gpu) for gpu in gpu_monitor.get_gpu_status()]
else: print("GPUtil not available.")
```

### 8. Getting Optimizer and Scheduler Suggestions
*(Same as v0.3.0)*
```python
import torch.nn as nn
from TrainSense import OptimizerHelper, ArchitectureAnalyzer, print_section
model=nn.LSTM(10,20); arch_analyzer=ArchitectureAnalyzer(model); arch_info=arch_analyzer.analyze()
print_section("Optimizer/Scheduler Suggestions")
print("Optimizer:", OptimizerHelper.suggest_optimizer(arch_info['total_parameters'], arch_info['layer_count'], arch_info['primary_architecture_type']))
# ... etc ...
```

### 9. Generating Heuristic Hyperparameters (`UltraOptimizer`)
*(Same as v0.3.0)*
```python
from TrainSense import UltraOptimizer, SystemConfig, ArchitectureAnalyzer, print_section
# (Define model, get sys_config_summary, arch_info, data_stats)
model=nn.Linear(128,10); sys_config=SystemConfig(); cfg_sum=sys_config.get_summary(); arch_analyzer=ArchitectureAnalyzer(model); arch_info=arch_analyzer.analyze(); data_stats={"data_size":10000}
ultra_optimizer = UltraOptimizer(data_stats, arch_info, cfg_sum)
print_section("Heuristic Parameter Set (UltraOptimizer)")
result = ultra_optimizer.compute_heuristic_hyperparams(); print("Params:", result.get("hyperparameters"))
```

### 10. Using the Comprehensive Reporter (`DeepAnalyzer`)

Now integrates more analyses if requested.

```python
from TrainSense import DeepAnalyzer # ... plus other components ...
# (Assume all components are initialized: training_analyzer, arch_analyzer, model_profiler, sys_diag, grad_analyzer)
# (Assume backward pass run if gradient_analysis=True)
# (Assume dummy_loader, criterion, optimizer defined)

deep_analyzer = DeepAnalyzer(training_analyzer, arch_analyzer, model_profiler, sys_diag, grad_analyzer)
print_section("Comprehensive Report")
report = deep_analyzer.comprehensive_report(
    profile_inference=True,
    profile_training=True,           # Enable training profiling in report
    gradient_analysis=True,          # Enable gradient analysis in report
    inference_input_shape=input_shape, # Need shape for inference
    training_data_loader=dummy_loader, # Need loader for training profile
    criterion=criterion,               # Need criterion for training profile
    optimizer=optimizer                # Need optimizer for training profile
    # save_html_path="report.html"     # Optionally save directly
)
print("Report generated (dictionary). Access keys like 'training_step_profiling', 'gradient_analysis'.")
print("Overall Recommendations:", report.get("overall_recommendations"))
```

### 11. Generating HTML Reports (New!)

Export the comprehensive report to an HTML file. Requires `jinja2` (`pip install trainsense[html]`).

```python
from TrainSense import DeepAnalyzer # ... plus other components ...
# (Assume deep_analyzer is initialized and backward pass run if needed)
# (Assume dummy_loader, criterion, optimizer defined if profiling training)

print_section("Generate HTML Report")
try:
    # Make sure necessary data exists (e.g., run backward pass if needed)
    # ... run backward pass if gradient_analysis=True ...

    html_path = "trainsense_comprehensive_report.html"
    report_dict = deep_analyzer.comprehensive_report(
        profile_inference=True, profile_training=True, gradient_analysis=True,
        inference_input_shape=input_shape, training_data_loader=dummy_loader,
        criterion=criterion, optimizer=optimizer,
        # NEW argument to save HTML
        save_html_path=html_path
    )
    if os.path.exists(html_path):
        print(f"Successfully generated HTML report: {html_path}")
    else:
        # Check if 'error' key exists in report_dict for clues
        if report_dict.get("html_export_error"):
             print(f"HTML report generation failed: {report_dict['html_export_error']}")
        else:
             print("HTML report generation failed. Check logs. Is jinja2 installed? (`pip install trainsense[html]`)")

except ImportError:
    print("HTML report generation requires 'jinja2'. Install with `pip install trainsense[html]`")
except Exception as e:
    print(f"Error generating report or HTML: {e}")

```

### 12. Plotting Training Breakdown (Optional)
*(Same as v0.3.0 example, using results from `profile_training_step`)*
```python
from TrainSense import plot_training_step_breakdown
# (Assume 'train_profile_results' dictionary exists from ModelProfiler)
if train_profile_results and "error" not in train_profile_results:
    plot_training_step_breakdown(train_profile_results, save_path="training_breakdown.png", show_plot=False)
```

### 13. Plotting Gradient Histogram (Optional)
*(Same as v0.3.0 example, using `GradientAnalyzer` instance)*
```python
from TrainSense import GradientAnalyzer
# (Assume grad_analyzer initialized and backward pass run)
if grad_analyzer:
    grad_analyzer.plot_gradient_norm_histogram(save_path="grad_histogram.png", show_plot=False)
```

### 14. Real-Time Monitoring (Experimental)
*(Same as v0.3.0 example)*
```python
from TrainSense import RealTimeMonitor
import time
monitor = RealTimeMonitor(interval_sec=1.0)
with monitor: # Use as context manager
    print("Monitoring for 3 seconds...")
    time.sleep(3.1)
history = monitor.get_history()
print(f"Collected {len(history)} snapshots during monitoring.")
```

### 15. Integration Hooks/Callbacks (Experimental)

**Using PyTorch Hooks:**
```python
from TrainSense import TrainStepMonitorHook
# (Assume model, criterion, optimizer, loader defined)
hook = TrainStepMonitorHook(model)
with hook: # Attaches/detaches hooks automatically
    model.train()
    for batch in loader: # Example loop
        hook.record_batch_start()
        try:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            hook.record_loss(loss) # Record before backward
            loss.backward()
            optimizer.step()
        except Exception: hook.record_batch_error(); raise
summary = hook.get_summary()
print("Hook Summary:", summary)
```

**Using TRL Callback (Conceptual - Requires `transformers` & `trl`):**
```python
# Conceptual Example - Requires TRL setup
# from TrainSense import TrainSenseTRLCallback, GradientAnalyzer
# from transformers import TrainingArguments, Trainer # Or SFTTrainer
# grad_analyzer_trl = GradientAnalyzer(your_trl_model)
# callback = TrainSenseTRLCallback(gradient_analyzer=grad_analyzer_trl)
# training_args = TrainingArguments(...)
# trainer = Trainer(model=your_trl_model, args=training_args, ..., callbacks=[callback])
# trainer.train()
```

### 16. Using the Logger
*(Same as v0.3.0)*
```python
import logging
# Configure standard logging or TrainLogger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MyApp")
logger.info("Analysis started.")
logger.warning("Potential issue found.")
```

## Interpreting the Output

*   **Comprehensive HTML Report:** This is often the easiest way to view results. Look for warnings, high resource usage, performance breakdowns, and gradient issues all in one place.
*   **Training Step Profiling:**
    *   **High `% Data Fetch/Prep`:** Bottleneck in I/O or preprocessing. Increase `DataLoader` `num_workers`, optimize transforms, check disk speed.
    *   **High `% Backward Pass`:** Expected for complex models. If step time is high, check detailed profiler (`profiler_top_ops_summary` in results dict) for specific layer bottlenecks. Consider activation checkpointing.
*   **Gradient Analysis:**
    *   **High `Global Grad Norm`:** Risk of exploding gradients. Consider gradient clipping.
    *   **Low `Global Grad Norm`:** Risk of vanishing gradients. Check initialization, activations, normalization.
    *   **`NaN/Inf Grads Found > 0`:** Critical issue. Check LR, data, operations, mixed precision usage.
*   **Other Patterns:** High CPU/Low GPU often means data loading issues. High GPU memory needs batch size reduction, AMP, or model optimization.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub. (Add specific guidelines if available).

## License

This project is licensed under the MIT License. See the LICENSE file for details.
