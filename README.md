# TrainSense: Analyze, Profile, and Optimize your PyTorch Training Workflow

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

TrainSense is a Python toolkit designed to provide deep insights into your PyTorch model training environment and performance. It helps you understand your system's capabilities, analyze your model's architecture, evaluate hyperparameter choices, profile execution bottlenecks, and ultimately optimize your deep learning workflow.

Whether you're debugging slow training, trying to maximize GPU utilization, or simply want a clearer picture of your setup, TrainSense offers a suite of tools to assist you.

## Key Features

*   **System Analysis:**
    *   **`SystemConfig`:** Detects hardware (CPU, RAM, GPU), OS, Python, PyTorch, CUDA, and cuDNN versions.
    *   **`SystemDiagnostics`:** Monitors real-time system resource usage (CPU, Memory, Disk, Network).
*   **Model Architecture Insight:**
    *   **`ArchitectureAnalyzer`:** Counts parameters (total/trainable), layers, analyzes layer types, estimates input shape, infers architecture type (CNN, RNN, Transformer...), and provides complexity assessment and recommendations.
*   **Hyperparameter Sanity Checks:**
    *   **`TrainingAnalyzer`:** Evaluates batch size, learning rate, and epochs based on system resources and model complexity. Provides recommendations and suggests automatic adjustments.
*   **Performance Profiling:**
    *   **`ModelProfiler`:** Measures inference speed (latency, throughput) and integrates `torch.profiler` for detailed operator-level CPU/GPU time and memory usage analysis. Identifies bottlenecks.
*   **GPU Monitoring:**
    *   **`GPUMonitor`:** Provides real-time, detailed GPU status including load, memory utilization (used, total), and temperature (requires `GPUtil`).
*   **Training Optimization Guidance:**
    *   **`OptimizerHelper`:** Suggests suitable optimizers (Adam, AdamW, SGD) and learning rate schedulers based on model characteristics. Recommends initial learning rates.
    *   **`UltraOptimizer`:** Generates a full set of heuristic hyperparameters (batch size, LR, epochs, optimizer, scheduler) as a starting point, based on system, model, and basic data stats.
*   **Consolidated Reporting:**
    *   **`DeepAnalyzer`:** Orchestrates the analysis, profiling, and diagnostics modules to generate a comprehensive report with aggregated insights and recommendations.
*   **Flexible Logging:**
    *   **`TrainLogger`:** Configurable logging to console and rotating files.

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
    Clone the repository (if you have it locally) or install directly if published (replace with pip install command if on PyPI). Assuming you have the code locally:
    ```bash
    # Ensure requirements.txt exists and lists psutil, GPUtil
    pip install -r requirements.txt
    pip install .
    # Or for development mode:
    # pip install -e .
    ```

## Core Concepts

TrainSense works by examining different facets of your training setup:

1.  **System Context (`SystemConfig`, `SystemDiagnostics`, `GPUMonitor`):** Understand the hardware and software environment your model runs in. Is CUDA available? How much GPU memory? What's the current CPU load?
2.  **Model Introspection (`ArchitectureAnalyzer`):** Look inside the model. How complex is it? What kinds of layers are used? This influences resource needs and hyperparameter choices.
3.  **Hyperparameter Evaluation (`TrainingAnalyzer`, `OptimizerHelper`, `UltraOptimizer`):** Assess if the chosen batch size, learning rate, epochs, and optimizer are sensible given the system and model context. Get suggestions for better starting points.
4.  **Performance Measurement (`ModelProfiler`):** Run the model (usually in inference mode) to measure its speed and, more importantly, use detailed profiling to see exactly where time is spent (CPU vs GPU, specific operations) and how much memory is consumed.
5.  **Synthesis (`DeepAnalyzer`):** Combine all these pieces of information into a single report, highlighting potential issues (e.g., "High memory usage detected," "Low GPU utilization suggests bottleneck") and providing actionable recommendations.

## Getting Started: A Quick Example

```python
import torch
import torch.nn as nn
import logging

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Import Key TrainSense Components ---
from TrainSense.system_config import SystemConfig
from TrainSense.arch_analyzer import ArchitectureAnalyzer
from TrainSense.model_profiler import ModelProfiler
from TrainSense.deep_analyzer import DeepAnalyzer
from TrainSense.analyzer import TrainingAnalyzer
from TrainSense.system_diagnostics import SystemDiagnostics
from TrainSense.utils import print_section

# --- Define Your Model ---
# Replace with your actual model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        # Calculate flattened size based on an example input (e.g., 32x32)
        # For 3x32x32 -> Conv1(16x32x32) -> Pool(16x16x16) -> Flatten(16*16*16=4096)
        self.fc = nn.Linear(16 * 16 * 16, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

model = SimpleCNN()

# --- Define Initial Training Parameters ---
batch_size = 64
learning_rate = 0.01
epochs = 20

# --- Instantiate TrainSense Components ---
print_section("Initializing TrainSense Components")
try:
    sys_config = SystemConfig()
    sys_diag = SystemDiagnostics()
    arch_analyzer = ArchitectureAnalyzer(model)

    # Use estimated input shape or provide a known one
    # Let's assume 3x32x32 images for this CNN
    example_input_shape = (batch_size, 3, 32, 32) # Use your actual batch size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_profiler = ModelProfiler(model, device=device)

    # Get architecture info needed by TrainingAnalyzer
    arch_info = arch_analyzer.analyze()

    training_analyzer = TrainingAnalyzer(
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        system_config=sys_config, # Pass the config object
        arch_info=arch_info
    )

    # The DeepAnalyzer brings it all together
    deep_analyzer = DeepAnalyzer(
        training_analyzer=training_analyzer,
        arch_analyzer=arch_analyzer,
        model_profiler=model_profiler,
        system_diag=sys_diag
    )
    print("TrainSense Components Initialized.")

    # --- Run Comprehensive Analysis ---
    print_section("Running Comprehensive Analysis")
    # Profiling needs an input shape
    report = deep_analyzer.comprehensive_report(profile_input_shape=example_input_shape)
    print("Comprehensive Analysis Complete.")

    # --- Display Key Findings ---
    print_section("Key Findings from Report")

    print("\n>>> Overall Recommendations:")
    if report.get("overall_recommendations"):
        for rec in report["overall_recommendations"]:
            print(f"- {rec}")
    else:
        print("- No overall recommendations generated.")

    print("\n>>> Hyperparameter Suggestions:")
    adjustments = report.get("hyperparameter_analysis", {}).get("suggested_adjustments", {})
    if adjustments and adjustments != {"batch_size": batch_size, "learning_rate": learning_rate, "epochs": epochs}:
         for k, v in adjustments.items():
             print(f"- Suggested {k}: {v}")
    else:
        print("- Initial hyperparameters seem reasonable or no adjustments suggested.")

    print("\n>>> Profiling Summary:")
    profiling_data = report.get("model_profiling", {})
    if "error" not in profiling_data and profiling_data:
        avg_time = profiling_data.get('avg_total_time_ms', 'N/A')
        throughput = profiling_data.get('throughput_samples_per_sec', 'N/A')
        max_mem = profiling_data.get('max_memory_allocated_formatted', 'N/A')
        gpu_util = profiling_data.get('avg_gpu_time_percent', None)
        print(f"- Avg Inference Time: {avg_time:.2f} ms" if isinstance(avg_time, float) else f"- Avg Inference Time: {avg_time}")
        print(f"- Throughput: {throughput:.2f} samples/sec" if isinstance(throughput, float) else f"- Throughput: {throughput}")
        print(f"- Peak Memory Allocated: {max_mem}")
        if gpu_util is not None:
            print(f"- GPU Utilization (during profiling): {gpu_util:.1f}%")
    elif "error" in profiling_data:
        print(f"- Profiling Error: {profiling_data['error']}")
    else:
        print("- Profiling data not available in report.")

except Exception as e:
    logging.exception("An error occurred during the TrainSense example.")
    print(f"\n--- ERROR --- \nAn error occurred: {e}")
    print("Please check the logs and ensure all components were initialized correctly.")

```

## Detailed Usage Examples

Here's how to use individual components:

### 1. Checking System Configuration

Get a snapshot of your hardware and software setup.

```python
from TrainSense.system_config import SystemConfig
from TrainSense.utils import print_section

sys_config = SystemConfig()
summary = sys_config.get_summary() # Get a concise summary

print_section("System Summary")
for key, value in summary.items():
    print(f"- {key.replace('_', ' ').title()}: {value}")

# You can also get the full detailed config
# full_config = sys_config.get_config()
# print("\nFull Config:", full_config)
```

### 2. Analyzing Your Model's Architecture

Understand the structure and complexity of your `nn.Module`.

```python
import torch.nn as nn
from TrainSense.arch_analyzer import ArchitectureAnalyzer
from TrainSense.utils import print_section

# Define or load your model
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

arch_analyzer = ArchitectureAnalyzer(model)
analysis = arch_analyzer.analyze() # Performs the analysis

print_section("Architecture Analysis")
print(f"- Total Parameters: {analysis.get('total_parameters', 0):,}")
print(f"- Trainable Parameters: {analysis.get('trainable_parameters', 0):,}")
print(f"- Layer Count: {analysis.get('layer_count', 'N/A')}")
print(f"- Primary Architecture Type: {analysis.get('primary_architecture_type', 'N/A')}")
print(f"- Complexity Category: {analysis.get('complexity_category', 'N/A')}")
print(f"- Estimated Input Shape: {analysis.get('estimated_input_shape', 'N/A')}") # Useful for profiler!
print(f"- Recommendation: {analysis.get('recommendation', 'N/A')}")

print("\n- Layer Types:")
for layer_type, count in analysis.get('layer_types_summary', {}).items():
    print(f"  - {layer_type}: {count}")
```

### 3. Getting Hyperparameter Recommendations

Check if your initial batch size, learning rate, and epochs make sense.

```python
from TrainSense.analyzer import TrainingAnalyzer
from TrainSense.system_config import SystemConfig # Needed for context
from TrainSense.arch_analyzer import ArchitectureAnalyzer # Needed for context
from TrainSense.utils import print_section
import torch.nn as nn # For dummy model

# --- Get Context (System & Model) ---
model = nn.Linear(10, 2) # Simple dummy model
sys_config = SystemConfig()
arch_analyzer = ArchitectureAnalyzer(model)
arch_info = arch_analyzer.analyze()
# -------------------------------------

# --- Define Current Hyperparameters ---
current_batch_size = 512
current_lr = 0.1
current_epochs = 5
# ------------------------------------

analyzer = TrainingAnalyzer(
    batch_size=current_batch_size,
    learning_rate=current_lr,
    epochs=current_epochs,
    system_config=sys_config, # Provide system context
    arch_info=arch_info       # Provide model context
)

print_section("Hyperparameter Checks")
recommendations = analyzer.check_hyperparameters()
print("Recommendations:")
for r in recommendations:
    print(f"- {r}")

print("\nSuggested Adjustments (Heuristic):")
adjustments = analyzer.auto_adjust()
for k, v in adjustments.items():
    original_val = getattr(analyzer, k) # Get original value from analyzer instance
    if v != original_val:
        print(f"- Adjust {k}: from {original_val} to {v}")
    else:
        print(f"- Keep {k}: {v} (unchanged)")
```

### 4. Profiling Model Performance

Measure speed and resource usage. **Requires a correct `input_shape`!**

```python
import torch
import torch.nn as nn
from TrainSense.model_profiler import ModelProfiler
from TrainSense.utils import print_section, format_bytes

# --- Define Model and Device ---
model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# -----------------------------

# --- Define Input Shape (Crucial!) ---
# Must match your model's expected input for a single batch
batch_size_for_profiling = 32
input_features = 64
input_shape = (batch_size_for_profiling, input_features)
# ------------------------------------

profiler = ModelProfiler(model, device=device)

print_section("Model Profiling")
print(f"Profiling on {device} with input shape: {input_shape}")

try:
    profile_results = profiler.profile_model(
        input_shape=input_shape,
        iterations=100,      # Number of inference runs for timing
        warmup=20,           # Warmup runs (ignored for timing)
        use_torch_profiler=True # Enable detailed profiling
    )

    if "error" in profile_results:
        print(f"\n!! Profiling Error: {profile_results['error']}")
    else:
        print(f"\n--- Performance ---")
        print(f"- Avg. Inference Time: {profile_results.get('avg_total_time_ms', 0):.3f} ms")
        print(f"- Throughput: {profile_results.get('throughput_samples_per_sec', 0):.1f} samples/sec")

        print(f"\n--- Memory ---")
        print(f"- Peak Memory Allocated: {profile_results.get('max_memory_allocated_formatted', 'N/A')}")
        if device.type == 'cuda':
             print(f"- Peak Memory Reserved (CUDA): {profile_results.get('max_memory_reserved_formatted', 'N/A')}")

        if profile_results.get('use_torch_profiler'):
            print(f"\n--- Detailed Profiler Stats (Averages) ---")
            cpu_perc = profile_results.get('avg_cpu_time_percent', 0)
            gpu_perc = profile_results.get('avg_gpu_time_percent', 0)
            print(f"- Device Utilization: CPU {cpu_perc:.1f}% | GPU {gpu_perc:.1f}%")
            print(f"- Avg CPU Time Total: {profile_results.get('avg_cpu_time_total_ms', 0):.3f} ms")
            if device.type == 'cuda':
                print(f"- Avg CUDA Time Total: {profile_results.get('avg_cuda_time_total_ms', 0):.3f} ms")
            # Optionally print the detailed operator table
            # print("\n--- Top Operators by Self CPU Time ---")
            # print(profile_results.get('profiler_top_ops_summary', 'Table not available.'))

except ValueError as ve:
     print(f"\n!! Input Shape Error: {ve}")
except Exception as e:
     print(f"\n!! An unexpected error occurred during profiling: {e}")


```
* **Important:** The profiler runs the model in `eval()` mode with `torch.no_grad()`. It measures inference performance. Training performance will differ due to backpropagation.
* **Memory:** `max_memory_allocated` tracks the peak PyTorch tensor memory. `max_memory_reserved` (CUDA only) tracks the total memory blocked by the CUDA memory manager. Profiler memory stats (`profiler_avg_...`) reflect usage *during* the specific profiled operations.

### 5. Monitoring GPU Status

Get real-time stats for your NVIDIA GPU(s). Requires `GPUtil` to be installed and functional.

```python
from TrainSense.gpu_monitor import GPUMonitor
from TrainSense.utils import print_section

try:
    gpu_monitor = GPUMonitor()
    print_section("GPU Status")

    if gpu_monitor.is_available():
        status_list = gpu_monitor.get_gpu_status()
        if status_list:
            for gpu_status in status_list:
                 print(f"GPU ID: {gpu_status.get('id', 'N/A')}")
                 print(f"  Name: {gpu_status.get('name', 'N/A')}")
                 print(f"  Load: {gpu_status.get('load', 0):.1f}%")
                 print(f"  Memory Util: {gpu_status.get('memory_utilization_percent', 0):.1f}% ({gpu_status.get('memory_used_mb', 0):.0f}/{gpu_status.get('memory_total_mb', 0):.0f} MB)")
                 print(f"  Temperature: {gpu_status.get('temperature_celsius', 'N/A')} C")
                 print("-" * 10)
            # Get a summary across all GPUs
            summary = gpu_monitor.get_status_summary()
            if summary and summary['count'] > 1:
                 print("\nOverall GPU Summary:")
                 print(f"- Average Load: {summary.get('avg_load_percent', 0):.1f}%")
                 print(f"- Average Memory Util: {summary.get('avg_memory_utilization_percent', 0):.1f}%")
                 print(f"- Max Temperature: {summary.get('max_temperature_celsius', 'N/A')} C")
        else:
             print("- GPUtil is available, but no GPUs were detected or status could not be retrieved.")
    else:
         print("- GPUtil library not installed or failed to initialize.")

except Exception as e:
    print(f"An error occurred while monitoring GPUs: {e}")

```

### 6. Getting Optimizer and Scheduler Suggestions

Leverage heuristics based on model size and type.

```python
import torch.nn as nn
from TrainSense.optimizer import OptimizerHelper
from TrainSense.arch_analyzer import ArchitectureAnalyzer # Needed for context
from TrainSense.utils import print_section

# --- Get Model Context ---
model = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True) # Example RNN
arch_analyzer = ArchitectureAnalyzer(model)
arch_info = arch_analyzer.analyze()
model_params = arch_info.get('total_parameters', 0)
model_arch_type = arch_info.get('primary_architecture_type', 'Unknown')
# -------------------------

print_section("Optimizer/Scheduler Suggestions")
print(f"Model Type: {model_arch_type}, Params: {model_params:,}")

suggested_optimizer = OptimizerHelper.suggest_optimizer(model_params, architecture_type=model_arch_type)
print(f"\nSuggested Optimizer: {suggested_optimizer}")

# Get the base name (e.g., "AdamW") for scheduler suggestion
base_optimizer_name = suggested_optimizer.split(" ")[0]
suggested_scheduler = OptimizerHelper.suggest_learning_rate_scheduler(base_optimizer_name)
print(f"Suggested Scheduler type for {base_optimizer_name}: {suggested_scheduler}")

suggested_initial_lr = OptimizerHelper.suggest_initial_learning_rate(model_arch_type, model_params)
print(f"Suggested Initial Learning Rate: {suggested_initial_lr:.1e}") # Format in scientific notation
```

### 7. Generating Heuristic Hyperparameters (`UltraOptimizer`)

Get a full starting set of parameters based on system, model, and basic data info.

```python
from TrainSense.ultra_optimizer import UltraOptimizer
from TrainSense.system_config import SystemConfig
from TrainSense.arch_analyzer import ArchitectureAnalyzer
from TrainSense.utils import print_section
import torch.nn as nn

# --- Get Context ---
model = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, 10)) # Moderate MLP
sys_config = SystemConfig()
config_summary = sys_config.get_summary() # UltraOptimizer uses the summary dict
arch_analyzer = ArchitectureAnalyzer(model)
arch_info = arch_analyzer.analyze()
# Provide some basic stats about your training data
data_stats = {"data_size": 150000, "num_classes": 10}
# ------------------

ultra_optimizer = UltraOptimizer(
    training_data_stats=data_stats,
    model_arch_stats=arch_info,
    system_config_summary=config_summary
)

print_section("Heuristic Parameter Set (UltraOptimizer)")
heuristic_result = ultra_optimizer.compute_heuristic_hyperparams()
params = heuristic_result.get("hyperparameters", {})
reasoning = heuristic_result.get("reasoning", {})

print("Generated Hyperparameters:")
for key, value in params.items():
    print(f"- {key}: {value}")

print("\nReasoning:")
for key, reason in reasoning.items():
    print(f"- {key}: {reason}")
```

### 8. Using the Logger

Configure logging to file and/or console.

```python
import logging
import os
from TrainSense.logger import TrainLogger, get_trainsense_logger

# --- Configure Logging ONCE (e.g., at the start of your script) ---
log_dir = "my_training_logs"
# if not os.path.exists(log_dir): # Logger creates dir now
#     os.makedirs(log_dir)

# Initialize the singleton logger instance with configuration
# This setup affects all subsequent calls to get_trainsense_logger()
logger_instance = TrainLogger(
    log_file=os.path.join(log_dir, "trainsense_run.log"),
    level=logging.DEBUG, # Log DEBUG and higher messages
    log_to_console=True,
    console_level=logging.INFO # Show INFO and higher on console
)
# ----------------------------------------------------------------

# --- Get the configured logger anywhere else in your code ---
logger = get_trainsense_logger()
# -----------------------------------------------------------

# --- Example Logging Calls ---
logger.debug("This is a detailed debug message.")
logger.info("Starting data preprocessing.")
logger.warning("Learning rate seems high, consider lowering.")
try:
    x = 1 / 0
except ZeroDivisionError:
    logger.error("Calculation failed due to division by zero.", exc_info=True) # Log exception info

logger.info("Example finished.")
```

## Interpreting the Output

*   **High CPU Usage / Low GPU Utilization:** Check your data loading pipeline (`DataLoader` `num_workers`, transforms), preprocessing steps, or operations that might be running heavily on the CPU, preventing the GPU from being fed quickly enough.
*   **High GPU Memory Usage (`max_memory_allocated`):** Your model or batch size might be too large for the GPU VRAM. Consider reducing batch size, using gradient accumulation, mixed-precision training (`torch.cuda.amp`), or model optimization techniques (pruning, quantization).
*   **Profiler Bottlenecks:** Look at the `profiler_top_ops_summary`. If specific operations (besides expected convolutions/linear layers) take a disproportionate amount of time, investigate if they can be optimized or replaced.
*   **Hyperparameter Warnings:** Pay attention to recommendations from `TrainingAnalyzer` regarding batch size, learning rate, and epochs. They are heuristics but often point towards potential issues like instability (high LR), slow convergence (low LR), or overfitting/underfitting (epochs).
*   **Architecture Complexity:** Use the `complexity_category` and parameter count from `ArchitectureAnalyzer` to gauge resource requirements. "Very Complex / Large" models will demand significant GPU memory and time.

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request. (Add more details here if you have specific contribution guidelines).

## License

This project is licensed under the MIT License - see the LICENSE file for details.
