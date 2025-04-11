# TrainSense/model_profiler.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import time
import logging
from typing import Tuple, Dict, Any, Optional, Union, Callable, Iterable, List
# Import PyTorch profiler components
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from torch.optim import Optimizer
from torch.utils.data import DataLoader # For type hinting
from itertools import cycle # To handle DataLoaders of varying lengths robustly

# Import utility functions from the TrainSense package
try:
    from .utils import format_bytes, format_time, validate_positive_integer
except ImportError:
    # Fallback for tests if utils are not found via relative import
    from utils import format_bytes, format_time, validate_positive_integer


# Get a logger instance specific to this module
logger = logging.getLogger(__name__)

class ModelProfiler:
    """
    Profiles PyTorch models for inference speed, training step duration,
    and resource utilization using basic timing and the integrated `torch.profiler`.

    Provides methods to profile:
    - `profile_model`: For inference performance analysis.
    - `profile_training_step`: For analyzing a full training step including data loading.
    """
    def __init__(self, model: nn.Module, device: Optional[Union[str, torch.device]] = None):
        """
        Initializes the ModelProfiler.

        Args:
            model (nn.Module): The PyTorch model instance to profile.
            device (Optional[Union[str, torch.device]]): The target device ('cpu', 'cuda', 'cuda:0', etc.)
                to run profiling on. If None, it autodetects CUDA availability and uses 'cuda'
                if possible, otherwise falls back to 'cpu'.

        Raises:
            TypeError: If the input 'model' is not an instance of torch.nn.Module.
        """
        if not isinstance(model, nn.Module):
            raise TypeError("Input 'model' must be an instance of torch.nn.Module.")

        self.model = model
        # Determine and validate the target device
        self.device = self._resolve_device(device)
        # Attempt to move the model to the target device, log warning on failure
        try:
            self.model.to(self.device)
        except Exception as e:
            logger.warning(f"Could not move model to target device {self.device}: {e}")

        logger.info(f"ModelProfiler initialized for model {type(model).__name__} on device '{self.device}'")

    def _resolve_device(self, device: Optional[Union[str, torch.device]]) -> torch.device:
        """
        Determines and validates the torch.device to use for profiling.

        Args:
            device (Optional[Union[str, torch.device]]): The user-specified device preference.

        Returns:
            torch.device: The validated torch.device object to be used.
        """
        if device:
            try:
                # Convert string to torch.device object if necessary
                resolved_device = torch.device(device)
                # Check CUDA availability if a CUDA device is specified
                if resolved_device.type == 'cuda':
                    if not torch.cuda.is_available():
                         logger.warning(f"Device specified as '{device}' but CUDA is not available. Falling back to CPU.")
                         return torch.device("cpu")
                    # Try a simple operation to ensure the specific CUDA device is accessible
                    _ = torch.tensor([1.0], device=resolved_device)
                # Return the validated device
                logger.info(f"Using specified device: '{resolved_device}'")
                return resolved_device
            except RuntimeError as e:
                 logger.warning(f"Specified device '{device}' caused a RuntimeError ({e}). Falling back.")
                 # Fallback logic: try default CUDA if available, otherwise CPU
                 if torch.cuda.is_available():
                      logger.warning("Falling back to default CUDA device.")
                      return torch.device("cuda")
                 else:
                      logger.warning("Falling back to CPU device.")
                      return torch.device("cpu")
            except Exception as e:
                 logger.warning(f"Failed to resolve specified device '{device}': {e}. Falling back.")
                 if torch.cuda.is_available(): return torch.device("cuda")
                 else: return torch.device("cpu")
        # Auto-detection if no device specified
        elif torch.cuda.is_available():
            logger.info("CUDA available, selecting default CUDA device.")
            return torch.device("cuda")
        else:
             logger.info("CUDA not available, selecting CPU device.")
             return torch.device("cpu")

    def _generate_dummy_input(self, input_shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Generates a dummy input tensor with the specified shape and dtype on the target device.

        Args:
            input_shape (Tuple[int, ...]): The desired shape of the input tensor (e.g., (Batch, Channels, H, W)).
            dtype (torch.dtype): The data type for the tensor (e.g., torch.float32, torch.float16).

        Returns:
            torch.Tensor: The generated dummy tensor moved to the profiler's target device.

        Raises:
            ValueError: If tensor creation fails.
        """
        logger.debug(f"Generating dummy input tensor with shape: {input_shape}, dtype: {dtype}")
        try:
            # Generate on CPU first, then move, as it can be more reliable in some multi-GPU setups.
            tensor_cpu = torch.randn(*input_shape, dtype=dtype)
            return tensor_cpu.to(self.device)
        except Exception as e:
            logger.error(f"Failed to create dummy tensor with shape {input_shape} on device {self.device}: {e}", exc_info=True)
            raise ValueError(f"Failed to create dummy input with shape {input_shape}: {e}") from e

    def _generate_dummy_target(self, output: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        """
        Generates a dummy target tensor compatible with the model's output and a given criterion.
        This uses heuristics for common loss functions. For complex cases, provide a custom `target_generator`.

        Args:
            output (torch.Tensor): The output tensor from the model's forward pass.
            criterion (nn.Module): The loss function instance being used.

        Returns:
            torch.Tensor: A dummy target tensor on the same device as the output.

        Raises:
            ValueError: If target generation fails.
        """
        target_shape = output.shape
        output_device = output.device # Ensure target is on the same device as output
        logger.debug(f"Generating dummy target for output shape {target_shape} and criterion {type(criterion).__name__} on device {output_device}")
        try:
            # --- Heuristics for Common Loss Types ---
            # Classification Losses (expecting class indices)
            if isinstance(criterion, (nn.CrossEntropyLoss, nn.NLLLoss)):
                num_classes = output.shape[1] if len(output.shape) > 1 else None # Assume C is dim 1
                if num_classes is None: raise ValueError("Cannot determine number of classes for CrossEntropy/NLL loss.")

                # Standard case: (Batch, Classes) output -> (Batch,) target
                if len(output.shape) == 2:
                     return torch.randint(0, num_classes, (target_shape[0],), device=output_device, dtype=torch.long)
                # Segmentation case: (Batch, Classes, H, W) output -> (Batch, H, W) target
                elif len(output.shape) == 4 and output.shape[1] == num_classes:
                     logger.debug("Assuming segmentation-like target shape based on loss and 4D output.")
                     return torch.randint(0, num_classes, (target_shape[0], *target_shape[2:]), device=output_device, dtype=torch.long)
                # Other cases (e.g., 3D for video/volumetric) - make a guess or warn
                else:
                     logger.warning(f"Ambiguous output shape {output.shape} for {type(criterion).__name__}. Generating 1D target (Batch,). Provide target_generator if incorrect.")
                     return torch.randint(0, num_classes, (target_shape[0],), device=output_device, dtype=torch.long) # Fallback

            # Regression Losses (expecting targets with same shape as output)
            elif isinstance(criterion, (nn.MSELoss, nn.L1Loss, nn.SmoothL1Loss)):
                return torch.randn_like(output)

            # Binary Classification Losses (expecting targets with same shape, values 0 or 1)
            elif isinstance(criterion, (nn.BCELoss, nn.BCEWithLogitsLoss)):
                 # Use rand for [0, 1) range, then convert to float (required by BCE*)
                 return torch.rand_like(output).float() # Ensure float type

            # Default: Generate zeros of the same shape (might not work for all losses)
            else:
                 logger.warning(f"Unsupported criterion type {type(criterion).__name__} for automatic target generation. Using zeros_like. Provide target_generator if needed.")
                 return torch.zeros_like(output)

        except Exception as e:
            logger.error(f"Failed to generate dummy target for output shape {output.shape} and criterion {type(criterion).__name__}: {e}", exc_info=True)
            raise ValueError(f"Failed to generate dummy target for criterion {type(criterion).__name__}.") from e


    @torch.no_grad() # Disable gradient calculations for inference profiling
    def profile_model(self,
                      input_shape: Tuple[int, ...],
                      iterations: int = 50,
                      warmup: int = 10,
                      use_torch_profiler: bool = True,
                      profiler_activities: Optional[list] = None,
                      profiler_sort_by: str = "self_cpu_time_total",
                      profiler_row_limit: int = 10,
                      input_dtype: torch.dtype = torch.float32
                     ) -> Dict[str, Any]:
        """
        Profiles the model's inference performance (forward pass only).

        Args:
            input_shape (Tuple[int, ...]): The shape of a single input batch (e.g., (1, 3, 224, 224)).
            iterations (int): Number of inference iterations to time after warmup.
            warmup (int): Number of untimed warmup iterations to run before profiling.
            use_torch_profiler (bool): Whether to use the detailed `torch.profiler` for operator-level analysis.
            profiler_activities (Optional[list]): Activities for `torch.profiler` (e.g., [ProfilerActivity.CPU, ProfilerActivity.CUDA]).
                                                  Autodetected based on device if None.
            profiler_sort_by (str): Key to sort the `torch.profiler` summary table by (e.g., 'cpu_time_total', 'cuda_time_total').
            profiler_row_limit (int): Number of rows to display in the `torch.profiler` summary table.
            input_dtype (torch.dtype): Data type for the dummy input tensor (e.g., torch.float32, torch.float16).

        Returns:
            Dict[str, Any]: A dictionary containing profiling results:
                - 'profiling_type': 'inference'
                - 'input_shape', 'input_dtype', 'device', 'iterations', 'warmup', 'use_torch_profiler'
                - 'avg_total_time_ms': Average time per iteration (ms).
                - 'throughput_batches_per_sec': Batches processed per second.
                - 'throughput_samples_per_sec': Samples processed per second (based on batch_size).
                - 'total_timed_duration_sec': Total time for timed iterations.
                - 'max_memory_allocated_mb': Peak GPU memory allocated during profiling (MB, CUDA only).
                - 'max_memory_allocated_formatted': Human-readable peak allocated memory.
                - 'max_memory_reserved_mb': Peak GPU memory reserved during profiling (MB, CUDA only).
                - 'max_memory_reserved_formatted': Human-readable peak reserved memory.
                - 'profiler_data': Dictionary with detailed results from `torch.profiler` (if used).
                    - Contains averages (CPU/CUDA time, memory) and 'profiler_top_ops_summary' table.
                - 'profiler_error': Error message from `torch.profiler` run/analysis, if any.
                - 'error': General error message if profiling failed (e.g., OOM).
                - 'error_details': Specific error details (e.g., OOM message).
        """
        # --- Input Validation ---
        validate_positive_integer(iterations, "Profiling iterations", allow_zero=False)
        validate_positive_integer(warmup, "Profiling warmup iterations", allow_zero=True)
        if not isinstance(input_shape, tuple) or not all(isinstance(d, int) and d > 0 for d in input_shape):
            raise ValueError(f"input_shape must be a tuple of positive integers, got {input_shape}.")
        if not isinstance(input_dtype, torch.dtype):
            raise TypeError(f"input_dtype must be a torch.dtype, got {type(input_dtype)}")

        logger.info(f"[Inference Profiling] Starting - Input: {input_shape}, Dtype: {input_dtype}, Iters: {iterations}, Warmup: {warmup}, Device: {self.device}, TorchProfiler: {use_torch_profiler}")

        # --- Setup ---
        original_mode_is_train = self.model.training # Store original mode
        self.model.eval() # <<< Set model to evaluation mode for inference profiling
        try:
             dummy_input = self._generate_dummy_input(input_shape, dtype=input_dtype)
        except ValueError as e:
             # Restore original mode before raising
             self.model.train(mode=original_mode_is_train)
             raise e # Re-raise the exception

        # Initialize results dictionary
        results: Dict[str, Any] = {
            "profiling_type": "inference", "input_shape": input_shape, "input_dtype": str(input_dtype),
            "device": str(self.device), "iterations": iterations, "warmup": warmup,
            "use_torch_profiler": use_torch_profiler, "error": None, "error_details": None,
            "profiler_data": None, "profiler_error": None,
            "avg_total_time_ms": 0.0, "throughput_batches_per_sec": 0.0, "throughput_samples_per_sec": 0.0,
            "total_timed_duration_sec": 0.0, "max_memory_allocated_b": None, "max_memory_reserved_b": None,
            "max_memory_allocated_mb": None, "max_memory_reserved_mb": None,
            "max_memory_allocated_formatted": "N/A", "max_memory_reserved_formatted": "N/A"
        }

        # Reset CUDA memory stats if applicable
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache() # Clear cache before starting

        prof = None # Define profiler context manager variable outside try block

        try:
             # --- Warmup Phase ---
             logger.debug("[Inference Profiling] Running warmup...")
             for _ in range(warmup):
                 _ = self.model(dummy_input)
             # Synchronize CUDA device after warmup if needed
             if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
             logger.debug("[Inference Profiling] Warmup complete.")

             # --- Simple Timing Loop ---
             logger.debug("[Inference Profiling] Running timed iterations for basic metrics...")
             start_time = time.perf_counter()
             for _ in range(iterations):
                 _ = self.model(dummy_input)
             # Synchronize CUDA device after timed iterations
             if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
             end_time = time.perf_counter()

             total_time_sec = end_time - start_time
             avg_total_time_sec = total_time_sec / iterations if iterations > 0 else 0
             throughput_batches = iterations / total_time_sec if total_time_sec > 0 else float('inf')
             batch_size = input_shape[0] # Get batch size from input shape

             results["avg_total_time_ms"] = avg_total_time_sec * 1000
             results["throughput_batches_per_sec"] = throughput_batches
             results["throughput_samples_per_sec"] = throughput_batches * batch_size # Samples/sec
             results["total_timed_duration_sec"] = total_time_sec
             logger.info(f"[Inference Profiling] Basic timing: Avg time={results['avg_total_time_ms']:.3f} ms/batch, Throughput={results['throughput_samples_per_sec']:.1f} samples/sec")

             # --- Detailed Profiling using torch.profiler (Optional) ---
             if use_torch_profiler:
                 logger.info("[Inference Profiling] Running detailed profiling with torch.profiler...")
                 # Determine profiler activities based on device
                 if profiler_activities is None:
                      profiler_activities = [ProfilerActivity.CPU]
                      if self.device.type == 'cuda':
                           profiler_activities.append(ProfilerActivity.CUDA)
                           logger.debug(f"Profiler activities: {profiler_activities}")

                 # Define a schedule for the profiler (wait, warmup, active, repeat)
                 # Using fewer iterations for profiler run than basic timing to limit overhead
                 profile_active = min(iterations, 5) # Number of active recording steps
                 profile_warmup = min(warmup, 2)    # Number of warmup steps within profiler
                 wait = 1                           # Steps to wait before starting warmup
                 repeat = 1                         # Number of times to repeat the cycle
                 prof_schedule = schedule(wait=wait, warmup=profile_warmup, active=profile_active, repeat=repeat)
                 num_profiler_steps = (wait + profile_warmup + profile_active) * repeat

                 try:
                     # Use the profiler context manager
                     with profile(activities=profiler_activities,
                                  record_shapes=True,        # Record tensor shapes
                                  profile_memory=True,       # Enable memory profiling (CPU and CUDA)
                                  with_stack=False,          # Disable Python stack tracing (reduces overhead)
                                  schedule=prof_schedule,    # Use the defined schedule
                                  on_trace_ready=None        # Use default handler (needed for schedule)
                                 ) as prof:
                         # Run the model within the profiler context for the required number of steps
                         for i in range(num_profiler_steps):
                             with record_function(f"inference_iteration_{i}"): # Add labels to profiler timeline
                                 _ = self.model(dummy_input)
                             prof.step() # Signal the profiler to advance according to the schedule

                     # --- Analyze Profiler Results ---
                     # (Analysis happens after exiting the 'with' block)
                     logger.info("[Inference Profiling] Torch profiler run complete. Analyzing results...")
                     key_averages = prof.key_averages() # Get aggregated statistics
                     results["profiler_data"] = {}
                     if not key_averages:
                         logger.warning("[Inference Profiling] Torch profiler recorded no events.")
                     else:
                         # Extract key metrics from the profiler averages
                         total_avg = key_averages.total_average()
                         results["profiler_data"]["total_events_averaged"] = len(key_averages)

                         # CPU Time
                         cpu_time_total_us = total_avg.cpu_time_total
                         self_cpu_time_total_us = total_avg.self_cpu_time_total
                         results["profiler_data"]["avg_cpu_time_total_ms"] = cpu_time_total_us / 1000
                         results["profiler_data"]["avg_self_cpu_time_total_ms"] = self_cpu_time_total_us / 1000

                         # CUDA Time (if applicable)
                         cuda_time_total_us = getattr(total_avg, 'cuda_time_total', 0)
                         self_cuda_time_total_us = getattr(total_avg, 'self_cuda_time_total', 0)
                         results["profiler_data"]["avg_cuda_time_total_ms"] = cuda_time_total_us / 1000
                         results["profiler_data"]["avg_self_cuda_time_total_ms"] = self_cuda_time_total_us / 1000

                         # Device Utilization Percentages
                         combined_time_us = cpu_time_total_us + cuda_time_total_us
                         results["profiler_data"]["avg_cpu_time_percent"] = (cpu_time_total_us / combined_time_us * 100) if combined_time_us > 0 else 0
                         results["profiler_data"]["avg_gpu_time_percent"] = (cuda_time_total_us / combined_time_us * 100) if combined_time_us > 0 else 0

                         # Memory Usage from Profiler (Average over profiled steps)
                         cpu_mem_usage_b = getattr(total_avg, 'cpu_memory_usage', 0)
                         cuda_mem_usage_b = getattr(total_avg, 'cuda_memory_usage', 0) if self.device.type == 'cuda' else 0
                         results["profiler_data"]["profiler_avg_cpu_memory_usage_b"] = cpu_mem_usage_b
                         results["profiler_data"]["profiler_avg_cpu_memory_usage_formatted"] = format_bytes(cpu_mem_usage_b)
                         results["profiler_data"]["profiler_avg_gpu_memory_usage_b"] = cuda_mem_usage_b
                         results["profiler_data"]["profiler_avg_gpu_memory_usage_formatted"] = format_bytes(cuda_mem_usage_b) if cuda_mem_usage_b > 0 else "N/A"

                         # Generate Summary Table
                         sort_key = profiler_sort_by if profiler_sort_by else "self_cpu_time_total"
                         try:
                            table_str = prof.key_averages().table(sort_by=sort_key, row_limit=profiler_row_limit)
                            results["profiler_data"]["profiler_top_ops_summary"] = table_str
                            logger.debug(f"[Inference Profiling] Top {profiler_row_limit} operators by {sort_key}:\n{table_str}")
                         except KeyError as ke:
                             logger.warning(f"Profiler sort key '{sort_key}' not found (KeyError: {ke}). Defaulting to 'self_cpu_time_total'.")
                             try: # Try default sort key
                                 table_str = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=profiler_row_limit)
                                 results["profiler_data"]["profiler_top_ops_summary"] = table_str
                             except Exception as table_err_fb:
                                 err_msg = f"Error generating profiler table even with default sort: {table_err_fb}"
                                 logger.error(err_msg)
                                 results["profiler_data"]["profiler_top_ops_summary"] = err_msg
                         except Exception as table_err:
                             err_msg = f"Error generating profiler table: {table_err}"
                             logger.error(err_msg, exc_info=True)
                             results["profiler_data"]["profiler_top_ops_summary"] = err_msg
                         # --- End Profiler Analysis ---

                 except Exception as prof_err:
                      logger.error(f"[Inference Profiling] Failed during torch.profiler execution/analysis: {prof_err}", exc_info=True)
                      results["profiler_error"] = f"Profiler run/analysis failed: {prof_err}"
                 # 'prof' context is automatically handled by 'with' statement


             # --- Capture Peak Memory Overall (after all profiling) ---
             if self.device.type == 'cuda':
                 results["max_memory_allocated_b"] = torch.cuda.max_memory_allocated(self.device)
                 results["max_memory_reserved_b"] = torch.cuda.max_memory_reserved(self.device)
                 results["max_memory_allocated_mb"] = results["max_memory_allocated_b"] / (1024**2)
                 results["max_memory_reserved_mb"] = results["max_memory_reserved_b"] / (1024**2)
                 results["max_memory_allocated_formatted"] = format_bytes(results["max_memory_allocated_b"])
                 results["max_memory_reserved_formatted"] = format_bytes(results["max_memory_reserved_b"])
                 logger.info(f"[Inference Profiling] Peak CUDA Memory: Allocated={results['max_memory_allocated_formatted']}, Reserved={results['max_memory_reserved_formatted']}")
             else: # CPU Profiling Memory (less precise)
                 # Use profiler average as an estimate if available
                 if results.get("profiler_data"):
                     cpu_mem_b = results["profiler_data"].get("profiler_avg_cpu_memory_usage_b")
                     if cpu_mem_b is not None:
                        results["max_memory_allocated_b"] = cpu_mem_b
                        results["max_memory_allocated_mb"] = cpu_mem_b / (1024**2)
                        results["max_memory_allocated_formatted"] = format_bytes(cpu_mem_b) + " (prof avg est.)"
                 # max_memory_reserved is not applicable/easily measurable for CPU via torch profiler

        # --- Error Handling ---
        except torch.cuda.OutOfMemoryError as oom_err:
            err_msg = f"CUDA OutOfMemoryError during inference profiling."
            logger.error(err_msg, exc_info=False) # Log OOM without full stack trace usually
            results["error"] = "CUDA OutOfMemoryError"
            results["error_details"] = str(oom_err)
            # Try to get peak memory usage *at the point of OOM*
            if self.device.type == 'cuda':
                try:
                     # max_memory_allocated might reflect the peak before the failed allocation
                     mem_allocated_oom = torch.cuda.max_memory_allocated(self.device)
                     results["max_memory_allocated_b"] = mem_allocated_oom
                     results["max_memory_allocated_mb"] = mem_allocated_oom / (1024**2)
                     results["max_memory_allocated_formatted"] = format_bytes(mem_allocated_oom)
                     results["memory_at_oom_approx_mb"] = results["max_memory_allocated_mb"] # Add specific field
                     logger.error(f"OOM occurred. Peak memory allocated before failure: {results['max_memory_allocated_formatted']}")
                except Exception as mem_err:
                     logger.error(f"Could not get memory stats after OOM: {mem_err}")
                finally:
                     torch.cuda.empty_cache() # Attempt to release memory
        except Exception as e:
            err_msg = f"General error during inference profiling: {e}"
            logger.error(err_msg, exc_info=True)
            # Avoid overwriting a more specific OOM error
            if results["error"] is None:
                 results["error"] = "General profiling error"
                 results["error_details"] = str(e)
        finally:
             # --- Cleanup ---
             # Restore the model's original training mode
             self.model.train(mode=original_mode_is_train)
             logger.debug(f"Restored model training mode to: {original_mode_is_train}")
             # Optional: Clean up dummy input? (Usually handled by garbage collection)
             # del dummy_input
             if self.device.type == 'cuda': torch.cuda.empty_cache()

        return results


    def profile_training_step(self,
                              data_loader: Iterable,
                              criterion: nn.Module,
                              optimizer: Optimizer,
                              target_generator: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                              input_dtype: torch.dtype = torch.float32,
                              iterations: int = 10,
                              warmup: int = 3,
                              use_torch_profiler: bool = True,
                              profiler_activities: Optional[list] = None,
                              profiler_sort_by: str = "self_cpu_time_total",
                              profiler_row_limit: int = 15
                             ) -> Dict[str, Any]:
        """
        Profiles a full training step, including data loading, forward, loss, backward, and optimizer step.
        Provides a time breakdown of these stages.

        Args:
            data_loader (Iterable): An iterable (like `torch.utils.data.DataLoader`) that yields batches.
                                    Assumes batches are tuples/lists (input, target) or single input tensors.
                                    Uses `itertools.cycle` to handle short datasets during profiling.
            criterion (nn.Module): The loss function instance.
            optimizer (Optimizer): The optimizer instance.
            target_generator (Optional[Callable]): A function `target = func(output)` used to generate
                                                   targets dynamically if the `data_loader` yields only inputs,
                                                   or if the criterion requires a specific target format derived
                                                   from the model output. If None, attempts automatic generation
                                                   based on criterion type.
            input_dtype (torch.dtype): Expected data type for the model input tensors in the batch.
            iterations (int): Number of training steps to profile after warmup.
            warmup (int): Number of untimed warmup training steps.
            use_torch_profiler (bool): Whether to use the detailed `torch.profiler`.
            profiler_activities (Optional[list]): Activities for `torch.profiler`. Autodetected.
            profiler_sort_by (str): Key to sort the `torch.profiler` summary table by.
            profiler_row_limit (int): Number of rows to show in the `torch.profiler` summary table.

        Returns:
            Dict[str, Any]: A dictionary containing profiling results:
                - 'profiling_type': 'training_step'
                - 'input_dtype', 'device', 'iterations_requested', 'iterations_completed', 'warmup'
                - 'use_torch_profiler', 'optimizer_type', 'criterion_type'
                - 'avg_step_time_ms': Average time for the complete step.
                - 'avg_data_fetch_time_ms', 'avg_data_prep_time_ms', 'avg_data_total_load_time_ms'
                - 'avg_forward_time_ms', 'avg_loss_time_ms', 'avg_backward_time_ms', 'avg_optimizer_time_ms'
                - 'percent_time_data_fetch', ..., 'percent_time_optimizer': Percentage breakdown.
                - 'max_memory_allocated_mb', 'max_memory_allocated_formatted' (CUDA only peak)
                - 'max_memory_reserved_mb', 'max_memory_reserved_formatted' (CUDA only peak)
                - 'profiler_data': Detailed results from `torch.profiler` (if used).
                - 'profiler_error': Error from `torch.profiler` run/analysis, if any.
                - 'error': General error message if profiling failed (e.g., OOM, DataLoader issue).
                - 'warning': Non-fatal warnings (e.g., DataLoader exhausted).
        """
        # --- Input Validation ---
        validate_positive_integer(iterations, "Training profiling iterations", allow_zero=False)
        validate_positive_integer(warmup, "Training profiling warmup", allow_zero=True)
        if not isinstance(input_dtype, torch.dtype):
            raise TypeError(f"input_dtype must be a torch.dtype, got {type(input_dtype)}")
        if not hasattr(data_loader, '__iter__') or not hasattr(data_loader, '__next__'):
            # Check if it's iterable using `collections.abc.Iterable`? More robust?
             logger.warning("data_loader might not be a standard Iterable. Using itertools.cycle, errors may occur.")
        if not isinstance(criterion, nn.Module): raise TypeError("criterion must be an nn.Module.")
        if not isinstance(optimizer, Optimizer): raise TypeError("optimizer must be a torch.optim.Optimizer.")

        logger.info(f"[Training Profiling] Starting - Iters: {iterations}, Warmup: {warmup}, Device: {self.device}, TorchProfiler: {use_torch_profiler}")

        # --- Setup ---
        original_mode_is_train = self.model.training
        self.model.train() # <<< Set model to training mode
        try:
             # Move criterion to the target device if possible
             criterion.to(self.device)
        except Exception as e:
             logger.warning(f"Could not move criterion {type(criterion).__name__} to device {self.device}: {e}")

        # Use itertools.cycle for robust iteration, especially over potentially short dataloaders
        try:
            cycled_loader = cycle(data_loader)
            # Try getting one batch to check loader validity early and potentially get shapes
            logger.debug("Fetching one batch to validate DataLoader...")
            first_batch = next(cycled_loader)
            logger.debug("DataLoader validation successful.")
            # Recreate cycle to ensure we start from the beginning for warmup + profiling runs
            cycled_loader = cycle(data_loader)
        except StopIteration:
             logger.error("DataLoader provided is empty.")
             self.model.train(mode=original_mode_is_train) # Restore mode
             return {"error": "DataLoader is empty.", "profiling_type": "training_step", "iterations_completed": 0}
        except Exception as loader_err:
             logger.error(f"Failed to get first batch from data_loader using cycle: {loader_err}", exc_info=True)
             self.model.train(mode=original_mode_is_train) # Restore mode
             return {"error": f"DataLoader error: {loader_err}", "profiling_type": "training_step", "iterations_completed": 0}

        # Initialize results dictionary
        results: Dict[str, Any] = {
            "profiling_type": "training_step", "input_dtype": str(input_dtype), "device": str(self.device),
            "iterations_requested": iterations, "iterations_completed": 0, "warmup": warmup,
            "use_torch_profiler": use_torch_profiler, "optimizer_type": type(optimizer).__name__,
            "criterion_type": type(criterion).__name__, "error": None, "warning": None,
            "profiler_data": None, "profiler_error": None,
            "total_profiled_duration_sec": 0.0, "avg_step_time_ms": 0.0,
            # Time breakdown averages (ms)
            "avg_data_fetch_time_ms": 0.0, "avg_data_prep_time_ms": 0.0, "avg_data_total_load_time_ms": 0.0,
            "avg_forward_time_ms": 0.0, "avg_loss_time_ms": 0.0, "avg_backward_time_ms": 0.0,
            "avg_optimizer_time_ms": 0.0,
            # Percentage breakdown
            "percent_time_data_fetch": 0.0, "percent_time_data_prep": 0.0, "percent_time_data_total_load": 0.0,
            "percent_time_forward": 0.0, "percent_time_loss": 0.0, "percent_time_backward": 0.0,
            "percent_time_optimizer": 0.0,
            # Memory
            "max_memory_allocated_b": None, "max_memory_allocated_mb": None, "max_memory_allocated_formatted": "N/A",
            "max_memory_reserved_b": None, "max_memory_reserved_mb": None, "max_memory_reserved_formatted": "N/A"
        }

        # Reset CUDA memory stats if applicable
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()

        # Lists to store detailed timing info for each step
        step_times_sec: List[float] = []
        data_fetch_times_sec: List[float] = []
        data_prep_times_sec: List[float] = []
        forward_times_sec: List[float] = []
        loss_times_sec: List[float] = []
        backward_times_sec: List[float] = []
        optimizer_times_sec: List[float] = []

        prof = None # Define profiler context variable outside try block

        try:
            # --- Warmup Phase ---
            logger.debug("[Training Profiling] Running warmup...")
            for wu_i in range(warmup):
                 try:
                     # Fetch data
                     batch = next(cycled_loader)
                     # Prepare inputs and targets (move to device)
                     if isinstance(batch, (list, tuple)):
                         inputs = batch[0].to(self.device, dtype=input_dtype, non_blocking=True)
                         # Handle target: use provided or generate
                         targets_cpu = batch[1] if len(batch) > 1 else None
                         if targets_cpu is None: # Generate if not provided
                             with torch.no_grad(): outputs_tmp = self.model(inputs)
                             targets_cpu = target_generator(outputs_tmp) if target_generator else self._generate_dummy_target(outputs_tmp, criterion)
                         targets = targets_cpu.to(self.device, non_blocking=True)
                     else: # Assume batch is just input
                         inputs = batch.to(self.device, dtype=input_dtype, non_blocking=True)
                         # Generate target
                         with torch.no_grad(): outputs_tmp = self.model(inputs)
                         targets = (target_generator(outputs_tmp) if target_generator else self._generate_dummy_target(outputs_tmp, criterion)).to(self.device)

                     # Perform a full training step
                     optimizer.zero_grad(set_to_none=True) # Use set_to_none=True for potential memory savings
                     outputs = self.model(inputs)
                     loss = criterion(outputs, targets)
                     loss.backward()
                     optimizer.step()
                 except StopIteration:
                     # This shouldn't happen with cycle() unless the original loader was empty
                     logger.error(f"[Training Profiling] DataLoader exhausted unexpectedly during warmup {wu_i+1}. Was the loader initially empty?")
                     results["error"] = "DataLoader exhausted during warmup (potentially empty)."
                     raise # Stop profiling
                 except Exception as wu_err:
                     logger.error(f"Error during warmup step {wu_i+1}: {wu_err}", exc_info=True)
                     results["error"] = f"Error during warmup step: {wu_err}"
                     raise # Stop profiling

            # Synchronize after warmup if using CUDA
            if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
            logger.debug("[Training Profiling] Warmup complete.")

            # --- Timed/Profiled Iterations ---
            logger.debug(f"[Training Profiling] Running {iterations} profiled iterations...")

            # Setup torch.profiler if enabled
            profiler_instance = None
            if use_torch_profiler:
                if profiler_activities is None:
                    profiler_activities = [ProfilerActivity.CPU]
                    if self.device.type == 'cuda': profiler_activities.append(ProfilerActivity.CUDA)
                # Define schedule - shorter active period than basic timing
                profile_active = min(iterations, 5)
                profile_warmup = min(max(0, warmup - 1), 2) # Use fewer warmup steps inside profiler
                wait = 0 # Start immediately after profiler warmup
                repeat = 1
                prof_schedule = schedule(wait=wait, warmup=profile_warmup, active=profile_active, repeat=repeat)
                num_profiler_steps = (wait + profile_warmup + profile_active) * repeat
                logger.info(f"[Training Profiling] Profiler schedule: wait={wait}, warmup={profile_warmup}, active={profile_active}")

                profiler_instance = profile(
                    activities=profiler_activities, record_shapes=True,
                    profile_memory=True, with_stack=False, schedule=prof_schedule
                )
                prof = profiler_instance.__enter__() # Manually enter context

            total_start_time = time.perf_counter()
            actual_iterations_run = 0

            # --- Main Profiling Loop ---
            for i in range(iterations):
                iter_start_time = time.perf_counter()
                inputs, targets = None, None # Reset for safety

                try:
                    # --- 1. Data Fetching ---
                    t0 = time.perf_counter()
                    with record_function("train::data_fetch"): # Label for profiler
                        batch = next(cycled_loader)
                    t1 = time.perf_counter()
                    data_fetch_times_sec.append(t1 - t0)

                    # --- 2. Data Preparation & Move to Device ---
                    with record_function("train::data_prep_move"):
                        if isinstance(batch, (list, tuple)):
                            inputs = batch[0].to(device=self.device, dtype=input_dtype, non_blocking=True)
                            targets_cpu = batch[1] if len(batch) > 1 else None
                            if targets_cpu is None:
                                with torch.no_grad(): outputs_tmp = self.model(inputs) # Need dummy forward if target depends on output
                                targets_cpu = target_generator(outputs_tmp) if target_generator else self._generate_dummy_target(outputs_tmp, criterion)
                            targets = targets_cpu.to(device=self.device, non_blocking=True)
                        else: # Assume batch is input only
                            inputs = batch.to(device=self.device, dtype=input_dtype, non_blocking=True)
                            with torch.no_grad(): outputs_tmp = self.model(inputs)
                            targets = (target_generator(outputs_tmp) if target_generator else self._generate_dummy_target(outputs_tmp, criterion)).to(device=self.device)
                    t2 = time.perf_counter()
                    data_prep_times_sec.append(t2 - t1)

                    # Check if data loading was successful
                    if inputs is None or targets is None:
                        raise RuntimeError(f"Failed to prepare inputs or targets at iteration {i}.")

                    # --- 3. Forward Pass ---
                    with record_function("train::forward"):
                        outputs = self.model(inputs)
                    t3 = time.perf_counter()
                    forward_times_sec.append(t3 - t2)

                    # --- 4. Loss Calculation ---
                    with record_function("train::loss"):
                        loss = criterion(outputs, targets)
                    t4 = time.perf_counter()
                    loss_times_sec.append(t4 - t3)

                    # --- 5. Backward Pass ---
                    # Zero gradients before backward
                    optimizer.zero_grad(set_to_none=True)
                    with record_function("train::backward"):
                        loss.backward()
                    t5 = time.perf_counter()
                    backward_times_sec.append(t5 - t4)

                    # --- 6. Optimizer Step ---
                    with record_function("train::optimizer_step"):
                        optimizer.step()
                    t6 = time.perf_counter()
                    optimizer_times_sec.append(t6 - t5)

                # --- Error Handling within Loop ---
                except StopIteration:
                    # Should not happen with cycle unless original loader was empty (handled earlier)
                    # but catch defensively.
                    logger.warning(f"[Training Profiling] DataLoader cycle unexpectedly stopped after {i} valid iterations.")
                    results["warning"] = f"DataLoader stopped unexpectedly after {i} iterations."
                    break # Exit the loop
                except Exception as step_err:
                    # Catch OOM or other errors during the step
                    err_msg = f"Error during training step {i} (Fwd/Loss/Bwd/Optim): {step_err}"
                    logger.error(err_msg, exc_info=True)
                    results["error"] = f"Training step {i} failed: {step_err}"
                    # Handle OOM specifically if possible
                    if isinstance(step_err, torch.cuda.OutOfMemoryError):
                         results["error"] = f"CUDA OutOfMemoryError at step {i}"
                         results["error_details"] = str(step_err)
                         # Try to capture memory state at OOM (might be inaccurate)
                         if self.device.type == 'cuda':
                             try:
                                 mem_allocated_oom = torch.cuda.max_memory_allocated(self.device)
                                 results["max_memory_allocated_b"] = mem_allocated_oom
                                 results["max_memory_allocated_mb"] = mem_allocated_oom / (1024**2)
                                 results["max_memory_allocated_formatted"] = format_bytes(mem_allocated_oom)
                                 results["memory_at_oom_approx_mb"] = results["max_memory_allocated_mb"]
                                 logger.error(f"OOM at step {i}. Peak memory allocated before failure approx: {results['max_memory_allocated_formatted']}")
                             except Exception as mem_err: logger.error(f"Could not get memory stats after OOM: {mem_err}")
                             finally: torch.cuda.empty_cache()
                    break # Exit the loop on any error

                # --- Post-Step ---
                # Synchronize for accurate timing if using CUDA
                if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
                iter_end_time = time.perf_counter()
                step_times_sec.append(iter_end_time - iter_start_time)
                actual_iterations_run += 1

                # Step the profiler if active
                if prof and i < num_profiler_steps:
                    prof.step()
            # --- End of Profiling Loop ---

            total_end_time = time.perf_counter()
            results["iterations_completed"] = actual_iterations_run # Store actual completed count

            # --- Exit Profiler Context ---
            if profiler_instance:
                try:
                    profiler_instance.__exit__(None, None, None)
                except Exception as prof_exit_err:
                     logger.error(f"Error exiting profiler context: {prof_exit_err}", exc_info=True)
                     if results["profiler_error"] is None: results["profiler_error"] = f"Profiler exit error: {prof_exit_err}"

            # --- Analyze Profiler Results (if used and successful) ---
            if use_torch_profiler and profiler_instance and results["error"] is None and actual_iterations_run > 0:
                logger.info("[Training Profiling] Analyzing detailed profiler results...")
                try:
                    key_averages = profiler_instance.key_averages()
                    results["profiler_data"] = {}
                    if not key_averages:
                        logger.warning("[Training Profiling] Torch profiler did not record any events.")
                    else:
                        # --- Copy profiler analysis logic (similar to inference) ---
                        total_avg = key_averages.total_average()
                        results["profiler_data"]["total_events_averaged"] = len(key_averages)
                        cpu_time_total_us = total_avg.cpu_time_total; self_cpu_time_total_us = total_avg.self_cpu_time_total
                        cuda_time_total_us = getattr(total_avg, 'cuda_time_total', 0); self_cuda_time_total_us = getattr(total_avg, 'self_cuda_time_total', 0)
                        combined_time_us = cpu_time_total_us + cuda_time_total_us
                        results["profiler_data"]["avg_cpu_time_total_ms"] = cpu_time_total_us / 1000
                        results["profiler_data"]["avg_self_cpu_time_total_ms"] = self_cpu_time_total_us / 1000
                        results["profiler_data"]["avg_cuda_time_total_ms"] = cuda_time_total_us / 1000
                        results["profiler_data"]["avg_self_cuda_time_total_ms"] = self_cuda_time_total_us / 1000
                        results["profiler_data"]["avg_cpu_time_percent"] = (cpu_time_total_us / combined_time_us * 100) if combined_time_us > 0 else 0
                        results["profiler_data"]["avg_gpu_time_percent"] = (cuda_time_total_us / combined_time_us * 100) if combined_time_us > 0 else 0
                        cpu_mem_usage_b = getattr(total_avg, 'cpu_memory_usage', 0)
                        cuda_mem_usage_b = getattr(total_avg, 'cuda_memory_usage', 0) if self.device.type == 'cuda' else 0
                        results["profiler_data"]["profiler_avg_cpu_memory_usage_b"] = cpu_mem_usage_b
                        results["profiler_data"]["profiler_avg_cpu_memory_usage_formatted"] = format_bytes(cpu_mem_usage_b)
                        results["profiler_data"]["profiler_avg_gpu_memory_usage_b"] = cuda_mem_usage_b
                        results["profiler_data"]["profiler_avg_gpu_memory_usage_formatted"] = format_bytes(cuda_mem_usage_b) if cuda_mem_usage_b > 0 else "N/A"
                        sort_key = profiler_sort_by if profiler_sort_by else "self_cpu_time_total"
                        try:
                            table_str = key_averages.table(sort_by=sort_key, row_limit=profiler_row_limit)
                            results["profiler_data"]["profiler_top_ops_summary"] = table_str
                            # logger.debug(f"[Training Profiling] Top {profiler_row_limit} operators by {sort_key}:\n{table_str}")
                        except KeyError as ke:
                            logger.warning(f"Profiler sort key '{sort_key}' not found (KeyError: {ke}). Defaulting.")
                            try: results["profiler_data"]["profiler_top_ops_summary"] = key_averages.table(sort_by="self_cpu_time_total", row_limit=profiler_row_limit)
                            except Exception as table_err_fb: results["profiler_data"]["profiler_top_ops_summary"] = f"Error: {table_err_fb}"
                        except Exception as table_err:
                            err_msg = f"Error generating profiler table: {table_err}"; logger.error(err_msg, exc_info=True)
                            results["profiler_data"]["profiler_top_ops_summary"] = err_msg
                        # --- End profiler analysis ---
                except Exception as prof_err:
                     logger.error(f"[Training Profiling] Failed during torch.profiler analysis: {prof_err}", exc_info=True)
                     results["profiler_error"] = f"Profiler analysis failed: {prof_err}"


            # --- Basic Timing Analysis (if iterations completed) ---
            if actual_iterations_run > 0:
                results["total_profiled_duration_sec"] = total_end_time - total_start_time
                avg_step_s = sum(step_times_sec) / actual_iterations_run
                results["avg_step_time_ms"] = avg_step_s * 1000
                results["avg_data_fetch_time_ms"] = (sum(data_fetch_times_sec) / actual_iterations_run) * 1000
                results["avg_data_prep_time_ms"] = (sum(data_prep_times_sec) / actual_iterations_run) * 1000
                results["avg_data_total_load_time_ms"] = results["avg_data_fetch_time_ms"] + results["avg_data_prep_time_ms"]
                results["avg_forward_time_ms"] = (sum(forward_times_sec) / actual_iterations_run) * 1000
                results["avg_loss_time_ms"] = (sum(loss_times_sec) / actual_iterations_run) * 1000
                results["avg_backward_time_ms"] = (sum(backward_times_sec) / actual_iterations_run) * 1000
                results["avg_optimizer_time_ms"] = (sum(optimizer_times_sec) / actual_iterations_run) * 1000

                # Calculate percentages
                step_time_ms = results["avg_step_time_ms"]
                if step_time_ms > 1e-9: # Avoid division by zero for very fast steps
                    results["percent_time_data_fetch"] = max(0.0, min(100.0, (results["avg_data_fetch_time_ms"] / step_time_ms) * 100))
                    results["percent_time_data_prep"] = max(0.0, min(100.0, (results["avg_data_prep_time_ms"] / step_time_ms) * 100))
                    results["percent_time_data_total_load"] = max(0.0, min(100.0, (results["avg_data_total_load_time_ms"] / step_time_ms) * 100))
                    results["percent_time_forward"] = max(0.0, min(100.0, (results["avg_forward_time_ms"] / step_time_ms) * 100))
                    results["percent_time_loss"] = max(0.0, min(100.0, (results["avg_loss_time_ms"] / step_time_ms) * 100))
                    results["percent_time_backward"] = max(0.0, min(100.0, (results["avg_backward_time_ms"] / step_time_ms) * 100))
                    results["percent_time_optimizer"] = max(0.0, min(100.0, (results["avg_optimizer_time_ms"] / step_time_ms) * 100))
                    # Calculate 'Other' time if percentages don't add up (rounding errors, etc.)
                    total_percent = sum(results[f"percent_time_{phase}"] for phase in ["data_total_load", "forward", "loss", "backward", "optimizer"])
                    results["percent_time_other"] = max(0.0, 100.0 - total_percent)

                # Log basic timing summary
                log_breakdown = (f"Step={step_time_ms:.2f} | "
                                 f"DataLoad={results['avg_data_total_load_time_ms']:.2f} ({results['percent_time_data_total_load']:.1f}%) | "
                                 f"Fwd={results['avg_forward_time_ms']:.2f} ({results['percent_time_forward']:.1f}%) | "
                                 f"Loss={results['avg_loss_time_ms']:.2f} ({results['percent_time_loss']:.1f}%) | "
                                 f"Bwd={results['avg_backward_time_ms']:.2f} ({results['percent_time_backward']:.1f}%) | "
                                 f"Optim={results['avg_optimizer_time_ms']:.2f} ({results['percent_time_optimizer']:.1f}%)")
                logger.info(f"[Training Profiling] Basic timing breakdown (avg ms over {actual_iterations_run} steps): {log_breakdown}")

            elif results["error"] is None: # No iterations ran, but no explicit error reported yet
                logger.warning("[Training Profiling] No iterations completed successfully. Timing results unavailable.")
                # Set error only if not already set by a previous exception
                results["error"] = "No iterations completed for timing."

            # --- Capture Overall Peak Memory (after profiling) ---
            # This reflects the peak across the entire profiled run (warmup + timed steps)
            if self.device.type == 'cuda' and results["error"] != "DataLoader exhausted during warmup (potentially empty).": # Only report if we ran something
                 # Don't overwrite memory captured during OOM handling
                 if results.get("memory_at_oom_approx_mb") is None:
                     results["max_memory_allocated_b"] = torch.cuda.max_memory_allocated(self.device)
                     results["max_memory_reserved_b"] = torch.cuda.max_memory_reserved(self.device)
                     results["max_memory_allocated_mb"] = results["max_memory_allocated_b"] / (1024**2)
                     results["max_memory_reserved_mb"] = results["max_memory_reserved_b"] / (1024**2)
                     results["max_memory_allocated_formatted"] = format_bytes(results["max_memory_allocated_b"])
                     results["max_memory_reserved_formatted"] = format_bytes(results["max_memory_reserved_b"])
                     logger.info(f"[Training Profiling] Peak CUDA Memory: Allocated={results['max_memory_allocated_formatted']}, Reserved={results['max_memory_reserved_formatted']}")
            # CPU memory peak is harder to track accurately without external tools or heavy profiling

        # --- Global Error Handling (Catch errors before warmup or during setup) ---
        except Exception as e:
            err_msg = f"General error during training profiling setup or execution: {e}"
            logger.error(err_msg, exc_info=True)
            if results["error"] is None: # Avoid overwriting more specific errors
                 results["error"] = "General training profiling error"
                 results["error_details"] = str(e)
            # Handle OOM specifically during setup/warmup if not caught later
            if isinstance(e, torch.cuda.OutOfMemoryError) and results["error"] != f"CUDA OutOfMemoryError at step {i}":
                 results["error"] = "CUDA OutOfMemoryError (likely during warmup/setup)"
                 results["error_details"] = str(e)
                 if self.device.type == 'cuda': torch.cuda.empty_cache()

        finally:
            # --- Cleanup ---
            # Ensure profiler context is exited if manually managed and instance exists
            # (Defensive check, should be handled by __exit__ call after loop)
            # if use_torch_profiler and profiler_instance and profiler_instance._profiler_kind != torch.autograd.profiler.ProfilerState.Disabled:
            #      try:
            #           profiler_instance.__exit__(None, None, None)
            #      except Exception as final_prof_exit_err:
            #           logger.error(f"Error during final profiler exit in finally block: {final_prof_exit_err}")

            # Restore original model training mode
            self.model.train(mode=original_mode_is_train)
            logger.debug(f"Restored model training mode to: {original_mode_is_train}")
            # Optional: Clear cache after profiling run?
            if self.device.type == 'cuda': torch.cuda.empty_cache()


        # Final check on completed iterations vs requested
        if results["iterations_requested"] != results["iterations_completed"] and results["error"] is None and results["warning"] is None:
            results["warning"] = f"Completed {results['iterations_completed']} iterations, less than requested {results['iterations_requested']}."

        return results