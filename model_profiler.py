# TrainSense/model_profiler.py
import torch
import torch.nn as nn
import time
import logging
from typing import Tuple, Dict, Any, Optional, Union
# Corrected import: Removed KinetoEvent
from torch.profiler import profile, record_function, ProfilerActivity, schedule

from .utils import format_bytes, format_time, validate_positive_integer

logger = logging.getLogger(__name__)

class ModelProfiler:
    def __init__(self, model: nn.Module, device: Optional[Union[str, torch.device]] = None):
        if not isinstance(model, nn.Module):
            raise TypeError("Input 'model' must be an instance of torch.nn.Module.")

        self.model = model
        self.device = self._resolve_device(device)
        self.model.to(self.device)
        logger.info(f"ModelProfiler initialized for model {type(model).__name__} on device '{self.device}'")

    def _resolve_device(self, device: Optional[Union[str, torch.device]]) -> torch.device:
        if device:
            return torch.device(device)
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
             return torch.device("cpu")

    def _generate_dummy_input(self, input_shape: Tuple[int, ...]) -> torch.Tensor:
        logger.debug(f"Generating dummy input tensor with shape: {input_shape}")
        # Add flexibility for dtype later if needed
        try:
            return torch.randn(*input_shape, device=self.device)
        except Exception as e:
            logger.error(f"Failed to create dummy tensor with shape {input_shape} on device {self.device}: {e}", exc_info=True)
            raise ValueError(f"Failed to create dummy input with shape {input_shape}: {e}") from e


    def profile_model(self,
                      input_shape: Tuple[int, ...],
                      iterations: int = 50,
                      warmup: int = 10,
                      use_torch_profiler: bool = True,
                      profiler_activities: Optional[list] = None,
                      profiler_sort_by: str = "self_cpu_time_total",
                      profiler_row_limit: int = 10
                     ) -> Dict[str, Any]:

        validate_positive_integer(iterations, "Profiling iterations")
        validate_positive_integer(warmup, "Profiling warmup iterations", allow_zero=True)

        if not isinstance(input_shape, tuple) or not all(isinstance(d, int) and d > 0 for d in input_shape):
             # Check for positive dimensions too
            raise ValueError(f"input_shape must be a tuple of positive integers, got {input_shape}.")

        logger.info(f"Starting model profiling: input_shape={input_shape}, iterations={iterations}, warmup={warmup}, device={self.device}, use_torch_profiler={use_torch_profiler}")

        self.model.eval()
        dummy_input = self._generate_dummy_input(input_shape)
        results: Dict[str, Any] = { # Initialize results dict
            "input_shape": input_shape,
            "device": str(self.device),
            "iterations": iterations,
            "warmup": warmup,
            "use_torch_profiler": use_torch_profiler,
        }

        # Reset CUDA memory stats before profiling run
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache() # Try to start with a cleaner slate

        try:
             with torch.no_grad():
                 # --- Warmup phase ---
                 logger.debug(f"Running {warmup} warmup iterations...")
                 for i in range(warmup):
                     _ = self.model(dummy_input)
                 if self.device.type == 'cuda':
                     torch.cuda.synchronize(self.device) # Synchronize specific device
                 logger.debug("Warmup complete.")

                 # --- Simple timing ---
                 logger.debug(f"Running {iterations} timed iterations...")
                 start_time = time.perf_counter()
                 for i in range(iterations):
                     _ = self.model(dummy_input)
                 if self.device.type == 'cuda':
                     torch.cuda.synchronize(self.device) # Synchronize specific device
                 end_time = time.perf_counter()

                 total_time_sec = end_time - start_time
                 avg_total_time_sec = total_time_sec / iterations if iterations > 0 else 0
                 # Prevent division by zero for throughput
                 throughput = iterations / total_time_sec if total_time_sec > 0 else float('inf')

                 results["avg_total_time_ms"] = avg_total_time_sec * 1000
                 results["throughput_samples_per_sec"] = throughput
                 results["total_timed_duration_sec"] = total_time_sec


                 logger.info(f"Basic timing complete: Avg time={results['avg_total_time_ms']:.2f} ms, Throughput={throughput:.2f} samples/sec")

                 # --- Detailed profiling using torch.profiler ---
                 if use_torch_profiler:
                     logger.info("Running detailed profiling with torch.profiler...")
                     if profiler_activities is None:
                          profiler_activities = [ProfilerActivity.CPU]
                          if self.device.type == 'cuda':
                               profiler_activities.append(ProfilerActivity.CUDA)

                     # Profile fewer iterations for detailed view to keep overhead manageable
                     profile_iterations = min(iterations, 10)
                     profile_warmup = min(warmup, 5) # Warmup within profiler context too
                     wait = 1 # Number of idle steps before starting profiler recording

                     # Schedule: wait, warmup, active, repeat
                     prof_schedule = schedule(wait=wait, warmup=profile_warmup, active=profile_iterations, repeat=1)
                     num_profiler_steps = wait + profile_warmup + profile_iterations

                     try:
                         # Use context manager for the profiler
                         with profile(activities=profiler_activities,
                                      record_shapes=True,
                                      profile_memory=True, # Enable memory profiling
                                      with_stack=False, # Stack traces can be very verbose and add overhead
                                      schedule=prof_schedule,
                                      # on_trace_ready=torch.profiler.tensorboard_trace_handler("./log_dir") # Optional: Save trace
                                      ) as prof:
                             # The loop runs for the total number of steps defined by the schedule
                             for i in range(num_profiler_steps):
                                 with record_function(f"iteration_{i}"): # Optional: Mark iterations
                                     _ = self.model(dummy_input)
                                 prof.step() # Signal the profiler scheduler

                         logger.info("Torch profiler run complete. Analyzing results...")

                         # --- Analyze profiler results ---
                         # Use key_averages() for aggregated stats over the 'active' steps
                         key_averages = prof.key_averages()
                         total_avg = key_averages.total_average() # This is a FunctionEventAvg object

                         results["profiler_total_events_averaged"] = len(key_averages)

                         # Calculate combined time from CPU and CUDA components (in microseconds)
                         # Check if cuda_time_total attribute exists, default to 0 if not
                         cuda_time_total_us = getattr(total_avg, 'cuda_time_total', 0)
                         combined_time_us = total_avg.cpu_time_total + cuda_time_total_us

                         results["avg_cpu_time_total_ms"] = total_avg.cpu_time_total / 1000 # us to ms
                         results["avg_self_cpu_time_total_ms"] = total_avg.self_cpu_time_total / 1000

                         # *** Corrected Percentage Calculation ***
                         results["avg_cpu_time_percent"] = (total_avg.cpu_time_total / combined_time_us * 100) if combined_time_us > 0 else 0

                         if self.device.type == 'cuda' and cuda_time_total_us > 0:
                             results["avg_cuda_time_total_ms"] = cuda_time_total_us / 1000 # us to ms
                             results["avg_self_cuda_time_total_ms"] = getattr(total_avg, 'self_cuda_time_total', 0) / 1000
                             # *** Corrected Percentage Calculation ***
                             results["avg_gpu_time_percent"] = (cuda_time_total_us / combined_time_us * 100) if combined_time_us > 0 else 0
                         else:
                             results["avg_cuda_time_total_ms"] = 0
                             results["avg_self_cuda_time_total_ms"] = 0
                             results["avg_gpu_time_percent"] = 0

                         # Memory stats from profiler (represent usage *during* profiled steps)
                         results["profiler_avg_cpu_memory_usage_b"] = getattr(total_avg, 'cpu_memory_usage', 0)
                         results["profiler_avg_self_cpu_memory_usage_b"] = getattr(total_avg, 'self_cpu_memory_usage', 0)
                         results["profiler_avg_cpu_memory_usage_formatted"] = format_bytes(results["profiler_avg_cpu_memory_usage_b"])

                         if self.device.type == 'cuda':
                             results["profiler_avg_gpu_memory_usage_b"] = getattr(total_avg, 'cuda_memory_usage', 0)
                             results["profiler_avg_self_gpu_memory_usage_b"] = getattr(total_avg, 'self_cuda_memory_usage', 0)
                             results["profiler_avg_gpu_memory_usage_formatted"] = format_bytes(results["profiler_avg_gpu_memory_usage_b"])

                         # Overall peak memory since last reset (more reliable for total footprint)
                         if self.device.type == 'cuda':
                             results["max_memory_allocated_b"] = torch.cuda.max_memory_allocated(self.device)
                             results["max_memory_allocated_mb"] = results["max_memory_allocated_b"] / (1024**2)
                             results["max_memory_allocated_formatted"] = format_bytes(results["max_memory_allocated_b"])
                             results["max_memory_reserved_b"] = torch.cuda.max_memory_reserved(self.device) # Added reserved memory
                             results["max_memory_reserved_formatted"] = format_bytes(results["max_memory_reserved_b"])

                         else: # CPU Memory Peak (Hard to get accurately without external tools/specific profiler features)
                              # Use profiler CPU usage as best estimate available within torch
                             results["max_memory_allocated_b"] = results["profiler_avg_cpu_memory_usage_b"]
                             results["max_memory_allocated_mb"] = results["max_memory_allocated_b"] / (1024**2)
                             results["max_memory_allocated_formatted"] = format_bytes(results["max_memory_allocated_b"]) + " (estimated peak based on profiler avg)"


                         # Top N operators table
                         sort_by_key = profiler_sort_by if profiler_sort_by else "self_cpu_time_total"
                         try:
                             results["profiler_top_ops_summary"] = key_averages.table(sort_by=sort_by_key, row_limit=profiler_row_limit)
                             logger.info(f"Top {profiler_row_limit} operators by {sort_by_key}:\n{results['profiler_top_ops_summary']}")
                         except Exception as table_err:
                              logger.warning(f"Could not generate profiler table sorted by '{sort_by_key}': {table_err}")
                              results["profiler_top_ops_summary"] = f"Error generating table: {table_err}"


                     except Exception as prof_err:
                          logger.error(f"Failed during torch.profiler execution or analysis: {prof_err}", exc_info=True)
                          results["profiler_error"] = f"Profiler run/analysis failed: {str(prof_err)}"

                 else: # Not using torch profiler, capture basic memory if CUDA
                      if self.device.type == 'cuda':
                          results["max_memory_allocated_b"] = torch.cuda.max_memory_allocated(self.device)
                          results["max_memory_allocated_mb"] = results["max_memory_allocated_b"] / (1024**2)
                          results["max_memory_allocated_formatted"] = format_bytes(results["max_memory_allocated_b"])
                          results["max_memory_reserved_b"] = torch.cuda.max_memory_reserved(self.device)
                          results["max_memory_reserved_formatted"] = format_bytes(results["max_memory_reserved_b"])
                      else:
                          results["max_memory_allocated_mb"] = None
                          results["max_memory_allocated_formatted"] = "N/A (CPU Peak requires profiler or external tool)"


        except torch.cuda.OutOfMemoryError as oom_err:
             logger.error(f"CUDA Out of Memory during profiling with input shape {input_shape}: {oom_err}", exc_info=False) # Avoid huge traceback for OOM
             results["error"] = "CUDA OutOfMemoryError"
             results["error_details"] = str(oom_err)
             if self.device.type == 'cuda':
                 # Capture peak memory before OOM if possible
                 try:
                     results["max_memory_allocated_b"] = torch.cuda.max_memory_allocated(self.device)
                     results["max_memory_allocated_mb"] = results["max_memory_allocated_b"] / (1024**2)
                     results["max_memory_allocated_formatted"] = format_bytes(results["max_memory_allocated_b"])
                     results["max_memory_reserved_b"] = torch.cuda.max_memory_reserved(self.device)
                     results["max_memory_reserved_formatted"] = format_bytes(results["max_memory_reserved_b"])
                     results["memory_required_at_oom_approx_mb"] = results["max_memory_allocated_mb"] # Approximation
                 except Exception as mem_err:
                      logger.warning(f"Could not get memory stats after OOM: {mem_err}")
                 finally:
                      torch.cuda.empty_cache() # Attempt to free memory
        except Exception as e:
             logger.error(f"An error occurred during model profiling: {e}", exc_info=True)
             results["error"] = f"General profiling error: {str(e)}"

        return results