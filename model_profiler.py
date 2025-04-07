# TrainSense/model_profiler.py
import torch
import torch.nn as nn
import time
import logging
from typing import Tuple, Dict, Any, Optional, Union, Callable, Iterable
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from torch.optim import Optimizer
from torch.utils.data import DataLoader # Bien qu'on utilise Iterable, c'est utile pour le type hinting et compréhension

# Assurez-vous que les utils sont correctement importés depuis le même package
from .utils import format_bytes, format_time, validate_positive_integer

logger = logging.getLogger(__name__)

class ModelProfiler:
    """
    Profiles PyTorch models for inference speed, training step duration,
    and resource utilization using basic timing and torch.profiler.
    """
    def __init__(self, model: nn.Module, device: Optional[Union[str, torch.device]] = None):
        """
        Initializes the ModelProfiler.

        Args:
            model (nn.Module): The PyTorch model to profile.
            device (Optional[Union[str, torch.device]]): The device ('cpu', 'cuda', etc.)
                to run profiling on. Autodetects if None.
        """
        if not isinstance(model, nn.Module):
            raise TypeError("Input 'model' must be an instance of torch.nn.Module.")

        self.model = model
        self.device = self._resolve_device(device)
        self.model.to(self.device)
        logger.info(f"ModelProfiler initialized for model {type(model).__name__} on device '{self.device}'")

    def _resolve_device(self, device: Optional[Union[str, torch.device]]) -> torch.device:
        """Determines the torch device to use."""
        if device:
            return torch.device(device)
        elif torch.cuda.is_available():
            logger.info("CUDA available, selecting CUDA device.")
            return torch.device("cuda")
        else:
             logger.info("CUDA not available, selecting CPU device.")
             return torch.device("cpu")

    def _generate_dummy_input(self, input_shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Generates a dummy input tensor."""
        logger.debug(f"Generating dummy input tensor with shape: {input_shape}, dtype: {dtype}")
        try:
            return torch.randn(*input_shape, device=self.device, dtype=dtype)
        except Exception as e:
            logger.error(f"Failed to create dummy tensor with shape {input_shape} on device {self.device}: {e}", exc_info=True)
            raise ValueError(f"Failed to create dummy input with shape {input_shape}: {e}") from e

    def _generate_dummy_target(self, output: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        """
        Generates a dummy target compatible with the model output and criterion.
        Attempts common cases but might require a custom target_generator.
        """
        target_shape = output.shape
        logger.debug(f"Generating dummy target for output shape {target_shape} and criterion {type(criterion).__name__}")
        try:
            # Handle common loss types
            if isinstance(criterion, (nn.CrossEntropyLoss, nn.NLLLoss)):
                num_classes = output.shape[-1]
                if len(output.shape) == 2: # (batch, classes)
                     return torch.randint(0, num_classes, (target_shape[0],), device=self.device, dtype=torch.long)
                else: # Assume segmentation or similar (batch, classes, H, W) -> target (batch, H, W)
                     logger.debug("Assuming segmentation-like target shape based on CrossEntropyLoss and output shape.")
                     return torch.randint(0, num_classes, (target_shape[0], *target_shape[2:]), device=self.device, dtype=torch.long)
            elif isinstance(criterion, (nn.MSELoss, nn.L1Loss, nn.SmoothL1Loss)):
                # Requires target with same shape and type as output (usually float)
                return torch.randn_like(output)
            elif isinstance(criterion, (nn.BCELoss, nn.BCEWithLogitsLoss)):
                 # Target should be same shape, typically floats between 0 and 1
                 # For BCEWithLogitsLoss, target can be float; for BCELoss, target usually is float {0, 1} but randn is ok for profiling shape
                 return torch.rand_like(output) # Use rand instead of randn for [0, 1) range
            else:
                 logger.warning(f"Unsupported criterion type {type(criterion).__name__} for automatic target generation. Using zeros_like. Provide target_generator if needed.")
                 return torch.zeros_like(output)
        except Exception as e:
            logger.error(f"Failed to generate dummy target for output shape {output.shape} and criterion {type(criterion).__name__}: {e}", exc_info=True)
            raise ValueError(f"Failed to generate dummy target for criterion {type(criterion).__name__}.") from e

    def profile_model(self,
                      input_shape: Tuple[int, ...],
                      iterations: int = 50,
                      warmup: int = 10,
                      use_torch_profiler: bool = True,
                      profiler_activities: Optional[list] = None,
                      profiler_sort_by: str = "self_cpu_time_total",
                      profiler_row_limit: int = 10
                     ) -> Dict[str, Any]:
        """
        Profiles the model's inference performance.

        Args:
            input_shape (Tuple[int, ...]): The shape of a single input sample or batch
                                           (e.g., (batch_size, channels, height, width)).
            iterations (int): Number of inference iterations to time.
            warmup (int): Number of warmup iterations to run before timing.
            use_torch_profiler (bool): Whether to use the detailed torch.profiler.
            profiler_activities (Optional[list]): Activities for torch.profiler (e.g., [ProfilerActivity.CPU, ProfilerActivity.CUDA]). Autodetected if None.
            profiler_sort_by (str): Key to sort the torch.profiler table by.
            profiler_row_limit (int): Number of rows to show in the profiler table.

        Returns:
            Dict[str, Any]: A dictionary containing profiling results (times, throughput, memory, profiler data if used).
        """
        validate_positive_integer(iterations, "Profiling iterations")
        validate_positive_integer(warmup, "Profiling warmup iterations", allow_zero=True)

        if not isinstance(input_shape, tuple) or not all(isinstance(d, int) and d > 0 for d in input_shape):
            raise ValueError(f"input_shape must be a tuple of positive integers, got {input_shape}.")

        logger.info(f"[Inference Profiling] Starting: input_shape={input_shape}, iterations={iterations}, warmup={warmup}, device={self.device}, use_torch_profiler={use_torch_profiler}")

        original_mode_is_train = self.model.training
        self.model.eval() # <<< Set to evaluation mode for inference profiling
        dummy_input = self._generate_dummy_input(input_shape)
        results: Dict[str, Any] = {
            "profiling_type": "inference", # Identify the type of profiling
            "input_shape": input_shape,
            "device": str(self.device),
            "iterations": iterations,
            "warmup": warmup,
            "use_torch_profiler": use_torch_profiler,
        }

        # Reset CUDA memory stats before the run
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()

        try:
             with torch.no_grad(): # <<< No gradients needed for inference
                 # --- Warmup phase ---
                 logger.debug("[Inference Profiling] Running warmup...")
                 for _ in range(warmup):
                     _ = self.model(dummy_input)
                 if self.device.type == 'cuda':
                     torch.cuda.synchronize(self.device)
                 logger.debug("[Inference Profiling] Warmup complete.")

                 # --- Simple timing ---
                 logger.debug("[Inference Profiling] Running timed iterations...")
                 start_time = time.perf_counter()
                 for _ in range(iterations):
                     _ = self.model(dummy_input)
                 # Ensure all GPU operations are finished before stopping the timer
                 if self.device.type == 'cuda':
                     torch.cuda.synchronize(self.device)
                 end_time = time.perf_counter()

                 total_time_sec = end_time - start_time
                 # Handle potential division by zero if iterations is 0 or time is negligible
                 avg_total_time_sec = total_time_sec / iterations if iterations > 0 else 0
                 throughput = iterations / total_time_sec if total_time_sec > 0 else float('inf')

                 results["avg_total_time_ms"] = avg_total_time_sec * 1000
                 results["throughput_samples_per_sec"] = throughput
                 results["total_timed_duration_sec"] = total_time_sec
                 logger.info(f"[Inference Profiling] Basic timing complete: Avg time={results['avg_total_time_ms']:.3f} ms, Throughput={throughput:.1f} samples/sec")

                 # --- Detailed profiling using torch.profiler ---
                 if use_torch_profiler:
                     logger.info("[Inference Profiling] Running detailed profiling with torch.profiler...")
                     if profiler_activities is None:
                          profiler_activities = [ProfilerActivity.CPU]
                          if self.device.type == 'cuda':
                               profiler_activities.append(ProfilerActivity.CUDA)
                     # Adjust profiler iterations for manageability
                     profile_iterations = min(iterations, 10)
                     profile_warmup = min(warmup, 5)
                     wait = 1
                     prof_schedule = schedule(wait=wait, warmup=profile_warmup, active=profile_iterations, repeat=1)
                     num_profiler_steps = wait + profile_warmup + profile_iterations

                     try:
                         # Use context manager for the profiler
                         with profile(activities=profiler_activities,
                                      record_shapes=True,
                                      profile_memory=True,
                                      with_stack=False, # Keep stack traces off unless needed for deep debugging
                                      schedule=prof_schedule
                                      ) as prof:
                             # Run the loop for the scheduled number of steps
                             for i in range(num_profiler_steps):
                                 with record_function(f"inference_iteration_{i}"): # Mark iterations
                                     _ = self.model(dummy_input)
                                 prof.step() # Signal scheduler

                         logger.info("[Inference Profiling] Torch profiler run complete. Analyzing results...")

                         # --- Analyze profiler results ---
                         key_averages = prof.key_averages()
                         results["profiler_data"] = {} # Store detailed profiler results here
                         if not key_averages:
                              logger.warning("[Inference Profiling] Torch profiler did not record any events.")
                         else:
                             total_avg = key_averages.total_average()
                             results["profiler_data"]["total_events_averaged"] = len(key_averages)

                             # Calculate combined time
                             cuda_time_total_us = getattr(total_avg, 'cuda_time_total', 0)
                             combined_time_us = total_avg.cpu_time_total + cuda_time_total_us

                             # CPU Time Stats
                             results["profiler_data"]["avg_cpu_time_total_ms"] = total_avg.cpu_time_total / 1000
                             results["profiler_data"]["avg_self_cpu_time_total_ms"] = total_avg.self_cpu_time_total / 1000
                             results["profiler_data"]["avg_cpu_time_percent"] = (total_avg.cpu_time_total / combined_time_us * 100) if combined_time_us > 0 else 0

                             # CUDA Time Stats
                             results["profiler_data"]["avg_cuda_time_total_ms"] = 0
                             results["profiler_data"]["avg_self_cuda_time_total_ms"] = 0
                             results["profiler_data"]["avg_gpu_time_percent"] = 0
                             if self.device.type == 'cuda' and cuda_time_total_us > 0:
                                 results["profiler_data"]["avg_cuda_time_total_ms"] = cuda_time_total_us / 1000
                                 results["profiler_data"]["avg_self_cuda_time_total_ms"] = getattr(total_avg, 'self_cuda_time_total', 0) / 1000
                                 results["profiler_data"]["avg_gpu_time_percent"] = (cuda_time_total_us / combined_time_us * 100) if combined_time_us > 0 else 0

                             # Memory Stats from Profiler (usage *during* profiled ops)
                             results["profiler_data"]["profiler_avg_cpu_memory_usage_b"] = getattr(total_avg, 'cpu_memory_usage', 0)
                             results["profiler_data"]["profiler_avg_cpu_memory_usage_formatted"] = format_bytes(results["profiler_data"]["profiler_avg_cpu_memory_usage_b"])
                             results["profiler_data"]["profiler_avg_gpu_memory_usage_b"] = 0
                             results["profiler_data"]["profiler_avg_gpu_memory_usage_formatted"] = "N/A"
                             if self.device.type == 'cuda':
                                results["profiler_data"]["profiler_avg_gpu_memory_usage_b"] = getattr(total_avg, 'cuda_memory_usage', 0)
                                results["profiler_data"]["profiler_avg_gpu_memory_usage_formatted"] = format_bytes(results["profiler_data"]["profiler_avg_gpu_memory_usage_b"])

                             # Top N operators table
                             sort_by_key = profiler_sort_by if profiler_sort_by else "self_cpu_time_total"
                             try:
                                results["profiler_data"]["profiler_top_ops_summary"] = key_averages.table(sort_by=sort_by_key, row_limit=profiler_row_limit)
                                logger.info(f"[Inference Profiling] Top {profiler_row_limit} operators by {sort_by_key}:\n{results['profiler_data']['profiler_top_ops_summary']}")
                             except KeyError as ke: # Handle case where sort key might not exist in rare cases
                                 logger.warning(f"Could not sort profiler table by '{sort_by_key}' (KeyError: {ke}). Defaulting to 'self_cpu_time_total'.")
                                 try:
                                     results["profiler_data"]["profiler_top_ops_summary"] = key_averages.table(sort_by="self_cpu_time_total", row_limit=profiler_row_limit)
                                     logger.info(f"[Inference Profiling] Top {profiler_row_limit} operators by self_cpu_time_total:\n{results['profiler_data']['profiler_top_ops_summary']}")
                                 except Exception as table_err_fallback:
                                     logger.error(f"Could not generate profiler table with default sort key: {table_err_fallback}", exc_info=True)
                                     results["profiler_data"]["profiler_top_ops_summary"] = f"Error generating table: {table_err_fallback}"
                             except Exception as table_err:
                                 logger.error(f"Could not generate profiler table sorted by '{sort_by_key}': {table_err}", exc_info=True)
                                 results["profiler_data"]["profiler_top_ops_summary"] = f"Error generating table: {table_err}"

                     except Exception as prof_err:
                          logger.error(f"[Inference Profiling] Failed during torch.profiler execution or analysis: {prof_err}", exc_info=True)
                          results["profiler_error"] = f"Profiler run/analysis failed: {str(prof_err)}"

                 # --- Capture peak memory overall after the run ---
                 if self.device.type == 'cuda':
                     results["max_memory_allocated_b"] = torch.cuda.max_memory_allocated(self.device)
                     results["max_memory_allocated_mb"] = results["max_memory_allocated_b"] / (1024**2)
                     results["max_memory_allocated_formatted"] = format_bytes(results["max_memory_allocated_b"])
                     results["max_memory_reserved_b"] = torch.cuda.max_memory_reserved(self.device)
                     results["max_memory_reserved_formatted"] = format_bytes(results["max_memory_reserved_b"])
                 else:
                     # Estimate CPU peak from profiler if available and profiling was run
                     if use_torch_profiler and "profiler_data" in results:
                        results["max_memory_allocated_b"] = results["profiler_data"].get("profiler_avg_cpu_memory_usage_b")
                     else:
                        results["max_memory_allocated_b"] = None # Cannot determine without profiler

                     if results["max_memory_allocated_b"] is not None:
                          results["max_memory_allocated_mb"] = results["max_memory_allocated_b"] / (1024**2)
                          results["max_memory_allocated_formatted"] = format_bytes(results["max_memory_allocated_b"]) + " (estimated peak based on profiler avg)"
                     else:
                          results["max_memory_allocated_mb"] = None
                          results["max_memory_allocated_formatted"] = "N/A (CPU Peak requires profiler or external tool)"

        # --- Error Handling ---
        except torch.cuda.OutOfMemoryError as oom_err:
            logger.error(f"[Inference Profiling] CUDA OOM: {oom_err}", exc_info=False)
            results["error"] = "CUDA OutOfMemoryError"
            results["error_details"] = str(oom_err)
            if self.device.type == 'cuda':
                try:
                     results["max_memory_allocated_b"] = torch.cuda.max_memory_allocated(self.device)
                     results["max_memory_allocated_mb"] = results["max_memory_allocated_b"] / (1024**2)
                     results["max_memory_allocated_formatted"] = format_bytes(results["max_memory_allocated_b"])
                     results["memory_required_at_oom_approx_mb"] = results["max_memory_allocated_mb"]
                except Exception: pass
                finally: torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"[Inference Profiling] Error: {e}", exc_info=True)
            results["error"] = f"General profiling error: {str(e)}"
        finally:
             # Restore original model training mode
             self.model.train(mode=original_mode_is_train)
             logger.debug(f"Restored model training mode to: {original_mode_is_train}")


        return results


    def profile_training_step(self,
                              data_loader: Iterable,
                              criterion: nn.Module,
                              optimizer: Optimizer,
                              target_generator: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                              iterations: int = 10,
                              warmup: int = 3,
                              use_torch_profiler: bool = True,
                              profiler_activities: Optional[list] = None,
                              profiler_sort_by: str = "self_cpu_time_total", # Can also use "self_cuda_time_total"
                              profiler_row_limit: int = 15
                             ) -> Dict[str, Any]:
        """
        Profiles a full training step (data loading, forward, loss, backward, optimizer step),
        separating data fetch and preparation time.

        Args:
            data_loader (Iterable): An iterable (like `torch.utils.data.DataLoader`) that yields batches.
                                    Assumes batches are tuples/lists where batch[0] is input, batch[1] is target,
                                    or single tensors if target_generator is provided.
            criterion (nn.Module): The loss function instance.
            optimizer (Optimizer): The optimizer instance.
            target_generator (Optional[Callable]): A function that takes model output tensor
                                                   and returns a compatible target tensor, if targets
                                                   cannot be inferred from the dataloader or criterion.
            iterations (int): Number of training steps to time.
            warmup (int): Number of warmup training steps to run before timing.
            use_torch_profiler (bool): Whether to use the detailed torch.profiler.
            profiler_activities (Optional[list]): Activities for torch.profiler. Autodetected if None.
            profiler_sort_by (str): Key to sort the torch.profiler table by.
            profiler_row_limit (int): Number of rows to show in the profiler table.

        Returns:
            Dict[str, Any]: A dictionary containing profiling results (step time breakdown, memory, profiler data if used).
        """
        validate_positive_integer(iterations, "Training profiling iterations", allow_zero=False)
        validate_positive_integer(warmup, "Training profiling warmup", allow_zero=True)

        logger.info(f"[Training Profiling] Starting: iterations={iterations}, warmup={warmup}, device={self.device}, use_torch_profiler={use_torch_profiler}")

        original_mode_is_train = self.model.training
        self.model.train() # <<< Set model to training mode
        criterion.to(self.device) # Ensure criterion is on the correct device

        # Ensure data loader provides data
        try:
            data_iter = iter(data_loader)
            # Try getting one batch to catch immediate errors
            _ = next(data_iter)
            # Reset iterator for the actual profiling
            data_iter = iter(data_loader)
        except Exception as loader_err:
             logger.error(f"Failed to get data from data_loader: {loader_err}", exc_info=True)
             return {"error": f"DataLoader error: {loader_err}"}


        results: Dict[str, Any] = {
            "profiling_type": "training_step", # Identify the type
            "device": str(self.device),
            "iterations": iterations,
            "warmup": warmup,
            "use_torch_profiler": use_torch_profiler,
            "optimizer_type": type(optimizer).__name__,
            "criterion_type": type(criterion).__name__,
        }

        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()

        # Store detailed step times
        step_times_sec = []
        data_fetch_times_sec = []
        data_prep_times_sec = []
        forward_times_sec = []
        loss_times_sec = []
        backward_times_sec = []
        optimizer_times_sec = []

        try:
            # --- Warmup Phase (Training Mode) ---
            logger.debug("[Training Profiling] Running warmup...")
            for wu_i in range(warmup):
                 try: batch = next(data_iter)
                 except StopIteration:
                     logger.error(f"[Training Profiling] DataLoader exhausted during warmup after {wu_i} iterations.")
                     results["error"] = "DataLoader exhausted during warmup."
                     return results # Cannot proceed

                 try:
                     # Perform a basic training step without detailed timing
                     if isinstance(batch, (list, tuple)):
                          inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                     else: # Assume batch is input, generate target
                          inputs = batch.to(self.device)
                          outputs_tmp = self.model(inputs) # Run model to get shape
                          targets = (target_generator(outputs_tmp) if target_generator
                                     else self._generate_dummy_target(outputs_tmp, criterion)).to(self.device)

                     optimizer.zero_grad(set_to_none=True) # Use set_to_none=True for potential perf gain
                     outputs = self.model(inputs)
                     loss = criterion(outputs, targets)
                     loss.backward()
                     optimizer.step()
                 except Exception as wu_err:
                     logger.error(f"Error during warmup step {wu_i+1}: {wu_err}", exc_info=True)
                     results["error"] = f"Error during warmup step: {wu_err}"
                     return results # Cannot proceed if warmup fails

            if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
            logger.debug("[Training Profiling] Warmup complete.")

            # --- Timed/Profiled Iterations ---
            logger.debug(f"[Training Profiling] Running {iterations} profiled iterations...")

            prof = None
            profiler_instance = None # Define outside the loop
            if use_torch_profiler:
                if profiler_activities is None:
                    profiler_activities = [ProfilerActivity.CPU]
                    if self.device.type == 'cuda': profiler_activities.append(ProfilerActivity.CUDA)
                # Profile fewer active steps for training to reduce overhead
                profile_active = min(iterations, 5)
                profile_warmup = 1
                wait = 0
                prof_schedule = schedule(wait=wait, warmup=profile_warmup, active=profile_active, repeat=1)
                num_profiler_steps = wait + profile_warmup + profile_active
                logger.info(f"[Training Profiling] Profiler schedule: wait={wait}, warmup={profile_warmup}, active={profile_active}")
                profiler_instance = profile(activities=profiler_activities, record_shapes=True, profile_memory=True, with_stack=False, schedule=prof_schedule)
                prof = profiler_instance.__enter__() # Manually manage context

            total_start_time = time.perf_counter()
            actual_iterations_run = 0

            for i in range(iterations):
                iter_start_time = time.perf_counter()

                # --- Data Fetching ---
                t0 = time.perf_counter()
                try:
                    # Use record_function to mark this phase for torch.profiler
                    with record_function("data_fetch"):
                         batch = next(data_iter)
                except StopIteration:
                    logger.warning(f"[Training Profiling] DataLoader exhausted after {i} valid iterations.")
                    results["warning"] = f"DataLoader exhausted after {i} valid iterations."
                    iterations = i # Update iteration count to actual number run
                    break # Stop profiling loop
                except Exception as fetch_err:
                     logger.error(f"Error fetching data at iteration {i}: {fetch_err}", exc_info=True)
                     results["error"] = f"Error fetching data at iteration {i}: {fetch_err}"
                     iterations = i; break
                t1 = time.perf_counter()
                data_fetch_times_sec.append(t1 - t0)

                # --- Data Preparation & Move ---
                try:
                    with record_function("data_prep_move"):
                        if isinstance(batch, (list, tuple)):
                            # Assuming standard (input, target) format
                            inputs = batch[0].to(self.device, non_blocking=True)
                            if len(batch) > 1:
                                targets_cpu = batch[1]
                            else: # Only inputs provided
                                logger.warning(f"Batch {i} has only one element, generating dummy target.")
                                with torch.no_grad(): outputs_tmp = self.model(inputs) # Get output shape without grads
                                targets_cpu = target_generator(outputs_tmp) if target_generator else self._generate_dummy_target(outputs_tmp, criterion)
                            targets = targets_cpu.to(self.device, non_blocking=True)
                        else: # Assume batch is input tensor
                            inputs = batch.to(self.device, non_blocking=True)
                            with torch.no_grad(): outputs_tmp = self.model(inputs) # Get output shape without grads
                            targets_cpu = target_generator(outputs_tmp) if target_generator else self._generate_dummy_target(outputs_tmp, criterion)
                            targets = targets_cpu.to(self.device, non_blocking=True)
                except Exception as data_err:
                     logger.error(f"Error processing/moving batch {i}: {data_err}", exc_info=True)
                     results["error"] = f"Error processing batch {i}: {data_err}"; iterations = i; break
                t2 = time.perf_counter()
                data_prep_times_sec.append(t2 - t1)

                # --- Forward Pass ---
                try:
                    with record_function("forward_pass"): outputs = self.model(inputs)
                except Exception as fwd_err:
                     logger.error(f"Error during forward pass at iteration {i}: {fwd_err}", exc_info=True)
                     results["error"] = f"Forward pass error at iteration {i}: {fwd_err}"; iterations = i; break
                t3 = time.perf_counter(); forward_times_sec.append(t3 - t2)

                # --- Loss Calculation ---
                try:
                    with record_function("loss_calculation"): loss = criterion(outputs, targets)
                except Exception as loss_err:
                    logger.error(f"Error during loss calculation at iteration {i}: {loss_err}", exc_info=True)
                    results["error"] = f"Loss calculation error at iteration {i}: {loss_err}"; iterations = i; break
                t4 = time.perf_counter(); loss_times_sec.append(t4 - t3)

                # --- Backward Pass ---
                try:
                    optimizer.zero_grad(set_to_none=True)
                    with record_function("backward_pass"): loss.backward()
                except Exception as bwd_err:
                     logger.error(f"Error during backward pass at iteration {i}: {bwd_err}", exc_info=True)
                     results["error"] = f"Backward pass error at iteration {i}: {bwd_err}"; iterations = i; break
                t5 = time.perf_counter(); backward_times_sec.append(t5 - t4)

                # --- Optimizer Step ---
                try:
                    with record_function("optimizer_step"): optimizer.step()
                except Exception as opt_err:
                     logger.error(f"Error during optimizer step at iteration {i}: {opt_err}", exc_info=True)
                     results["error"] = f"Optimizer step error at iteration {i}: {opt_err}"; iterations = i; break
                t6 = time.perf_counter(); optimizer_times_sec.append(t6 - t5)

                # Synchronize GPU if needed before ending iteration timer
                if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
                iter_end_time = time.perf_counter()
                step_times_sec.append(iter_end_time - iter_start_time)
                actual_iterations_run += 1 # Increment count of successful iterations

                # Step the profiler if active
                if prof and i < num_profiler_steps:
                    prof.step()

            total_end_time = time.perf_counter()
            results["iterations_completed"] = actual_iterations_run # Store actual count

            # Ensure profiler context is exited
            if prof:
                try: profiler_instance.__exit__(None, None, None)
                except Exception as prof_exit_err: logger.error(f"Error exiting profiler context: {prof_exit_err}", exc_info=True)

            # --- Profiler Analysis (if used and completed) ---
            if use_torch_profiler and profiler_instance and "error" not in results:
                logger.info("[Training Profiling] Analyzing detailed profiler results...")
                try:
                    key_averages = profiler_instance.key_averages() # Use the instance after exit
                    results["profiler_data"] = {}
                    if not key_averages:
                         logger.warning("[Training Profiling] Torch profiler did not record any events.")
                    else:
                        total_avg = key_averages.total_average()
                        results["profiler_data"]["total_events_averaged"] = len(key_averages)
                        # (Analysis logic: CPU/GPU time, percentages, memory, top ops - same as in inference)
                        cuda_time_total_us = getattr(total_avg, 'cuda_time_total', 0)
                        combined_time_us = total_avg.cpu_time_total + cuda_time_total_us
                        results["profiler_data"]["avg_cpu_time_total_ms"] = total_avg.cpu_time_total / 1000
                        results["profiler_data"]["avg_self_cpu_time_total_ms"] = total_avg.self_cpu_time_total / 1000
                        results["profiler_data"]["avg_cpu_time_percent"] = (total_avg.cpu_time_total / combined_time_us * 100) if combined_time_us > 0 else 0
                        # CUDA Time
                        results["profiler_data"]["avg_cuda_time_total_ms"] = 0
                        results["profiler_data"]["avg_self_cuda_time_total_ms"] = 0
                        results["profiler_data"]["avg_gpu_time_percent"] = 0
                        if self.device.type == 'cuda' and cuda_time_total_us > 0:
                            results["profiler_data"]["avg_cuda_time_total_ms"] = cuda_time_total_us / 1000
                            results["profiler_data"]["avg_self_cuda_time_total_ms"] = getattr(total_avg, 'self_cuda_time_total', 0) / 1000
                            results["profiler_data"]["avg_gpu_time_percent"] = (cuda_time_total_us / combined_time_us * 100) if combined_time_us > 0 else 0
                        # Memory
                        results["profiler_data"]["profiler_avg_cpu_memory_usage_b"] = getattr(total_avg, 'cpu_memory_usage', 0)
                        results["profiler_data"]["profiler_avg_cpu_memory_usage_formatted"] = format_bytes(results["profiler_data"]["profiler_avg_cpu_memory_usage_b"])
                        results["profiler_data"]["profiler_avg_gpu_memory_usage_b"] = 0
                        results["profiler_data"]["profiler_avg_gpu_memory_usage_formatted"] = "N/A"
                        if self.device.type == 'cuda':
                            results["profiler_data"]["profiler_avg_gpu_memory_usage_b"] = getattr(total_avg, 'cuda_memory_usage', 0)
                            results["profiler_data"]["profiler_avg_gpu_memory_usage_formatted"] = format_bytes(results["profiler_data"]["profiler_avg_gpu_memory_usage_b"])
                        # Top Ops Table
                        sort_by_key = profiler_sort_by if profiler_sort_by else "self_cpu_time_total"
                        try:
                            results["profiler_data"]["profiler_top_ops_summary"] = key_averages.table(sort_by=sort_by_key, row_limit=profiler_row_limit)
                            logger.info(f"[Training Profiling] Top {profiler_row_limit} operators by {sort_by_key}:\n{results['profiler_data']['profiler_top_ops_summary']}")
                        except KeyError as ke:
                            logger.warning(f"Could not sort profiler table by '{sort_by_key}' (KeyError: {ke}). Defaulting to 'self_cpu_time_total'.")
                            try:
                                results["profiler_data"]["profiler_top_ops_summary"] = key_averages.table(sort_by="self_cpu_time_total", row_limit=profiler_row_limit)
                                logger.info(f"[Training Profiling] Top {profiler_row_limit} operators by self_cpu_time_total:\n{results['profiler_data']['profiler_top_ops_summary']}")
                            except Exception as table_err_fallback:
                                logger.error(f"Could not generate profiler table with default sort key: {table_err_fallback}", exc_info=True)
                                results["profiler_data"]["profiler_top_ops_summary"] = f"Error generating table: {table_err_fallback}"
                        except Exception as table_err:
                            logger.error(f"Could not generate profiler table sorted by '{sort_by_key}': {table_err}", exc_info=True)
                            results["profiler_data"]["profiler_top_ops_summary"] = f"Error generating table: {table_err}"

                except Exception as prof_err:
                     logger.error(f"[Training Profiling] Failed during torch.profiler analysis: {prof_err}", exc_info=True)
                     results["profiler_error"] = f"Profiler analysis failed: {str(prof_err)}"

            # --- Basic Timing Analysis (using successfully completed iterations) ---
            if actual_iterations_run > 0:
                results["total_profiled_duration_sec"] = total_end_time - total_start_time
                # Calculate averages based on actual_iterations_run
                results["avg_step_time_ms"] = (sum(step_times_sec) / actual_iterations_run) * 1000
                results["avg_data_fetch_time_ms"] = (sum(data_fetch_times_sec) / actual_iterations_run) * 1000
                results["avg_data_prep_time_ms"] = (sum(data_prep_times_sec) / actual_iterations_run) * 1000
                results["avg_data_total_load_time_ms"] = results["avg_data_fetch_time_ms"] + results["avg_data_prep_time_ms"]
                results["avg_forward_time_ms"] = (sum(forward_times_sec) / actual_iterations_run) * 1000
                results["avg_loss_time_ms"] = (sum(loss_times_sec) / actual_iterations_run) * 1000
                results["avg_backward_time_ms"] = (sum(backward_times_sec) / actual_iterations_run) * 1000
                results["avg_optimizer_time_ms"] = (sum(optimizer_times_sec) / actual_iterations_run) * 1000

                # Calculate percentages
                step_time_ms = results["avg_step_time_ms"]
                if step_time_ms > 0:
                    results["percent_time_data_fetch"] = (results["avg_data_fetch_time_ms"] / step_time_ms) * 100
                    results["percent_time_data_prep"] = (results["avg_data_prep_time_ms"] / step_time_ms) * 100
                    results["percent_time_data_total_load"] = (results["avg_data_total_load_time_ms"] / step_time_ms) * 100
                    results["percent_time_forward"] = (results["avg_forward_time_ms"] / step_time_ms) * 100
                    results["percent_time_loss"] = (results["avg_loss_time_ms"] / step_time_ms) * 100
                    results["percent_time_backward"] = (results["avg_backward_time_ms"] / step_time_ms) * 100
                    results["percent_time_optimizer"] = (results["avg_optimizer_time_ms"] / step_time_ms) * 100
                else: # Avoid division by zero
                    for key in ["percent_time_data_fetch", "percent_time_data_prep", "percent_time_data_total_load",
                                "percent_time_forward", "percent_time_loss", "percent_time_backward", "percent_time_optimizer"]:
                        results[key] = 0.0

                logger.info(f"[Training Profiling] Basic timing breakdown (avg ms over {actual_iterations_run} steps): "
                            f"Step={step_time_ms:.2f}, "
                            f"DataFetch={results['avg_data_fetch_time_ms']:.2f} ({results['percent_time_data_fetch']:.1f}%), "
                            f"DataPrep={results['avg_data_prep_time_ms']:.2f} ({results['percent_time_data_prep']:.1f}%), "
                            f"Forward={results['avg_forward_time_ms']:.2f} ({results['percent_time_forward']:.1f}%), "
                            f"Loss={results['avg_loss_time_ms']:.2f} ({results['percent_time_loss']:.1f}%), "
                            f"Backward={results['avg_backward_time_ms']:.2f} ({results['percent_time_backward']:.1f}%), "
                            f"Optimizer={results['avg_optimizer_time_ms']:.2f} ({results['percent_time_optimizer']:.1f}%)")
            else:
                # Handle case where no iterations completed successfully
                logger.warning("[Training Profiling] No iterations completed successfully. Timing results are unavailable.")
                if "error" not in results: # Add error if not already present
                    results["error"] = "No iterations completed for timing."


            # --- Capture overall peak memory after the run ---
            if self.device.type == 'cuda':
                results["max_memory_allocated_b"] = torch.cuda.max_memory_allocated(self.device)
                results["max_memory_allocated_mb"] = results["max_memory_allocated_b"] / (1024**2)
                results["max_memory_allocated_formatted"] = format_bytes(results["max_memory_allocated_b"])
                results["max_memory_reserved_b"] = torch.cuda.max_memory_reserved(self.device)
                results["max_memory_reserved_formatted"] = format_bytes(results["max_memory_reserved_b"])
            else: # CPU estimate
                if use_torch_profiler and "profiler_data" in results:
                     results["max_memory_allocated_b"] = results["profiler_data"].get("profiler_avg_cpu_memory_usage_b")
                else:
                     results["max_memory_allocated_b"] = None
                if results["max_memory_allocated_b"] is not None:
                    results["max_memory_allocated_mb"] = results["max_memory_allocated_b"] / (1024**2)
                    results["max_memory_allocated_formatted"] = format_bytes(results["max_memory_allocated_b"]) + " (estimated peak based on profiler avg)"
                else:
                    results["max_memory_allocated_mb"] = None
                    results["max_memory_allocated_formatted"] = "N/A (CPU Peak requires profiler or external tool)"


        # --- Error Handling for the entire process ---
        except torch.cuda.OutOfMemoryError as oom_err:
            logger.error(f"[Training Profiling] CUDA OOM: {oom_err}", exc_info=False)
            results["error"] = "CUDA OutOfMemoryError"
            results["error_details"] = str(oom_err)
            if self.device.type == 'cuda':
                try:
                     results["max_memory_allocated_b"] = torch.cuda.max_memory_allocated(self.device)
                     results["max_memory_allocated_mb"] = results["max_memory_allocated_b"] / (1024**2)
                     results["max_memory_allocated_formatted"] = format_bytes(results["max_memory_allocated_b"])
                     results["memory_required_at_oom_approx_mb"] = results["max_memory_allocated_mb"]
                except Exception: pass
                finally: torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"[Training Profiling] Error: {e}", exc_info=True)
            if "error" not in results: # Avoid overwriting more specific errors
                 results["error"] = f"General training profiling error: {str(e)}"
        finally:
            # Restore original model training mode
            self.model.train(mode=original_mode_is_train)
            logger.debug(f"Restored model training mode to: {original_mode_is_train}")


        # Record actual iterations run if different from requested
        if "iterations_completed" in results and results["iterations_completed"] != results["iterations"]:
             results["iterations_requested"] = results.pop("iterations") # Rename original key
             results["iterations"] = results.pop("iterations_completed") # Use actual count as primary iterations key

        return results