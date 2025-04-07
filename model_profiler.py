# TrainSense/model_profiler.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import time
import logging
from typing import Tuple, Dict, Any, Optional, Union, Callable, Iterable
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from torch.optim import Optimizer
from torch.utils.data import DataLoader # Pour type hinting
from itertools import cycle # Pour gérer les DataLoaders courts

# Assurez-vous que les utils sont correctement importés depuis le même package
try:
    from .utils import format_bytes, format_time, validate_positive_integer
except ImportError:
    # Fallback pour les tests si utils ne sont pas trouvés via import relatif
    from utils import format_bytes, format_time, validate_positive_integer


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
        # Essayer de déplacer le modèle, mais ne pas planter si ça échoue (ex: déjà sur le bon device)
        try:
            self.model.to(self.device)
        except Exception as e:
            logger.warning(f"Could not move model to device {self.device}: {e}")

        logger.info(f"ModelProfiler initialized for model {type(model).__name__} on device '{self.device}'")

    def _resolve_device(self, device: Optional[Union[str, torch.device]]) -> torch.device:
        """Determines the torch device to use."""
        if device:
            # Convertir string en objet device
            resolved_device = torch.device(device)
            # Vérifier la validité du device cuda spécifié
            if resolved_device.type == 'cuda' and not torch.cuda.is_available():
                 logger.warning(f"Device specified as '{device}' but CUDA is not available. Falling back to CPU.")
                 return torch.device("cpu")
            if resolved_device.type == 'cuda':
                 try:
                      # Tester si on peut accéder au device
                      _ = torch.tensor([1.0], device=resolved_device)
                 except Exception as e:
                      logger.warning(f"Specified CUDA device '{device}' not accessible ({e}). Falling back to default CUDA or CPU.")
                      if torch.cuda.is_available(): return torch.device("cuda") # Fallback au premier GPU dispo
                      else: return torch.device("cpu")
            return resolved_device # Retourner le device validé ou cpu/autre
        elif torch.cuda.is_available():
            logger.info("CUDA available, selecting default CUDA device.")
            return torch.device("cuda")
        else:
             logger.info("CUDA not available, selecting CPU device.")
             return torch.device("cpu")

    def _generate_dummy_input(self, input_shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Generates a dummy input tensor."""
        logger.debug(f"Generating dummy input tensor with shape: {input_shape}, dtype: {dtype}")
        try:
            # Générer sur CPU puis déplacer peut être plus fiable dans certains cas multi-GPU
            tensor_cpu = torch.randn(*input_shape, dtype=dtype)
            return tensor_cpu.to(self.device)
        except Exception as e:
            logger.error(f"Failed to create dummy tensor with shape {input_shape} on device {self.device}: {e}", exc_info=True)
            raise ValueError(f"Failed to create dummy input with shape {input_shape}: {e}") from e

    def _generate_dummy_target(self, output: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        """
        Generates a dummy target compatible with the model output and criterion.
        Attempts common cases but might require a custom target_generator.
        """
        target_shape = output.shape
        output_device = output.device # Créer la target sur le même device que l'output
        logger.debug(f"Generating dummy target for output shape {target_shape} and criterion {type(criterion).__name__} on device {output_device}")
        try:
            # Handle common loss types
            if isinstance(criterion, (nn.CrossEntropyLoss)):
                num_classes = output.shape[-1]
                if len(output.shape) == 2: # (batch, classes)
                     return torch.randint(0, num_classes, (target_shape[0],), device=output_device, dtype=torch.long)
                elif len(output.shape) == 4 and output.shape[1] == num_classes: # Assume (batch, classes, H, W)
                     logger.debug("Assuming segmentation-like target shape based on CrossEntropyLoss and 4D output shape.")
                     return torch.randint(0, num_classes, (target_shape[0], *target_shape[2:]), device=output_device, dtype=torch.long)
                else:
                     logger.warning(f"Ambiguous output shape {output.shape} for CrossEntropyLoss. Generating target shape {output.shape[0]}. Provide target_generator if needed.")
                     return torch.randint(0, num_classes, (target_shape[0],), device=output_device, dtype=torch.long) # Fallback

            elif isinstance(criterion, nn.NLLLoss):
                 num_classes = output.shape[-1]
                 if len(output.shape) == 2: # (batch, classes)
                      return torch.randint(0, num_classes, (target_shape[0],), device=output_device, dtype=torch.long)
                 else: # Cas spatial ?
                      logger.warning(f"Using NLLLoss with non-2D output shape {output.shape}. Assuming segmentation target. Provide target_generator if needed.")
                      return torch.randint(0, num_classes, (target_shape[0], *target_shape[2:]), device=output_device, dtype=torch.long)

            elif isinstance(criterion, (nn.MSELoss, nn.L1Loss, nn.SmoothL1Loss)):
                return torch.randn_like(output)

            elif isinstance(criterion, (nn.BCELoss, nn.BCEWithLogitsLoss)):
                 return torch.rand_like(output) # Use rand for [0, 1) range

            else:
                 logger.warning(f"Unsupported criterion type {type(criterion).__name__}. Using zeros_like. Provide target_generator if needed.")
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
                      profiler_row_limit: int = 10,
                      input_dtype: torch.dtype = torch.float32 # Permettre de spécifier dtype
                     ) -> Dict[str, Any]:
        """
        Profiles the model's inference performance.

        Args:
            input_shape (Tuple[int, ...]): The shape of a single input batch.
            iterations (int): Number of inference iterations to time.
            warmup (int): Number of warmup iterations to run before timing.
            use_torch_profiler (bool): Whether to use the detailed torch.profiler.
            profiler_activities (Optional[list]): Activities for torch.profiler. Autodetected.
            profiler_sort_by (str): Key to sort the torch.profiler table by.
            profiler_row_limit (int): Number of rows to show in the profiler table.
            input_dtype (torch.dtype): Data type for the dummy input tensor.

        Returns:
            Dict[str, Any]: Profiling results (times, throughput, memory, profiler data).
        """
        validate_positive_integer(iterations, "Profiling iterations")
        validate_positive_integer(warmup, "Profiling warmup iterations", allow_zero=True)
        if not isinstance(input_shape, tuple) or not all(isinstance(d, int) and d > 0 for d in input_shape):
            raise ValueError(f"input_shape must be a tuple of positive integers, got {input_shape}.")
        if not isinstance(input_dtype, torch.dtype):
            raise TypeError(f"input_dtype must be a torch.dtype, got {type(input_dtype)}")

        logger.info(f"[Inference Profiling] Starting: input_shape={input_shape}, dtype={input_dtype}, iterations={iterations}, warmup={warmup}, device={self.device}, use_torch_profiler={use_torch_profiler}")

        original_mode_is_train = self.model.training
        self.model.eval() # <<< Set to evaluation mode
        try:
             dummy_input = self._generate_dummy_input(input_shape, dtype=input_dtype)
        except ValueError as e:
             self.model.train(mode=original_mode_is_train)
             raise e

        results: Dict[str, Any] = {
            "profiling_type": "inference",
            "input_shape": input_shape,
            "input_dtype": str(input_dtype),
            "device": str(self.device),
            "iterations": iterations,
            "warmup": warmup,
            "use_torch_profiler": use_torch_profiler,
            "error": None,
            "profiler_data": None,
            "profiler_error": None
        }

        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()

        profiler_instance = None # Définir hors du bloc try principal pour le finally

        try:
             with torch.no_grad(): # <<< No gradients for inference
                 # --- Warmup phase ---
                 logger.debug("[Inference Profiling] Running warmup...")
                 for _ in range(warmup): _ = self.model(dummy_input)
                 if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
                 logger.debug("[Inference Profiling] Warmup complete.")

                 # --- Simple timing ---
                 logger.debug("[Inference Profiling] Running timed iterations...")
                 start_time = time.perf_counter()
                 for _ in range(iterations): _ = self.model(dummy_input)
                 if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
                 end_time = time.perf_counter()

                 total_time_sec = end_time - start_time
                 avg_total_time_sec = total_time_sec / iterations if iterations > 0 else 0
                 throughput_batches = iterations / total_time_sec if total_time_sec > 0 else float('inf')

                 results["avg_total_time_ms"] = avg_total_time_sec * 1000
                 results["throughput_batches_per_sec"] = throughput_batches
                 results["throughput_samples_per_sec"] = throughput_batches * input_shape[0]
                 results["total_timed_duration_sec"] = total_time_sec
                 logger.info(f"[Inference Profiling] Basic timing complete: Avg time={results['avg_total_time_ms']:.3f} ms/batch, Throughput={results['throughput_samples_per_sec']:.1f} samples/sec")

                 # --- Detailed profiling using torch.profiler ---
                 prof = None # Définir avant le bloc try du profiler
                 if use_torch_profiler:
                     logger.info("[Inference Profiling] Running detailed profiling with torch.profiler...")
                     if profiler_activities is None:
                          profiler_activities = [ProfilerActivity.CPU]
                          if self.device.type == 'cuda': profiler_activities.append(ProfilerActivity.CUDA)
                     profile_iterations = min(iterations, 10); profile_warmup = min(warmup, 5); wait = 1
                     prof_schedule = schedule(wait=wait, warmup=profile_warmup, active=profile_iterations, repeat=1)
                     num_profiler_steps = wait + profile_warmup + profile_iterations

                     try:
                         # Utiliser le context manager directement est plus simple
                         with profile(activities=profiler_activities, record_shapes=True, profile_memory=True, with_stack=False, schedule=prof_schedule) as profiler_context:
                             for i in range(num_profiler_steps):
                                 with record_function(f"inference_iteration_{i}"): _ = self.model(dummy_input)
                                 profiler_context.step() # Utiliser la variable du contexte

                         prof = profiler_context # Assigner après la sortie du with pour l'analyse

                         logger.info("[Inference Profiling] Torch profiler run complete. Analyzing results...")
                         key_averages = prof.key_averages()
                         results["profiler_data"] = {}
                         if not key_averages: logger.warning("[Inference Profiling] Torch profiler recorded no events.")
                         else:
                             # --- Copier l'analyse du profiler ici ---
                             total_avg = key_averages.total_average()
                             results["profiler_data"]["total_events_averaged"] = len(key_averages)
                             cuda_time_total_us = getattr(total_avg, 'cuda_time_total', 0)
                             combined_time_us = total_avg.cpu_time_total + cuda_time_total_us
                             results["profiler_data"]["avg_cpu_time_total_ms"] = total_avg.cpu_time_total / 1000
                             results["profiler_data"]["avg_self_cpu_time_total_ms"] = total_avg.self_cpu_time_total / 1000
                             results["profiler_data"]["avg_cpu_time_percent"] = (total_avg.cpu_time_total / combined_time_us * 100) if combined_time_us > 0 else 0
                             results["profiler_data"]["avg_cuda_time_total_ms"] = 0; results["profiler_data"]["avg_self_cuda_time_total_ms"] = 0; results["profiler_data"]["avg_gpu_time_percent"] = 0
                             if self.device.type == 'cuda' and cuda_time_total_us > 0:
                                 results["profiler_data"]["avg_cuda_time_total_ms"] = cuda_time_total_us / 1000
                                 results["profiler_data"]["avg_self_cuda_time_total_ms"] = getattr(total_avg, 'self_cuda_time_total', 0) / 1000
                                 results["profiler_data"]["avg_gpu_time_percent"] = (cuda_time_total_us / combined_time_us * 100) if combined_time_us > 0 else 0
                             results["profiler_data"]["profiler_avg_cpu_memory_usage_b"] = getattr(total_avg, 'cpu_memory_usage', 0)
                             results["profiler_data"]["profiler_avg_cpu_memory_usage_formatted"] = format_bytes(results["profiler_data"]["profiler_avg_cpu_memory_usage_b"])
                             results["profiler_data"]["profiler_avg_gpu_memory_usage_b"] = 0; results["profiler_data"]["profiler_avg_gpu_memory_usage_formatted"] = "N/A"
                             if self.device.type == 'cuda':
                                results["profiler_data"]["profiler_avg_gpu_memory_usage_b"] = getattr(total_avg, 'cuda_memory_usage', 0)
                                results["profiler_data"]["profiler_avg_gpu_memory_usage_formatted"] = format_bytes(results["profiler_data"]["profiler_avg_gpu_memory_usage_b"])
                             sort_by_key = profiler_sort_by if profiler_sort_by else "self_cpu_time_total"
                             try:
                                results["profiler_data"]["profiler_top_ops_summary"] = key_averages.table(sort_by=sort_by_key, row_limit=profiler_row_limit)
                                logger.debug(f"[Inference Profiling] Top {profiler_row_limit} operators by {sort_by_key}:\n{results['profiler_data']['profiler_top_ops_summary']}")
                             except KeyError as ke:
                                 logger.warning(f"Could not sort profiler table by '{sort_by_key}' (KeyError: {ke}). Defaulting.")
                                 try: results["profiler_data"]["profiler_top_ops_summary"] = key_averages.table(sort_by="self_cpu_time_total", row_limit=profiler_row_limit)
                                 except Exception as table_err_fb: results["profiler_data"]["profiler_top_ops_summary"] = f"Error: {table_err_fb}"
                             except Exception as table_err:
                                 logger.error(f"Could not generate profiler table: {table_err}", exc_info=True)
                                 results["profiler_data"]["profiler_top_ops_summary"] = f"Error generating table: {table_err}"
                             # --- Fin de l'analyse du profiler ---

                     except Exception as prof_err:
                          logger.error(f"[Inference Profiling] Failed during torch.profiler execution/analysis: {prof_err}", exc_info=True)
                          results["profiler_error"] = f"Profiler run/analysis failed: {str(prof_err)}"
                     # Pas besoin de finally ici car le 'with' gère la sortie du contexte

                 # --- Capture peak memory overall after the run ---
                 if self.device.type == 'cuda':
                     results["max_memory_allocated_b"] = torch.cuda.max_memory_allocated(self.device)
                     results["max_memory_reserved_b"] = torch.cuda.max_memory_reserved(self.device)
                 else: # CPU
                     results["max_memory_allocated_b"] = results.get("profiler_data",{}).get("profiler_avg_cpu_memory_usage_b") # Estimate
                     results["max_memory_reserved_b"] = None # Non applicable

                 # Formater la mémoire
                 if results.get("max_memory_allocated_b") is not None:
                     results["max_memory_allocated_mb"] = results["max_memory_allocated_b"] / (1024**2)
                     results["max_memory_allocated_formatted"] = format_bytes(results["max_memory_allocated_b"])
                     if self.device.type != 'cuda' and use_torch_profiler and results.get("profiler_data"): results["max_memory_allocated_formatted"] += " (prof avg est.)"
                 else: results["max_memory_allocated_mb"] = None; results["max_memory_allocated_formatted"] = "N/A"
                 if results.get("max_memory_reserved_b") is not None: results["max_memory_reserved_formatted"] = format_bytes(results["max_memory_reserved_b"])
                 else: results["max_memory_reserved_formatted"] = "N/A"


        # --- Error Handling ---
        except torch.cuda.OutOfMemoryError as oom_err:
            logger.error(f"[Inference Profiling] CUDA OOM: {oom_err}", exc_info=False)
            results["error"] = "CUDA OutOfMemoryError"; results["error_details"] = str(oom_err)
            if self.device.type == 'cuda':
                try:
                     results["max_memory_allocated_b"] = torch.cuda.max_memory_allocated(self.device); results["max_memory_allocated_mb"] = results["max_memory_allocated_b"]/(1024**2); results["max_memory_allocated_formatted"] = format_bytes(results["max_memory_allocated_b"]); results["memory_required_at_oom_approx_mb"] = results["max_memory_allocated_mb"]
                except Exception: pass
                finally: torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"[Inference Profiling] Error: {e}", exc_info=True)
            if results["error"] is None: results["error"] = f"General profiling error: {str(e)}"
        finally:
             self.model.train(mode=original_mode_is_train) # <<< Restore original model mode
             logger.debug(f"Restored model training mode to: {original_mode_is_train}")

        return results


    def profile_training_step(self,
                              data_loader: Iterable,
                              criterion: nn.Module,
                              optimizer: Optimizer,
                              target_generator: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                              input_dtype: torch.dtype = torch.float32, # Permettre dtype pour les inputs
                              iterations: int = 10,
                              warmup: int = 3,
                              use_torch_profiler: bool = True,
                              profiler_activities: Optional[list] = None,
                              profiler_sort_by: str = "self_cpu_time_total",
                              profiler_row_limit: int = 15
                             ) -> Dict[str, Any]:
        """
        Profiles a full training step (data loading, forward, loss, backward, optimizer step),
        separating data fetch and preparation time.

        Args:
            data_loader (Iterable): An iterable (like `torch.utils.data.DataLoader`) that yields batches.
                                    Assumes batches are tuples/lists (input, target) or single input tensors.
            criterion (nn.Module): The loss function instance.
            optimizer (Optimizer): The optimizer instance.
            target_generator (Optional[Callable]): Function `target = func(output)` to generate targets
                                                   if loader yields only inputs or criterion needs specific format.
            input_dtype (torch.dtype): Data type expected for the model input tensors in the batch.
            iterations (int): Number of training steps to profile.
            warmup (int): Number of warmup training steps.
            use_torch_profiler (bool): Whether to use the detailed torch.profiler.
            profiler_activities (Optional[list]): Activities for torch.profiler. Autodetected.
            profiler_sort_by (str): Key to sort the torch.profiler table by.
            profiler_row_limit (int): Number of rows to show in the profiler table.

        Returns:
            Dict[str, Any]: Profiling results (step time breakdown, memory, profiler data).
        """
        validate_positive_integer(iterations, "Training profiling iterations", allow_zero=False)
        validate_positive_integer(warmup, "Training profiling warmup", allow_zero=True)
        if not isinstance(input_dtype, torch.dtype):
            raise TypeError(f"input_dtype must be a torch.dtype, got {type(input_dtype)}")

        logger.info(f"[Training Profiling] Starting: iterations={iterations}, warmup={warmup}, device={self.device}, use_torch_profiler={use_torch_profiler}")

        original_mode_is_train = self.model.training
        self.model.train() # <<< Set model to training mode
        try: criterion.to(self.device)
        except Exception as e: logger.warning(f"Could not move criterion to device {self.device}: {e}")

        # Use itertools.cycle for robust iteration over potentially short dataloaders
        try:
            cycled_loader = cycle(data_loader)
            # Try getting one batch to check loader validity early
            _ = next(cycled_loader)
            # Recreate cycle to ensure we start from the beginning for warmup + profiling runs
            cycled_loader = cycle(data_loader)
        except Exception as loader_err:
             logger.error(f"Failed to get data from data_loader using cycle: {loader_err}", exc_info=True)
             self.model.train(mode=original_mode_is_train) # Restore mode
             return {"error": f"DataLoader error: {loader_err}", "profiling_type": "training_step", "iterations_completed": 0}

        # Initialize results dictionary with default values
        results: Dict[str, Any] = {
            "profiling_type": "training_step", "input_dtype": str(input_dtype), "device": str(self.device),
            "iterations_requested": iterations, "iterations_completed": 0, "warmup": warmup,
            "use_torch_profiler": use_torch_profiler, "optimizer_type": type(optimizer).__name__,
            "criterion_type": type(criterion).__name__, "error": None, "warning": None,
            "profiler_data": None, "profiler_error": None,
            "total_profiled_duration_sec": 0.0, "avg_step_time_ms": 0.0,
            "avg_data_fetch_time_ms": 0.0, "avg_data_prep_time_ms": 0.0, "avg_data_total_load_time_ms": 0.0,
            "avg_forward_time_ms": 0.0, "avg_loss_time_ms": 0.0, "avg_backward_time_ms": 0.0,
            "avg_optimizer_time_ms": 0.0, "percent_time_data_fetch": 0.0, "percent_time_data_prep": 0.0,
            "percent_time_data_total_load": 0.0, "percent_time_forward": 0.0, "percent_time_loss": 0.0,
            "percent_time_backward": 0.0, "percent_time_optimizer": 0.0,
            "max_memory_allocated_b": None, "max_memory_allocated_mb": None, "max_memory_allocated_formatted": "N/A",
            "max_memory_reserved_b": None, "max_memory_reserved_formatted": "N/A"
        }

        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()

        # Store detailed step times
        step_times_sec = []; data_fetch_times_sec = []; data_prep_times_sec = []
        forward_times_sec = []; loss_times_sec = []; backward_times_sec = []; optimizer_times_sec = []

        profiler_instance = None; prof = None # Define outside try block

        try:
            # --- Warmup Phase ---
            logger.debug("[Training Profiling] Running warmup...")
            for wu_i in range(warmup):
                 try:
                     batch = next(cycled_loader)
                     if isinstance(batch, (list, tuple)): inputs, targets = batch[0].to(self.device, dtype=input_dtype), batch[1].to(self.device)
                     else: inputs = batch.to(self.device, dtype=input_dtype);
                     with torch.no_grad(): outputs_tmp = self.model(inputs); targets = (target_generator(outputs_tmp) if target_generator else self._generate_dummy_target(outputs_tmp, criterion)).to(self.device)
                     optimizer.zero_grad(set_to_none=True)
                     outputs = self.model(inputs)
                     loss = criterion(outputs, targets)
                     loss.backward()
                     optimizer.step()
                 except StopIteration: logger.error(f"[Training Profiling] DataLoader exhausted unexpectedly during warmup {wu_i+1}."); results["error"] = "DataLoader exhausted during warmup."; raise
                 except Exception as wu_err: logger.error(f"Error during warmup step {wu_i+1}: {wu_err}", exc_info=True); results["error"] = f"Error during warmup step: {wu_err}"; raise

            if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
            logger.debug("[Training Profiling] Warmup complete.")

            # --- Timed/Profiled Iterations ---
            logger.debug(f"[Training Profiling] Running {iterations} profiled iterations...")

            if use_torch_profiler:
                if profiler_activities is None: profiler_activities = [ProfilerActivity.CPU];
                if self.device.type == 'cuda': profiler_activities.append(ProfilerActivity.CUDA)
                profile_active = min(iterations, 5); profile_warmup = 1; wait = 0
                prof_schedule = schedule(wait=wait, warmup=profile_warmup, active=profile_active, repeat=1)
                num_profiler_steps = wait + profile_warmup + profile_active
                logger.info(f"[Training Profiling] Profiler schedule: wait={wait}, warmup={profile_warmup}, active={profile_active}")
                profiler_instance = profile(activities=profiler_activities, record_shapes=True, profile_memory=True, with_stack=False, schedule=prof_schedule)
                prof = profiler_instance.__enter__() # Entrée manuelle du contexte

            total_start_time = time.perf_counter()
            actual_iterations_run = 0

            for i in range(iterations):
                iter_start_time = time.perf_counter()
                inputs, targets = None, None

                # --- Data Fetching ---
                t0 = time.perf_counter()
                try:
                    with record_function("data_fetch"): batch = next(cycled_loader)
                except StopIteration: logger.warning(f"[Training Profiling] Cycled DataLoader exhausted after {i} valid iterations."); results["warning"] = f"Cycled DataLoader exhausted after {i} valid iterations."; break
                except Exception as fetch_err: logger.error(f"Error fetching data at iteration {i}: {fetch_err}", exc_info=True); results["error"] = f"Error fetching data at iteration {i}: {fetch_err}"; break
                t1 = time.perf_counter(); data_fetch_times_sec.append(t1 - t0)

                # --- Data Preparation & Move ---
                try:
                    with record_function("data_prep_move"):
                        if isinstance(batch, (list, tuple)): inputs = batch[0].to(device=self.device, dtype=input_dtype, non_blocking=True); targets_cpu = batch[1] if len(batch) > 1 else None
                        else: inputs = batch.to(device=self.device, dtype=input_dtype, non_blocking=True); targets_cpu = None
                        if targets_cpu is None: logger.debug(f"Batch {i} needs target generation.");
                        with torch.no_grad(): outputs_tmp = self.model(inputs); targets_cpu = target_generator(outputs_tmp) if target_generator else self._generate_dummy_target(outputs_tmp, criterion)
                        targets = targets_cpu.to(device=self.device, non_blocking=True)
                except Exception as data_err: logger.error(f"Error processing/moving batch {i}: {data_err}", exc_info=True); results["error"] = f"Error processing batch {i}: {data_err}"; break
                t2 = time.perf_counter(); data_prep_times_sec.append(t2 - t1)

                if inputs is None or targets is None: logger.error(f"Inputs or targets are None at iteration {i}."); results["error"] = f"Data prep failed at iteration {i}"; break

                # --- Forward, Loss, Backward, Optimizer ---
                try:
                    with record_function("forward_pass"): outputs = self.model(inputs)
                    t3 = time.perf_counter(); forward_times_sec.append(t3 - t2)
                    with record_function("loss_calculation"): loss = criterion(outputs, targets)
                    t4 = time.perf_counter(); loss_times_sec.append(t4 - t3)
                    optimizer.zero_grad(set_to_none=True)
                    with record_function("backward_pass"): loss.backward()
                    t5 = time.perf_counter(); backward_times_sec.append(t5 - t4)
                    with record_function("optimizer_step"): optimizer.step()
                    t6 = time.perf_counter(); optimizer_times_sec.append(t6 - t5)
                except Exception as step_err: logger.error(f"Error during Fwd/Loss/Bwd/Optim at iteration {i}: {step_err}", exc_info=True); results["error"] = f"Training step failed at iteration {i}: {step_err}"; break

                if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
                iter_end_time = time.perf_counter(); step_times_sec.append(iter_end_time - iter_start_time)
                actual_iterations_run += 1

                if prof and i < num_profiler_steps: prof.step()
            # --- End of Profiling Loop ---

            total_end_time = time.perf_counter()
            results["iterations_completed"] = actual_iterations_run # Update final count

            # Ensure profiler context is exited cleanly
            if profiler_instance: # Utiliser profiler_instance qui est défini avant le try
                try:
                    # Vérifier si le contexte a été effectivement entré avant de sortir
                    # Malheureusement, il n'y a pas d'API publique fiable pour ça.
                    # On suppose que si profiler_instance existe et use_torch_profiler est True,
                    # on doit appeler __exit__. C'est imparfait mais mieux que l'attribut privé.
                     if use_torch_profiler:
                          profiler_instance.__exit__(None, None, None)
                except Exception as prof_exit_err:
                     logger.error(f"Error exiting profiler context: {prof_exit_err}", exc_info=True)
                     # Ne pas écraser une erreur potentielle précédente du profiling lui-même
                     if results["profiler_error"] is None: results["profiler_error"] = f"Profiler exit error: {prof_exit_err}"


            # --- Profiler Analysis (if used and completed successfully) ---
            if use_torch_profiler and profiler_instance and results["error"] is None and actual_iterations_run > 0:
                logger.info("[Training Profiling] Analyzing detailed profiler results...")
                try:
                    key_averages = profiler_instance.key_averages()
                    results["profiler_data"] = {}
                    if not key_averages: logger.warning("[Training Profiling] Torch profiler did not record any events.")
                    else:
                        # --- Copier l'analyse du profiler ici ---
                        total_avg = key_averages.total_average()
                        results["profiler_data"]["total_events_averaged"] = len(key_averages)
                        cuda_time_total_us = getattr(total_avg, 'cuda_time_total', 0)
                        combined_time_us = total_avg.cpu_time_total + cuda_time_total_us
                        results["profiler_data"]["avg_cpu_time_total_ms"] = total_avg.cpu_time_total / 1000
                        results["profiler_data"]["avg_self_cpu_time_total_ms"] = total_avg.self_cpu_time_total / 1000
                        results["profiler_data"]["avg_cpu_time_percent"] = (total_avg.cpu_time_total / combined_time_us * 100) if combined_time_us > 0 else 0
                        results["profiler_data"]["avg_cuda_time_total_ms"] = 0; results["profiler_data"]["avg_self_cuda_time_total_ms"] = 0; results["profiler_data"]["avg_gpu_time_percent"] = 0
                        if self.device.type == 'cuda' and cuda_time_total_us > 0:
                            results["profiler_data"]["avg_cuda_time_total_ms"] = cuda_time_total_us / 1000
                            results["profiler_data"]["avg_self_cuda_time_total_ms"] = getattr(total_avg, 'self_cuda_time_total', 0) / 1000
                            results["profiler_data"]["avg_gpu_time_percent"] = (cuda_time_total_us / combined_time_us * 100) if combined_time_us > 0 else 0
                        results["profiler_data"]["profiler_avg_cpu_memory_usage_b"] = getattr(total_avg, 'cpu_memory_usage', 0)
                        results["profiler_data"]["profiler_avg_cpu_memory_usage_formatted"] = format_bytes(results["profiler_data"]["profiler_avg_cpu_memory_usage_b"])
                        results["profiler_data"]["profiler_avg_gpu_memory_usage_b"] = 0; results["profiler_data"]["profiler_avg_gpu_memory_usage_formatted"] = "N/A"
                        if self.device.type == 'cuda':
                            results["profiler_data"]["profiler_avg_gpu_memory_usage_b"] = getattr(total_avg, 'cuda_memory_usage', 0)
                            results["profiler_data"]["profiler_avg_gpu_memory_usage_formatted"] = format_bytes(results["profiler_data"]["profiler_avg_gpu_memory_usage_b"])
                        sort_by_key = profiler_sort_by if profiler_sort_by else "self_cpu_time_total"
                        try:
                            results["profiler_data"]["profiler_top_ops_summary"] = key_averages.table(sort_by=sort_by_key, row_limit=profiler_row_limit)
                            # logger.debug(f"[Training Profiling] Top {profiler_row_limit} operators by {sort_by_key}:\n{results['profiler_data']['profiler_top_ops_summary']}") # Log si besoin
                        except KeyError as ke:
                            logger.warning(f"Could not sort profiler table by '{sort_by_key}' (KeyError: {ke}). Defaulting.")
                            try: results["profiler_data"]["profiler_top_ops_summary"] = key_averages.table(sort_by="self_cpu_time_total", row_limit=profiler_row_limit)
                            except Exception as table_err_fb: results["profiler_data"]["profiler_top_ops_summary"] = f"Error: {table_err_fb}"
                        except Exception as table_err:
                            logger.error(f"Could not generate profiler table: {table_err}", exc_info=True)
                            results["profiler_data"]["profiler_top_ops_summary"] = f"Error generating table: {table_err}"
                        # --- Fin analyse profiler ---
                except Exception as prof_err:
                     logger.error(f"[Training Profiling] Failed during torch.profiler analysis: {prof_err}", exc_info=True)
                     results["profiler_error"] = f"Profiler analysis failed: {str(prof_err)}"


            # --- Basic Timing Analysis (only if iterations completed) ---
            if actual_iterations_run > 0:
                results["total_profiled_duration_sec"] = total_end_time - total_start_time
                results["avg_step_time_ms"] = (sum(step_times_sec) / actual_iterations_run) * 1000
                results["avg_data_fetch_time_ms"] = (sum(data_fetch_times_sec) / actual_iterations_run) * 1000
                results["avg_data_prep_time_ms"] = (sum(data_prep_times_sec) / actual_iterations_run) * 1000
                results["avg_data_total_load_time_ms"] = results["avg_data_fetch_time_ms"] + results["avg_data_prep_time_ms"]
                results["avg_forward_time_ms"] = (sum(forward_times_sec) / actual_iterations_run) * 1000
                results["avg_loss_time_ms"] = (sum(loss_times_sec) / actual_iterations_run) * 1000
                results["avg_backward_time_ms"] = (sum(backward_times_sec) / actual_iterations_run) * 1000
                results["avg_optimizer_time_ms"] = (sum(optimizer_times_sec) / actual_iterations_run) * 1000

                step_time_ms = results["avg_step_time_ms"]
                if step_time_ms > 1e-9: # Utiliser epsilon pour éviter division par zéro flottant
                    results["percent_time_data_fetch"] = max(0.0, min(100.0, (results["avg_data_fetch_time_ms"] / step_time_ms) * 100))
                    results["percent_time_data_prep"] = max(0.0, min(100.0, (results["avg_data_prep_time_ms"] / step_time_ms) * 100))
                    results["percent_time_data_total_load"] = max(0.0, min(100.0, (results["avg_data_total_load_time_ms"] / step_time_ms) * 100))
                    results["percent_time_forward"] = max(0.0, min(100.0, (results["avg_forward_time_ms"] / step_time_ms) * 100))
                    results["percent_time_loss"] = max(0.0, min(100.0, (results["avg_loss_time_ms"] / step_time_ms) * 100))
                    results["percent_time_backward"] = max(0.0, min(100.0, (results["avg_backward_time_ms"] / step_time_ms) * 100))
                    results["percent_time_optimizer"] = max(0.0, min(100.0, (results["avg_optimizer_time_ms"] / step_time_ms) * 100))
                # Garder 0.0 sinon

                logger.info(f"[Training Profiling] Basic timing breakdown (avg ms over {actual_iterations_run} steps): Step={step_time_ms:.2f}, DataFetch={results['avg_data_fetch_time_ms']:.2f} ({results['percent_time_data_fetch']:.1f}%), DataPrep={results['avg_data_prep_time_ms']:.2f} ({results['percent_time_data_prep']:.1f}%), Forward={results['avg_forward_time_ms']:.2f} ({results['percent_time_forward']:.1f}%), Loss={results['avg_loss_time_ms']:.2f} ({results['percent_time_loss']:.1f}%), Backward={results['avg_backward_time_ms']:.2f} ({results['percent_time_backward']:.1f}%), Optimizer={results['avg_optimizer_time_ms']:.2f} ({results['percent_time_optimizer']:.1f}%)")
            elif results["error"] is None: # No iterations ran, but no explicit error reported yet
                logger.warning("[Training Profiling] No iterations completed successfully. Timing results unavailable.")
                results["error"] = "No iterations completed for timing."

            # --- Capture overall peak memory after the run ---
            if self.device.type == 'cuda':
                results["max_memory_allocated_b"] = torch.cuda.max_memory_allocated(self.device)
                results["max_memory_reserved_b"] = torch.cuda.max_memory_reserved(self.device)
            else: # CPU estimate
                results["max_memory_allocated_b"] = results.get("profiler_data",{}).get("profiler_avg_cpu_memory_usage_b")
                results["max_memory_reserved_b"] = None

            # Formater la mémoire
            if results.get("max_memory_allocated_b") is not None:
                results["max_memory_allocated_mb"] = results["max_memory_allocated_b"] / (1024**2)
                results["max_memory_allocated_formatted"] = format_bytes(results["max_memory_allocated_b"])
                if self.device.type != 'cuda' and use_torch_profiler and results.get("profiler_data"): results["max_memory_allocated_formatted"] += " (prof avg est.)"
            else: results["max_memory_allocated_mb"] = None; results["max_memory_allocated_formatted"] = "N/A"
            if results.get("max_memory_reserved_b") is not None: results["max_memory_reserved_formatted"] = format_bytes(results["max_memory_reserved_b"])
            else: results["max_memory_reserved_formatted"] = "N/A"


        # --- Error Handling for the entire process ---
        except torch.cuda.OutOfMemoryError as oom_err:
            logger.error(f"[Training Profiling] CUDA OOM: {oom_err}", exc_info=False)
            results["error"] = "CUDA OutOfMemoryError"; results["error_details"] = str(oom_err)
            if self.device.type == 'cuda':
                try:
                     results["max_memory_allocated_b"] = torch.cuda.max_memory_allocated(self.device); results["max_memory_allocated_mb"] = results["max_memory_allocated_b"]/(1024**2); results["max_memory_allocated_formatted"] = format_bytes(results["max_memory_allocated_b"]); results["memory_required_at_oom_approx_mb"] = results["max_memory_allocated_mb"]
                except Exception: pass
                finally: torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"[Training Profiling] Error: {e}", exc_info=True)
            if results["error"] is None: results["error"] = f"General training profiling error: {str(e)}"
        finally:
            # Ensure context manager is exited if manually managed and instance exists
            # if use_torch_profiler and profiler_instance and prof is profiler_instance: # Check if prof points to the instance
            #     try:
            #         # This check is heuristic and might not be perfectly reliable
            #         # A better approach might be needed if context management is complex
            #         if getattr(profiler_instance, '_entered', False): # Hypothetical check
            #             profiler_instance.__exit__(None, None, None)
            #     except Exception as final_prof_exit_err:
            #         logger.error(f"Error during final profiler exit in finally block: {final_prof_exit_err}")
            # Restore original model training mode
            self.model.train(mode=original_mode_is_train)
            logger.debug(f"Restored model training mode to: {original_mode_is_train}")


        # Final adjustments to results dict
        results["iterations"] = actual_iterations_run # Report actual iterations completed
        if results["iterations_requested"] == actual_iterations_run:
            results.pop("iterations_completed", None) # Remove if redundant
            results.pop("iterations_requested", None)
        # elif "iterations_completed" not in results: # Add if loop was exited early before assignment
        #     results["iterations_completed"] = actual_iterations_run


        return results