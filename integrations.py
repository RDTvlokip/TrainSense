# TrainSense/integrations.py
import time
import logging
from typing import Dict, Any, Optional, List, Union
import torch
import torch.nn as nn
from torch.optim import Optimizer

# Import other TrainSense components needed
from .model_profiler import ModelProfiler
from .gradient_analyzer import GradientAnalyzer
from .monitoring import RealTimeMonitor

logger = logging.getLogger(__name__)

# --- PyTorch Hook Based Integration ---
# (TrainStepMonitorHook class remains unchanged)
class TrainStepMonitorHook:
    """
    Uses PyTorch forward and backward hooks to monitor aspects of a training step.
    This is a basic example demonstrating the concept.
    Can be attached to specific modules or the main model.
    """
    def __init__(self, model: nn.Module, log_level: int = logging.INFO):
        self.model = model
        self.log_level = log_level
        self.handles = []
        self.forward_start_time: Optional[float] = None
        self.forward_times: List[float] = []
        self.backward_start_time: Optional[float] = None
        self.backward_times: List[float] = []
        self.batch_errors = 0
        self.total_batches = 0
        self.current_loss: Optional[float] = None
        self._hook_enabled = False

    def _forward_pre_hook(self, module, input):
        """Hook executed before forward pass."""
        self.forward_start_time = time.perf_counter()

    def _forward_hook(self, module, input, output):
        """Hook executed after forward pass."""
        if self.forward_start_time is not None:
            duration = time.perf_counter() - self.forward_start_time
            self.forward_times.append(duration)
        self.forward_start_time = None # Reset for next call

    def _backward_hook(self, module, grad_input, grad_output):
        """Hook executed after backward pass (for the specific module)."""
        if self.backward_start_time is None:
            self.backward_start_time = time.perf_counter()

    def _full_backward_hook(self, grad):
        """Hook executed after the *entire* backward pass is done (attached to loss usually)."""
        if self.backward_start_time is not None:
             duration = time.perf_counter() - self.backward_start_time
             self.backward_times.append(duration)
        self.backward_start_time = None # Reset

    def attach_hooks(self, modules_to_hook: Optional[List[nn.Module]] = None):
        """Attaches hooks to the model or specified modules."""
        if self._hook_enabled: logger.warning("Hooks are already attached."); return
        if modules_to_hook is None: modules_to_hook = [self.model]
        self.handles = []
        for module in modules_to_hook:
            self.handles.append(module.register_forward_pre_hook(self._forward_pre_hook))
            self.handles.append(module.register_forward_hook(self._forward_hook))
            # self.handles.append(module.register_full_backward_hook(self._backward_hook))
        logger.info(f"Attached forward hooks to {len(modules_to_hook)} module(s). Attach backward hook to loss tensor externally.")
        self._hook_enabled = True

    def detach_hooks(self):
        """Removes all attached hooks."""
        for handle in self.handles: handle.remove()
        self.handles = []; self._hook_enabled = False
        logger.info("Detached all TrainSense hooks.")

    def reset_stats(self):
        """Resets collected statistics."""
        self.forward_times = []; self.backward_times = []
        self.batch_errors = 0; self.total_batches = 0; self.current_loss = None
        self.forward_start_time = None; self.backward_start_time = None
        logger.info("Hook statistics reset.")

    def record_batch_start(self): self.total_batches += 1; self.backward_start_time = None
    def record_batch_error(self): self.batch_errors += 1

    def record_loss(self, loss_tensor: torch.Tensor):
         if not self._hook_enabled: return
         self.current_loss = loss_tensor.item()
         if hasattr(loss_tensor, 'grad_fn') and loss_tensor.grad_fn is not None:
              if self.backward_start_time is None:
                   self.backward_start_time = time.perf_counter()
                   try: # Use try-except as register_hook might fail in some edge cases
                        loss_tensor.register_hook(self._full_backward_hook)
                   except Exception as hook_err:
                        logger.error(f"Failed to register backward hook on loss tensor: {hook_err}", exc_info=True)
                        self.backward_start_time = None # Reset timer if hook failed
         else: logger.warning("Loss tensor does not require grad, cannot attach backward hook.")

    def get_summary(self) -> Dict[str, Any]:
        """Returns a summary of collected statistics."""
        # (Summary calculation logic remains the same)
        summary = {
            "total_batches_processed": self.total_batches,
            "batch_error_count": self.batch_errors,
            "batch_error_rate": (self.batch_errors / self.total_batches * 100) if self.total_batches > 0 else 0,
            "avg_forward_time_ms": (sum(self.forward_times) / len(self.forward_times) * 1000) if self.forward_times else None,
            "avg_backward_time_ms": (sum(self.backward_times) / len(self.backward_times) * 1000) if self.backward_times else None,
            "last_recorded_loss": self.current_loss,
            "num_forward_timings": len(self.forward_times),
            "num_backward_timings": len(self.backward_times),
        }
        return summary

    def __enter__(self): self.attach_hooks(); self.reset_stats(); return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.detach_hooks(); logger.info(f"Hook Summary on exit: {self.get_summary()}")


# --- TRL Callback Integration ---
try:
    # Attempt to import the actual classes
    from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    # Define dummy classes if transformers is not installed
    # These dummies allow type hints using strings without causing runtime errors
    class TrainerCallback: pass
    class TrainingArguments: pass
    class TrainerState: pass
    class TrainerControl: pass
    TRANSFORMERS_AVAILABLE = False
    logger.info("Transformers library not found. TRL integration features will be unavailable.")

class TrainSenseTRLCallback(TrainerCallback if TRANSFORMERS_AVAILABLE else object): # Inherit from object if not available
    """
    A Hugging Face TrainerCallback to integrate TrainSense monitoring during SFTTrainer runs.
    Collects gradient stats, timing info (basic), and resource usage.
    """
    def __init__(self,
                 gradient_analyzer: Optional[GradientAnalyzer] = None,
                 real_time_monitor: Optional[RealTimeMonitor] = None,
                 log_grads_every_n_steps: int = 50,
                 log_level: int = logging.INFO):
        if not TRANSFORMERS_AVAILABLE:
            # Log warning instead of raising error to allow conditional usage
            logger.warning("Attempting to initialize TrainSenseTRLCallback, but 'transformers' library is not installed.")
            # raise ImportError("Transformers library is required to use TrainSenseTRLCallback.") # Or raise error

        self.gradient_analyzer = gradient_analyzer
        self.real_time_monitor = real_time_monitor
        self.log_grads_every_n_steps = log_grads_every_n_steps
        self.log_level=log_level
        self.step_start_time: Optional[float] = None
        self.step_times: List[float] = []
        self.batch_error_count = 0
        self.monitor_history_on_end: List[Dict[str, Any]] = []

    # --- Use STRINGS for type hints from transformers ---
    def on_train_begin(self, args: 'TrainingArguments', state: 'TrainerState', control: 'TrainerControl', **kwargs):
        if not TRANSFORMERS_AVAILABLE: return # Do nothing if library isn't there
        logger.log(self.log_level, "*** TrainSense TRL Callback: Training Started ***")
        self.step_times = []
        self.batch_error_count = 0
        if self.real_time_monitor:
            logger.log(self.log_level, "Starting real-time resource monitoring.")
            self.real_time_monitor.start()

    def on_step_begin(self, args: 'TrainingArguments', state: 'TrainerState', control: 'TrainerControl', **kwargs):
        if not TRANSFORMERS_AVAILABLE: return
        self.step_start_time = time.perf_counter()

    def on_step_end(self, args: 'TrainingArguments', state: 'TrainerState', control: 'TrainerControl', **kwargs):
        if not TRANSFORMERS_AVAILABLE: return
        if self.step_start_time:
            duration = time.perf_counter() - self.step_start_time
            self.step_times.append(duration)
            # Log step time more selectively, e.g., every N steps or if slow
            if state.global_step % 20 == 0: # Log every 20 steps
                 logger.log(self.log_level, f"Step {state.global_step} duration: {duration*1000:.2f} ms")
            self.step_start_time = None

        # Gradient Analysis (check if model is available in kwargs)
        model = kwargs.get('model')
        if model and self.gradient_analyzer and state.global_step > 0 and state.global_step % self.log_grads_every_n_steps == 0:
            logger.log(self.log_level, f"Analyzing gradients at step {state.global_step}...")
            try:
                if self.gradient_analyzer.model is not model:
                     logger.warning("GradientAnalyzer holds a different model instance than TRL trainer.")
                     # Decide if we should update GA's model: self.gradient_analyzer.model = model
                grad_summary = self.gradient_analyzer.summary() # Assumes backward just ran
                logger.log(self.log_level, f"Step {state.global_step} Grad Summary: {grad_summary}")
                # Add to Trainer state logs if possible
                if state.log_history is not None and isinstance(state.log_history, list):
                     # Find the log entry for the current step
                     for log_entry in reversed(state.log_history):
                         if log_entry.get('step') == state.global_step:
                             log_entry['grad_norm_L2'] = grad_summary.get('global_grad_norm_L2')
                             log_entry['grads_nan'] = grad_summary.get('num_params_nan_grad')
                             log_entry['grads_inf'] = grad_summary.get('num_params_inf_grad')
                             break
            except Exception as e:
                 logger.error(f"Error during gradient analysis at step {state.global_step}: {e}", exc_info=True)
        elif self.gradient_analyzer and not model:
             logger.warning(f"Gradient analysis requested at step {state.global_step}, but model not found in callback kwargs.")


    # on_exception might not always be reliable for batch errors depending on trainer implementation
    # def on_exception(self, args: 'TrainingArguments', state: 'TrainerState', control: 'TrainerControl', exception: Exception, **kwargs):
    #     if not TRANSFORMERS_AVAILABLE: return
    #     self.batch_error_count += 1
    #     logger.error(f"Exception caught during training at step {state.global_step}: {exception}", exc_info=True)

    def on_train_end(self, args: 'TrainingArguments', state: 'TrainerState', control: 'TrainerControl', **kwargs):
        if not TRANSFORMERS_AVAILABLE: return
        logger.log(self.log_level, "*** TrainSense TRL Callback: Training Ended ***")
        if self.real_time_monitor:
            logger.log(self.log_level, "Stopping real-time resource monitoring.")
            self.monitor_history_on_end = self.real_time_monitor.stop()
            # Optional: Log summary of resource usage
            # if self.monitor_history_on_end: ... process history ...

        # Log final training step stats
        if self.step_times:
            avg_step_time_ms = (sum(self.step_times) / len(self.step_times)) * 1000
            logger.log(self.log_level, f"Average step time over {len(self.step_times)} steps: {avg_step_time_ms:.2f} ms")
        logger.log(self.log_level, f"Total batch processing errors detected by callback (if implemented): {self.batch_error_count}")