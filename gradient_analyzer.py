# TrainSense/gradient_analyzer.py
# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, List, Tuple

# Use try-except for optional import of visualization components
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
    # Configure matplotlib backend non-interactively if possible (useful for server environments)
    # plt.switch_backend('Agg') # Uncomment if running in a headless environment and saving plots
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    mtick = None
    np = None
    # Log only once at import time if matplotlib/numpy are missing
    logging.getLogger(__name__).info(
        "matplotlib or numpy not found, plotting features in GradientAnalyzer will be disabled. "
        "Install with 'pip install trainsense[plotting]'"
    )

# Get a logger instance specific to this module
logger = logging.getLogger(__name__)

class GradientAnalyzer:
    """
    Analyzes gradient statistics for model parameters after a backward pass.

    Provides per-parameter statistics (norm, mean, std, min, max, NaN/Inf count),
    calculates the global gradient norm, and offers optional histogram plotting
    of gradient norms across layers.

    IMPORTANT: This analyzer requires `model.backward()` to have been called *before*
    its analysis methods (`analyze_gradients`, `summary`) are invoked, as it inspects
    the `.grad` attribute of parameters.
    """
    def __init__(self, model: nn.Module):
        """
        Initializes the GradientAnalyzer.

        Args:
            model (nn.Module): The PyTorch model whose gradients will be analyzed.

        Raises:
            TypeError: If the input 'model' is not an instance of torch.nn.Module.
        """
        if not isinstance(model, nn.Module):
            raise TypeError("Input 'model' must be an instance of torch.nn.Module.")
        self.model = model
        logger.info(f"GradientAnalyzer initialized for model type: {type(model).__name__}")
        # Cache for the detailed statistics of the last analysis run
        self._last_grad_stats: Optional[Dict[str, Dict[str, Any]]] = None
        # Cache for the last computed global norm
        self._last_global_grad_norm: Optional[float] = None

    @torch.no_grad() # Ensure no gradients are computed during analysis
    def analyze_gradients(self, norm_type: float = 2.0) -> Dict[str, Dict[str, Any]]:
        """
        Calculates statistics for the gradients of all trainable parameters that currently have gradients.
        Requires `model.backward()` to have been called recently.

        Args:
            norm_type (float): The type of norm to compute for per-parameter gradient norms
                               (e.g., 2.0 for L2 norm, 1.0 for L1, float('inf') for max absolute value).
                               Defaults to 2.0 (L2 norm).

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary where keys are parameter names (e.g., 'layer1.weight')
                                      and values are dictionaries containing gradient statistics for that parameter:
                                      - 'shape' (Tuple[int, ...]): Shape of the parameter/gradient.
                                      - 'has_grad' (bool): Whether a .grad attribute was found.
                                      - 'is_nan' (bool): True if any gradient value is NaN.
                                      - 'is_inf' (bool): True if any gradient value is Inf.
                                      - 'norm' (Optional[float]): The calculated norm of the gradient tensor (using `norm_type`). NaN if NaN/Inf present.
                                      - 'mean' (Optional[float]): Mean of the gradient values. NaN if NaN/Inf present.
                                      - 'std' (Optional[float]): Standard deviation of gradient values. NaN if NaN/Inf present.
                                      - 'min' (Optional[float]): Minimum gradient value. NaN if NaN/Inf present.
                                      - 'max' (Optional[float]): Maximum gradient value. NaN if NaN/Inf present.
                                      - 'abs_mean' (Optional[float]): Mean of the absolute gradient values. NaN if NaN/Inf present.
                                      - 'grad_param_norm_ratio' (Optional[float]): Ratio of gradient norm to parameter norm (same `norm_type`).
                                                                                   Can be NaN or Inf if parameter norm is zero or grads are NaN/Inf.
                                      Returns an empty dict if no parameters require gradients or none have `.grad` populated.
        """
        logger.info(f"Starting gradient analysis (per-parameter norm_type={norm_type})...")
        grad_stats: Dict[str, Dict[str, Any]] = {}
        found_grads_count = 0

        # Iterate through all named parameters in the model
        for name, param in self.model.named_parameters():
            stats: Dict[str, Any] = {
                "shape": tuple(param.shape),
                "has_grad": False, # Assume no grad initially
                "is_nan": False, "is_inf": False,
                "norm": None, "mean": None, "std": None, "min": None, "max": None,
                "abs_mean": None, "grad_param_norm_ratio": None
            }

            # Skip parameters that don't require gradients or haven't received any yet
            if not param.requires_grad:
                grad_stats[name] = stats # Store basic info even if no grad required
                continue

            if param.grad is None:
                # Parameter requires grad but hasn't received one (e.g., part of model not used in loss path)
                logger.debug(f"Parameter '{name}' requires grad but has no grad attribute yet.")
                grad_stats[name] = stats # Store basic info
                continue

            # If we reach here, the parameter requires grad and has a .grad attribute
            stats["has_grad"] = True
            found_grads_count += 1
            # Detach gradient tensor to prevent tracking, convert to float for stability, work on its original device
            grad_data = param.grad.detach().float()

            # --- Check for NaN/Inf first ---
            stats["is_nan"] = torch.isnan(grad_data).any().item()
            stats["is_inf"] = torch.isinf(grad_data).any().item()

            if stats["is_nan"] or stats["is_inf"]:
                # Log a warning and set numerical stats accordingly
                nan_inf_msg = f"NaN ({stats['is_nan']})" if stats["is_nan"] else ""
                nan_inf_msg += " and " if stats["is_nan"] and stats["is_inf"] else ""
                nan_inf_msg += f"Inf ({stats['is_inf']})" if stats["is_inf"] else ""
                logger.warning(f"Gradient for '{name}' contains {nan_inf_msg}. Numerical stats will be NaN/Inf.")
                stats["norm"] = float('inf') if stats["is_inf"] else float('nan') # Prioritize Inf over NaN for norm? Or always NaN? Let's use NaN.
                stats["norm"] = float('nan')
                stats["mean"] = float('nan')
                stats["std"] = float('nan')
                stats["min"] = float('nan') # Or potentially -Inf / Inf ? Let's use NaN for consistency.
                stats["max"] = float('nan')
                stats["abs_mean"] = float('nan')
                stats["grad_param_norm_ratio"] = float('nan') # Ratio is undefined
            else:
                 # --- Calculate statistics only if gradients are finite ---
                 try:
                    # Flatten for norm calculation, keep original shape for others
                    grad_flat = grad_data.flatten()
                    # Calculate specified norm (L2, L1, etc.)
                    grad_norm_val = torch.linalg.norm(grad_flat, ord=norm_type).item()
                    stats["norm"] = grad_norm_val

                    # Calculate other descriptive statistics on the original tensor
                    stats["mean"] = grad_data.mean().item()
                    stats["std"] = grad_data.std().item()
                    stats["min"] = grad_data.min().item()
                    stats["max"] = grad_data.max().item()
                    stats["abs_mean"] = grad_data.abs().mean().item()

                    # --- Calculate grad/param norm ratio ---
                    param_data_flat = param.data.detach().float().flatten()
                    # Use the same norm type for the parameter tensor
                    param_norm_val = torch.linalg.norm(param_data_flat, ord=norm_type).item()

                    # Handle division by zero or near-zero parameter norm
                    if param_norm_val > 1e-12: # Use a small epsilon for float comparison
                        stats["grad_param_norm_ratio"] = grad_norm_val / param_norm_val
                    else:
                        # If param norm is effectively zero:
                        # - If grad norm is also zero, ratio is 0 (or NaN? Let's use 0)
                        # - If grad norm is non-zero, ratio is Inf
                        stats["grad_param_norm_ratio"] = 0.0 if grad_norm_val < 1e-12 else float('inf')
                        logger.debug(f"Parameter '{name}' norm is close to zero ({param_norm_val:.2e}). Grad/Param ratio set to {stats['grad_param_norm_ratio']}.")

                 except Exception as stat_err:
                     # Catch potential errors during calculation (e.g., on specific devices/dtypes)
                     logger.error(f"Error calculating statistics for finite gradient of '{name}': {stat_err}", exc_info=True)
                     # Set stats to NaN if calculation failed unexpectedly
                     stats["norm"] = stats["mean"] = stats["std"] = stats["min"] = stats["max"] = stats["abs_mean"] = stats["grad_param_norm_ratio"] = float('nan')

            # Store the calculated stats for this parameter
            grad_stats[name] = stats

        # Log summary message
        if found_grads_count == 0:
            logger.warning("No gradients found in any parameters requiring them. Ensure `model.backward()` was called before analysis.")
        else:
             logger.info(f"Gradient analysis complete. Found gradients for {found_grads_count} parameters.")

        # Cache the results for potential reuse (e.g., by summary() or plotting)
        self._last_grad_stats = grad_stats
        # Invalidate cached global norm as per-parameter stats have been updated
        self._last_global_grad_norm = None
        return grad_stats

    @torch.no_grad()
    def calculate_global_gradient_norm(self, norm_type: float = 2.0) -> Optional[float]:
         """
         Calculates the total norm of gradients across all parameters that require gradients
         and currently have them. Uses `torch.nn.utils.clip_grad_norm_` without clipping.

         Args:
             norm_type (float): The type of norm to compute (e.g., 2.0 for L2 norm). Defaults to 2.0.

         Returns:
             Optional[float]: The computed global gradient norm. Returns None if no parameters
                              with gradients are found or if an error occurs during calculation.
                              Can return NaN or Inf if gradients contain these values.
         """
         # Filter parameters: must require grad and have a non-None .grad attribute
         parameters_with_grads = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]

         if not parameters_with_grads:
             logger.warning("No parameters with gradients found to compute global norm.")
             self._last_global_grad_norm = None # Ensure cache is None
             return None

         try:
             # Use clip_grad_norm_ which calculates and *returns* the total norm *before* clipping.
             # We pass float('inf') as max_norm to prevent any actual clipping.
             device = parameters_with_grads[0].device # Use device of first parameter
             # Note: This assumes all parameters with grads are on the same device, which is typical.
             # If parameters are spread across devices *without* model parallelism frameworks, this might need adjustment.

             # Create tensors on the correct device for norm calculation if needed by clip_grad_norm_
             # However, clip_grad_norm_ should handle device placement internally.

             # Calculate the global norm
             total_norm = torch.nn.utils.clip_grad_norm_(
                 parameters=parameters_with_grads,
                 max_norm=float('inf'), # No actual clipping
                 norm_type=norm_type
             )

             # Convert the result (which might be a Tensor) to a Python float
             total_norm_float = total_norm.item()

             # Check for NaN/Inf in the final result (can happen if inputs had issues)
             if not torch.isfinite(torch.tensor(total_norm_float)):
                  logger.warning(f"Computed global gradient norm (type {norm_type}) is not finite: {total_norm_float}. May indicate NaN/Inf in gradients.")
                  # Keep the NaN/Inf value as it's informative

             self._last_global_grad_norm = total_norm_float # Cache the result
             logger.debug(f"Calculated global gradient norm (type {norm_type}): {total_norm_float:.3e}")
             return total_norm_float

         except RuntimeError as e:
             # Catch potential runtime errors during norm calculation
             logger.error(f"RuntimeError computing global gradient norm (type {norm_type}): {e}. Gradients might contain unsupported values or issues.", exc_info=True)
             self._last_global_grad_norm = None # Invalidate cache on error
             return None # Indicate failure
         except Exception as e:
             # Catch any other unexpected errors
             logger.error(f"Unexpected error computing global gradient norm (type {norm_type}): {e}", exc_info=True)
             self._last_global_grad_norm = None
             return None

    def summary(self, force_recompute: bool = False, default_norm_type: float = 2.0) -> Dict[str, Any]:
         """
         Provides a summary of gradient statistics across all layers based on the last analysis.
         Re-runs analysis and global norm calculation if needed or forced.

         Args:
             force_recompute (bool): If True, forces re-computation of per-parameter stats
                                     and global norm, even if cached data exists. Defaults to False.
             default_norm_type (float): The norm type to use if per-parameter analysis needs to be re-run.
                                        The global norm reported is typically L2. Defaults to 2.0.

         Returns:
             Dict[str, Any]: A dictionary summarizing gradient statistics:
                             - 'num_params_analyzed' (int): Total parameters considered (those requiring grad).
                             - 'num_params_with_grads' (int): Parameters that had a `.grad` attribute.
                             - 'num_params_nan_grad' (int): Count of params with NaN gradients.
                             - 'num_params_inf_grad' (int): Count of params with Inf gradients.
                             - 'global_grad_norm_L2' (Optional[float]): The L2 global gradient norm.
                             - 'avg_grad_norm' (Optional[float]): Average of per-parameter L2 norms (excluding NaN/Inf).
                             - 'max_grad_norm' (Optional[float]): Maximum per-parameter L2 norm (excluding NaN/Inf).
                             - 'min_grad_norm' (Optional[float]): Minimum per-parameter L2 norm (excluding NaN/Inf).
                             - 'avg_grad_mean' (Optional[float]): Average of per-parameter gradient means.
                             - 'avg_grad_std' (Optional[float]): Average of per-parameter gradient std deviations.
                             - 'avg_grad_param_norm_ratio' (Optional[float]): Average of grad/param norm ratios.
                             - 'layer_with_max_grad_norm' (Optional[str]): Name of the parameter with the highest L2 norm.
                             - 'error' (Optional[str]): Error message if summary generation failed.
         """
         logger.info(f"Generating gradient summary (force_recompute={force_recompute})...")

         # --- Get Per-Parameter Stats ---
         # Use cached stats unless forced or cache is empty
         if self._last_grad_stats is None or force_recompute:
             logger.debug("Cache miss or force_recompute=True. Running analyze_gradients...")
             stats_per_layer = self.analyze_gradients(norm_type=default_norm_type)
             # Global norm cache is implicitly invalidated by analyze_gradients
         else:
             logger.debug("Using cached gradient statistics.")
             stats_per_layer = self._last_grad_stats

         # Handle case where analysis yielded no results
         if not stats_per_layer:
             logger.warning("No gradient statistics available to summarize.")
             return {"error": "No gradient statistics found (was model.backward() called?)."}

         # --- Calculate Global Norm (L2) ---
         # Use cached global norm unless forced or cache is empty/invalidated
         if self._last_global_grad_norm is None or force_recompute:
              logger.debug("Cache miss or force_recompute=True for global norm. Running calculate_global_gradient_norm...")
              global_grad_norm_l2 = self.calculate_global_gradient_norm(norm_type=2.0) # Standard L2 norm
         else:
              logger.debug("Using cached global gradient norm.")
              global_grad_norm_l2 = self._last_global_grad_norm


         # --- Aggregate Summary Statistics ---
         num_analyzed = len(stats_per_layer)
         num_with_grads = sum(1 for s in stats_per_layer.values() if s.get('has_grad'))
         num_nan = sum(1 for s in stats_per_layer.values() if s.get('is_nan'))
         num_inf = sum(1 for s in stats_per_layer.values() if s.get('is_inf'))

         # Filter for valid numerical stats (has grad, not NaN/Inf, and stat exists)
         valid_stats = [s for s in stats_per_layer.values() if s.get('has_grad') and not s.get('is_nan') and not s.get('is_inf')]

         # Helper to safely extract and average a list of numbers
         def safe_average(data_list: List[Optional[float]]) -> Optional[float]:
             valid_nums = [x for x in data_list if x is not None and isinstance(x, (int, float))]
             return sum(valid_nums) / len(valid_nums) if valid_nums else None

         # Extract lists of valid norms, means, stds, ratios (using the norm_type from the last analysis)
         all_norms = [s.get('norm') for s in valid_stats]
         all_means = [s.get('mean') for s in valid_stats]
         all_stds = [s.get('std') for s in valid_stats]
         # Filter ratios specifically for finite values as they can be Inf
         all_ratios = [s.get('grad_param_norm_ratio') for s in valid_stats if s.get('grad_param_norm_ratio') is not None and torch.isfinite(torch.tensor(s['grad_param_norm_ratio']))]


         # Find layer with max *valid* norm
         layer_with_max_norm = None
         max_norm_val = -1.0
         if valid_stats:
             # Iterate through the original dictionary to link stats back to names
             for name, stats in stats_per_layer.items():
                 # Check if this param had a valid norm calculated
                 current_norm = stats.get('norm')
                 if stats in valid_stats and current_norm is not None and current_norm > max_norm_val:
                      max_norm_val = current_norm
                      layer_with_max_norm = name


         # --- Build the Summary Dictionary ---
         summary = {
             "num_params_analyzed": num_analyzed,
             "num_params_with_grads": num_with_grads,
             "num_params_nan_grad": num_nan,
             "num_params_inf_grad": num_inf,
             "global_grad_norm_L2": global_grad_norm_l2, # Include the calculated L2 global norm
             # Aggregate stats (average, min, max) based on valid per-parameter norms
             "avg_grad_norm": safe_average(all_norms),
             "max_grad_norm": max(all_norms) if all_norms else None,
             "min_grad_norm": min(all_norms) if all_norms else None,
             "avg_grad_mean": safe_average(all_means),
             "avg_grad_std": safe_average(all_stds),
             "avg_grad_param_norm_ratio": safe_average(all_ratios),
             "layer_with_max_grad_norm": layer_with_max_norm,
             "error": None # Initialize error as None
         }

         # Add a warning if no parameters had gradients
         if num_with_grads == 0 and not summary.get("error"):
             summary["warning"] = "No parameters requiring gradients had a .grad attribute. Ensure model.backward() was called."

         logger.info("Gradient summary generation complete.")
         return summary

    def plot_gradient_norm_histogram(
        self,
        num_bins: int = 50,
        log_scale_norm: bool = True, # X-axis log scale
        log_scale_counts: bool = True, # Y-axis log scale
        title: str = "Histogram of Parameter Gradient Norms (L2)",
        save_path: Optional[str] = None,
        show_plot: bool = True,
        force_recompute_stats: bool = False # Option to force re-analysis
    ) -> bool:
        """
        Generates a histogram of the L2 norms of gradients for each parameter.

        Requires matplotlib and numpy (`pip install trainsense[plotting]`).
        Uses the results from the last call to `analyze_gradients()` unless forced.

        Args:
            num_bins (int): Number of bins for the histogram. Defaults to 50.
            log_scale_norm (bool): Use logarithmic scale for the gradient norm axis (X-axis).
                                   Often useful as norms can span orders of magnitude. Defaults to True.
            log_scale_counts (bool): Use logarithmic scale for the parameter count axis (Y-axis).
                                     Useful if counts vary widely between bins. Defaults to True.
            title (str): Title for the plot. Defaults to "Histogram of Parameter Gradient Norms (L2)".
            save_path (Optional[str]): Path to save the plot image (e.g., 'logs/grad_hist.png').
                                       If None, plot is not saved. Defaults to None.
            show_plot (bool): Whether to display the plot interactively using `plt.show()`.
                              If False, the plot is generated but not shown (useful for saving only). Defaults to True.
            force_recompute_stats (bool): If True, re-runs `analyze_gradients(norm_type=2.0)`
                                          before plotting. Defaults to False.

        Returns:
            bool: True if the plot was generated successfully, False otherwise (e.g., matplotlib
                  not installed, no valid gradients found, or plotting error).
        """
        # Check if plotting libraries are available
        if not MATPLOTLIB_AVAILABLE or plt is None or np is None:
            logger.error("Cannot plot: matplotlib/numpy not available. Install with 'pip install trainsense[plotting]'")
            return False

        # --- Get Gradient Stats ---
        # Use cached stats unless forced or cache is empty
        if self._last_grad_stats is None or force_recompute_stats:
            logger.info("Plotting requires gradient stats. Running analyze_gradients (L2 norm)...")
            grad_stats = self.analyze_gradients(norm_type=2.0) # Use L2 norm for this plot
        else:
            logger.debug("Using cached gradient statistics for plotting.")
            grad_stats = self._last_grad_stats

        # Check if stats were obtained
        if not grad_stats:
            logger.warning("No gradient statistics available to plot histogram.")
            return False

        # --- Extract Valid Norms for Plotting ---
        # Filter norms: must exist, not be NaN/Inf. For log scale, must be > 0.
        min_val_for_log = 1e-12 # Smallest value allowed for log scale
        norms_list = [
            s['norm'] for s in grad_stats.values()
            if s.get('has_grad') and \
               isinstance(s.get('norm'), (float, int)) and \
               not (s.get('is_nan') or s.get('is_inf')) and \
               (not log_scale_norm or s['norm'] > min_val_for_log) # Ensure positivity only if log scale is requested
        ]

        # Check if any valid norms remain after filtering
        if not norms_list:
            warn_msg = "No valid gradient norms found to plot histogram"
            if log_scale_norm: warn_msg += f" (ensure norms are > {min_val_for_log} if using log scale)."
            logger.warning(warn_msg)
            return False

        norms = np.array(norms_list)
        num_valid_norms = len(norms)
        logger.info(f"Plotting histogram for {num_valid_norms} valid gradient norms.")

        # --- Plotting Logic ---
        try:
            fig, ax = plt.subplots(figsize=(12, 7)) # Slightly larger figure

            # Determine bins dynamically
            use_log_x = log_scale_norm
            min_norm_val = norms.min()
            max_norm_val = norms.max()

            if use_log_x:
                # Check again if min is too low even after filtering (shouldn't happen with filter above, but defensive)
                if min_norm_val <= min_val_for_log:
                     use_log_x = False
                     logger.warning(f"Min valid norm ({min_norm_val:.2e}) is too close to zero; cannot use log scale for X-axis. Using linear.")
                     bins = np.linspace(min_norm_val, max_norm_val, num_bins + 1)
                else:
                     # Create logarithmically spaced bins
                     min_log = np.log10(min_norm_val)
                     max_log = np.log10(max_norm_val)
                     # Handle potential case where min == max after filtering/log
                     if np.isclose(min_log, max_log):
                         bins = np.linspace(min_norm_val * 0.9, max_norm_val * 1.1, num_bins + 1) # Linear bins around the value
                         use_log_x = False # Switch back to linear for X axis
                         logger.warning("Filtered gradient norms have very low variance; using linear scale for X-axis.")
                     else:
                         bins = np.logspace(min_log, max_log, num_bins + 1)
            else: # Explicitly linear scale for X-axis
                # Ensure bins cover the range, handle min==max case gracefully
                if np.isclose(min_norm_val, max_norm_val):
                     # Create bins centered around the value
                     center = min_norm_val
                     width = abs(center * 0.1) + 1e-9 # Small width around the value
                     bins = np.linspace(center - width, center + width, num_bins + 1)
                else:
                     bins = np.linspace(min_norm_val, max_norm_val, num_bins + 1)


            # --- Create Histogram ---
            counts, bin_edges, patches = ax.hist(
                norms,
                bins=bins,
                log=log_scale_counts, # Apply log scale to Y-axis if requested
                color='steelblue',    # Choose a color
                alpha=0.8,           # Transparency
                edgecolor='black',   # Edge color for bars
                linewidth=0.5
            )

            # --- Configure Axes and Labels ---
            ax.set_ylabel(f"Parameter Count {'(Log Scale)' if log_scale_counts else ''}")
            ax.set_xlabel(f"Gradient Norm (L2) {'(Log Scale)' if use_log_x else ''}")
            if use_log_x:
                ax.set_xscale('log') # Apply log scale to x-axis if determined above

            ax.set_title(title, fontsize=14, pad=15)
            ax.grid(axis='y', linestyle='--', alpha=0.7, which='major') # Grid lines for y-axis
            ax.grid(axis='x', linestyle=':', alpha=0.5, which='both') # Grid lines for x-axis (major and minor if log)

            # Improve tick formatting for log scale if used
            if use_log_x:
                ax.xaxis.set_major_locator(plt.LogLocator(base=10, numticks=10)) # Sensible major ticks
                # ax.xaxis.set_minor_locator(plt.LogLocator(base=10, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)) # Optional minor ticks
                ax.xaxis.set_major_formatter(mtick.LogFormatterSciNotation(base=10)) # Scientific notation for major ticks
            else:
                ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e')) # Scientific notation for linear scale

            # Add vertical lines for mean/median? (Optional)
            mean_norm = np.mean(norms)
            median_norm = np.median(norms)
            ax.axvline(mean_norm, color='red', linestyle='dashed', linewidth=1, label=f'Mean ({mean_norm:.2e})')
            ax.axvline(median_norm, color='green', linestyle='dotted', linewidth=1, label=f'Median ({median_norm:.2e})')
            ax.legend(fontsize=9)


            # --- Add Summary Statistics Text Box ---
            # Include stats about excluded parameters if any
            total_params_with_grads = sum(1 for s in grad_stats.values() if s.get('has_grad'))
            num_excluded = total_params_with_grads - num_valid_norms
            excluded_text = f"\n(Excluded: {num_excluded} NaN/Inf/Zero)" if num_excluded > 0 else ""

            stats_text = (
                f"Total Params w/ Grad: {total_params_with_grads}\n"
                f"Params Plotted: {num_valid_norms}{excluded_text}\n"
                f"Min Norm: {min_norm_val:.2e}\n"
                f"Max Norm: {max_norm_val:.2e}\n"
                f"Mean Norm: {mean_norm:.2e}\n"
                f"Median Norm: {median_norm:.2e}"
            )
            # Add standard deviation of log10 norms if log scale is used (often informative)
            if use_log_x and num_valid_norms > 1:
                try:
                     log10_norms = np.log10(norms)
                     # Avoid calculating std dev if variance is effectively zero
                     if not np.allclose(log10_norms, log10_norms[0]):
                         log10_std = np.std(log10_norms)
                         stats_text += f"\nStd Dev (log10): {log10_std:.2f}"
                except Exception as std_err:
                     logger.warning(f"Could not calculate log10 std dev: {std_err}")


            props = dict(boxstyle='round', facecolor='aliceblue', alpha=0.8, edgecolor='grey')
            # Place text box outside plot area for clarity if possible, otherwise inside
            ax.text(1.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='left', bbox=props)

            plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout slightly to make space for text box if placed outside

            # --- Save and Show ---
            if save_path:
                try:
                     # Ensure directory exists before saving
                     save_dir = os.path.dirname(save_path)
                     if save_dir: os.makedirs(save_dir, exist_ok=True) # Create directories if they don't exist
                     logger.info(f"Saving gradient norm histogram to: {save_path}")
                     plt.savefig(save_path, dpi=300, bbox_inches='tight')
                except Exception as save_err:
                     logger.error(f"Failed to save plot to {save_path}: {save_err}", exc_info=True)

            if show_plot:
                plt.show() # Display the plot window
            else:
                plt.close(fig) # Close the figure explicitly if not shown interactively

            return True # Plot generation successful

        except Exception as e:
            logger.error(f"Failed to generate gradient histogram plot: {e}", exc_info=True)
            # Ensure figure is closed if it was created before an error occurred
            if 'fig' in locals() and plt: plt.close(fig)
            return False # Plot generation failed