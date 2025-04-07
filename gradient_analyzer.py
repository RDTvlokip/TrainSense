# TrainSense/gradient_analyzer.py
# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, List, Tuple

# Use try-except for optional import of visualization components
try:
    # Import matplotlib and numpy if available
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    mtick = None
    np = None
    # Log only once at import time if matplotlib is missing
    logging.getLogger(__name__).info("Matplotlib or NumPy not found, plotting disabled for GradientAnalyzer.")


logger = logging.getLogger(__name__)

class GradientAnalyzer:
    """
    Analyzes gradient statistics for model parameters after a backward pass.
    Includes optional histogram plotting of gradient norms.
    """
    def __init__(self, model: nn.Module):
        """
        Initializes the GradientAnalyzer.

        Args:
            model (nn.Module): The PyTorch model whose gradients will be analyzed.
        """
        if not isinstance(model, nn.Module):
            raise TypeError("Input 'model' must be an instance of torch.nn.Module.")
        self.model = model
        logger.info(f"GradientAnalyzer initialized for model type: {type(model).__name__}")
        self._last_grad_stats: Optional[Dict[str, Dict[str, Any]]] = None # Cache last analysis results

    @torch.no_grad() # Ensure no gradient calculation during analysis
    def analyze_gradients(self, norm_type: float = 2.0) -> Dict[str, Dict[str, Any]]:
        """
        Calculates statistics for gradients of trainable parameters.
        Requires model.backward() to have been called recently.

        Args:
            norm_type (float): The type of norm to compute (e.g., 2.0 for L2 norm). Defaults to 2.0.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary where keys are parameter names
                                      and values are dictionaries containing gradient statistics
                                      (norm, mean, std, min, max, is_nan, is_inf, shape, grad_param_norm_ratio).
                                      Returns an empty dict if no gradients are found.
        """
        logger.info(f"Starting gradient analysis (norm_type={norm_type})...")
        grad_stats: Dict[str, Dict[str, Any]] = {}
        found_grads = False

        for name, param in self.model.named_parameters():
            # Skip parameters that don't require gradients or have no gradient yet
            if not param.requires_grad or param.grad is None:
                continue

            found_grads = True
            # Detach and convert to float for stability, work on the correct device
            grad_data = param.grad.detach().float()

            # Check for NaN/Inf first
            has_nan = torch.isnan(grad_data).any().item()
            has_inf = torch.isinf(grad_data).any().item()

            # Initialize stats dictionary for this parameter
            stats: Dict[str, Any] = {
                "shape": tuple(param.grad.shape),
                "is_nan": has_nan,
                "is_inf": has_inf,
                "norm": None, "mean": None, "std": None, "min": None, "max": None,
                "abs_mean": None, "grad_param_norm_ratio": None
            }

            if has_nan or has_inf:
                # Log warning and set relevant stats to NaN/Inf
                logger.warning(f"Gradient for '{name}' contains NaN ({has_nan}) or Inf ({has_inf}). Stats will reflect this.")
                stats["norm"] = float('nan') if has_nan else float('inf')
                stats["mean"] = float('nan')
                stats["std"] = float('nan')
                stats["min"] = float('nan')
                stats["max"] = float('nan')
                stats["abs_mean"] = float('nan')
                stats["grad_param_norm_ratio"] = float('nan') # Ratio is undefined
            else:
                 # Calculate statistics only if gradients are finite and non-zero (for norm)
                 try:
                    grad_flat = grad_data.flatten() # Flatten for norm calculation
                    # Use specified norm type
                    grad_norm = torch.linalg.norm(grad_flat, ord=norm_type).item()
                    stats["norm"] = grad_norm

                    # Calculate other statistics on the original shape tensor
                    stats["mean"] = grad_data.mean().item()
                    stats["std"] = grad_data.std().item()
                    stats["min"] = grad_data.min().item()
                    stats["max"] = grad_data.max().item()
                    stats["abs_mean"] = grad_data.abs().mean().item()

                    # Calculate grad/param norm ratio
                    param_flat = param.data.detach().float().flatten()
                    param_norm = torch.linalg.norm(param_flat, ord=norm_type).item()
                    if param_norm > 1e-9: # Use a slightly larger epsilon for float comparison
                        stats["grad_param_norm_ratio"] = stats["norm"] / param_norm
                    else:
                        # Handle case where parameter norm is zero or near-zero
                        stats["grad_param_norm_ratio"] = float('inf') if stats["norm"] > 1e-9 else 0.0
                        logger.debug(f"Parameter '{name}' norm is close to zero ({param_norm:.2e}). Ratio set accordingly.")

                 except Exception as stat_err:
                     logger.error(f"Error calculating stats for finite gradient '{name}': {stat_err}", exc_info=True)
                     # Set stats to NaN on error during calculation
                     stats["norm"] = stats["mean"] = stats["std"] = stats["min"] = stats["max"] = stats["abs_mean"] = stats["grad_param_norm_ratio"] = float('nan')

            grad_stats[name] = stats

        if not found_grads:
            logger.warning("No gradients found in any trainable parameters. Ensure model.backward() was called.")

        self._last_grad_stats = grad_stats # Cache the results for potential reuse (e.g., plotting)
        logger.info(f"Gradient analysis complete. Analyzed {len(grad_stats)} parameters with gradients.")
        return grad_stats

    def summary(self, norm_type: float = 2.0) -> Dict[str, Any]:
         """
         Provides a summary of gradient statistics across all layers based on the last analysis.
         Re-runs analysis if no cached data is found.

         Args:
             norm_type (float): The norm type used ONLY if analysis needs to be re-run.
                                The global norm reported is typically L2. Defaults to 2.0.

         Returns:
             Dict[str, Any]: A dictionary summarizing gradient statistics across the model.
         """
         # Use cached stats if available, otherwise run analysis
         stats_per_layer = self._last_grad_stats if self._last_grad_stats is not None else self.analyze_gradients(norm_type)

         if not stats_per_layer:
             return {"error": "No gradients found to summarize."}

         # Filter out parameters with NaN/Inf gradients for summary statistics calculation
         valid_stats = [s for s in stats_per_layer.values() if not (s.get('is_nan') or s.get('is_inf'))]

         # Extract valid numerical stats from the filtered list
         all_norms = [s['norm'] for s in valid_stats if isinstance(s.get('norm'), (int, float))]
         all_means = [s['mean'] for s in valid_stats if isinstance(s.get('mean'), (int, float))]
         all_stds = [s['std'] for s in valid_stats if isinstance(s.get('std'), (int, float))]
         all_ratios = [s['grad_param_norm_ratio'] for s in valid_stats if isinstance(s.get('grad_param_norm_ratio'), (int, float))]

         # Count NaN/Inf occurrences across all analyzed parameters
         num_nan = sum(1 for s in stats_per_layer.values() if s.get('is_nan'))
         num_inf = sum(1 for s in stats_per_layer.values() if s.get('is_inf'))

         # --- CORRECTED GLOBAL NORM CALCULATION ---
         global_grad_norm_val = 0.0
         # Generator for parameters that require grad and have a non-None grad
         parameters_with_grads = (p for p in self.model.parameters() if p.requires_grad and p.grad is not None)

         if True: # Always try, handle errors inside
             try:
                 # Use the utility function clip_grad_norm_ which *returns* the total norm
                 # Pass float('inf') as max_norm to prevent actual clipping.
                 # Standard L2 norm is calculated by default (norm_type=2.0).
                 global_grad_norm_val = torch.nn.utils.clip_grad_norm_(
                     parameters=parameters_with_grads,
                     max_norm=float('inf'), # Don't actually clip, just compute the norm
                     norm_type=2.0 # Standard L2 norm calculation
                 ).item() # Get the scalar value

                 # Check for NaN/Inf in the final result (can happen if inputs had issues)
                 if not torch.isfinite(torch.tensor(global_grad_norm_val)):
                      logger.warning(f"Computed global gradient norm is not finite ({global_grad_norm_val}). Setting to NaN.")
                      global_grad_norm_val = float('nan')

             except RuntimeError as e:
                 # Catch potential errors during norm calculation (e.g., empty parameter list after filtering?)
                 logger.error(f"RuntimeError computing global gradient norm: {e}. This might happen if no valid gradients were found.", exc_info=True)
                 global_grad_norm_val = float('nan') # Indicate failure
             except Exception as e:
                 logger.error(f"Unexpected error computing global gradient norm: {e}", exc_info=True)
                 global_grad_norm_val = float('nan')
         # ---------------------------------------

         # Build the summary dictionary
         summary = {
             "num_params_with_grads": len(stats_per_layer),
             "num_params_valid_grads": len(valid_stats), # Count params without NaN/Inf
             "num_params_nan_grad": num_nan,
             "num_params_inf_grad": num_inf,
             "global_grad_norm_L2": global_grad_norm_val, # Report the calculated L2 global norm
             "avg_grad_norm": sum(all_norms) / len(all_norms) if all_norms else 0.0,
             "max_grad_norm": max(all_norms) if all_norms else 0.0,
             "min_grad_norm": min(all_norms) if all_norms else 0.0,
             "avg_grad_mean": sum(all_means) / len(all_means) if all_means else 0.0,
             "avg_grad_std": sum(all_stds) / len(all_stds) if all_stds else 0.0,
             "avg_grad_param_norm_ratio": sum(all_ratios) / len(all_ratios) if all_ratios else None,
             "layer_with_max_grad_norm": None # Initialize
         }

         # Find layer with max *valid* norm
         if all_norms:
             # Find the parameter name corresponding to the maximum valid norm
             max_norm_val = -1.0
             max_norm_layer_name = None
             # Iterate through the original dictionary to get names associated with valid stats
             for name, stats in stats_per_layer.items():
                 # Check if this stat is valid and has a norm
                 if not (stats.get('is_nan') or stats.get('is_inf')) and isinstance(stats.get('norm'), (float, int)):
                     if stats['norm'] > max_norm_val:
                         max_norm_val = stats['norm']
                         max_norm_layer_name = name
             summary["layer_with_max_grad_norm"] = max_norm_layer_name

         return summary

    def plot_gradient_norm_histogram(
        self,
        num_bins: int = 50,
        log_scale_norm: bool = True,
        log_scale_counts: bool = True,
        title: str = "Histogram of Parameter Gradient Norms (L2)",
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> bool:
        """
        Generates a histogram of the L2 norms of gradients for each parameter.

        Requires matplotlib and numpy (`pip install trainsense[plotting]`).
        Uses the results from the last call to `analyze_gradients()`.

        Args:
            num_bins (int): Number of bins for the histogram.
            log_scale_norm (bool): Use logarithmic scale for the gradient norm axis (X-axis).
            log_scale_counts (bool): Use logarithmic scale for the parameter count axis (Y-axis).
            title (str): Title for the plot.
            save_path (Optional[str]): Path to save the plot image.
            show_plot (bool): Whether to display the plot interactively.

        Returns:
            bool: True if the plot was generated successfully, False otherwise.
        """
        if not MATPLOTLIB_AVAILABLE or plt is None or np is None:
            logger.error("matplotlib/numpy not available for plotting. Install with 'pip install trainsense[plotting]'")
            return False

        # Use cached stats if available, otherwise run analysis (default L2 norm)
        grad_stats = self._last_grad_stats if self._last_grad_stats is not None else self.analyze_gradients()

        if not grad_stats:
            logger.warning("No gradient statistics available to plot.")
            return False

        # Extract valid norms (exclude None, NaN, Inf, and zero/negative for log scale)
        norms_list = [s['norm'] for s in grad_stats.values() if isinstance(s.get('norm'), (float, int)) and not (s.get('is_nan') or s.get('is_inf')) and s['norm'] > 1e-12] # Filter very small/zero for log

        if not norms_list:
            logger.warning("No valid positive gradient norms found to plot histogram.")
            return False

        norms = np.array(norms_list)
        logger.info(f"Plotting histogram for {len(norms)} valid gradient norms.")

        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Determine bins, potentially logarithmic if log_scale_norm is True
            use_log_x = log_scale_norm
            min_norm_val = norms.min()
            max_norm_val = norms.max()

            if use_log_x:
                if min_norm_val <= 1e-12: # Check if min is too low even after filtering
                     use_log_x = False
                     logger.warning("Filtered norms still contain values too close to zero; cannot use log scale for X-axis. Using linear.")
                else:
                     min_log_norm = np.log10(min_norm_val)
                     max_log_norm = np.log10(max_norm_val)
                     # Handle potential case where min == max after filtering
                     if np.isclose(min_log_norm, max_log_norm):
                         bins = np.linspace(min_norm_val * 0.9, max_norm_val * 1.1, num_bins + 1)
                         use_log_x = False # Switch back to linear for X axis
                         logger.warning("Gradient norms have very low variance; using linear scale for X-axis.")
                     else:
                         bins = np.logspace(min_log_norm, max_log_norm, num_bins + 1)

            if not use_log_x: # Fallback or explicitly linear
                # Ensure bins cover the range, handle min==max case
                if np.isclose(min_norm_val, max_norm_val):
                     bins = np.linspace(min_norm_val - abs(min_norm_val*0.1) - 1e-9, max_norm_val + abs(max_norm_val*0.1) + 1e-9, num_bins + 1)
                else:
                     bins = np.linspace(min_norm_val, max_norm_val, num_bins + 1)


            # Plot histogram
            counts, bin_edges, patches = ax.hist(norms, bins=bins, log=log_scale_counts, color='teal', alpha=0.8, edgecolor='black')

            # Setup Axes
            ax.set_ylabel(f"Parameter Count {'(Log Scale)' if log_scale_counts else ''}")
            ax.set_xlabel(f"Gradient Norm (L2) {'(Log Scale)' if use_log_x else ''}")
            if use_log_x:
                ax.set_xscale('log') # Apply log scale to x-axis if decided
            ax.set_title(title)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.grid(axis='x', linestyle=':', alpha=0.5) # Add x grid too

            # Add summary stats text box
            stats_text = (
                f"Num Params: {len(norms)}\n"
                f"Mean Norm: {np.mean(norms):.2e}\n"
                f"Median Norm: {np.median(norms):.2e}\n"
                f"Min Norm: {min_norm_val:.2e}\n" # Use calculated min
                f"Max Norm: {max_norm_val:.2e}" # Use calculated max
            )
            # Calculate std dev of log10 only if log scale is used and makes sense
            if use_log_x and len(norms) > 1:
                try:
                     log10_norms = np.log10(norms)
                     # Avoid calculating std dev if variance is zero
                     if not np.allclose(log10_norms, log10_norms[0]):
                         log10_std = np.std(log10_norms)
                         stats_text += f"\nStd Dev (log10): {log10_std:.2f}"
                except Exception: pass # Ignore errors calculating std dev

            props = dict(boxstyle='round', facecolor='wheat', alpha=0.6)
            # Place text box slightly offset from the top-right corner
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='right', bbox=props)

            plt.tight_layout() # Adjust layout to prevent labels overlapping

            # Save and Show
            if save_path:
                try:
                     # Ensure directory exists before saving
                     os.makedirs(os.path.dirname(save_path), exist_ok=True)
                     logger.info(f"Saving gradient norm histogram to: {save_path}")
                     plt.savefig(save_path, dpi=300, bbox_inches='tight')
                except Exception as save_err:
                     logger.error(f"Failed to save plot to {save_path}: {save_err}", exc_info=True)

            if show_plot:
                plt.show()
            else:
                plt.close(fig) # Close the figure explicitly if not shown

            return True

        except Exception as e:
            logger.error(f"Failed to generate gradient histogram plot: {e}", exc_info=True)
            # Ensure figure is closed if created before error
            if 'fig' in locals() and plt: plt.close(fig)
            return False