# TrainSense/gradient_analyzer.py
import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class GradientAnalyzer:
    """
    Analyzes gradient statistics for model parameters after a backward pass.

    Requires the model to have had .backward() called recently so that
    parameter .grad attributes are populated.
    """
    def __init__(self, model: nn.Module):
        if not isinstance(model, nn.Module):
            raise TypeError("Input 'model' must be an instance of torch.nn.Module.")
        self.model = model
        logger.info(f"GradientAnalyzer initialized for model type: {type(model).__name__}")

    @torch.no_grad() # Ensure no gradient calculation during analysis
    def analyze_gradients(self, norm_type: float = 2.0) -> Dict[str, Dict[str, Any]]:
        """
        Calculates statistics for gradients of trainable parameters.

        Args:
            norm_type (float): The type of norm to compute (e.g., 2.0 for L2 norm). Defaults to 2.0.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary where keys are parameter names
                                      and values are dictionaries containing gradient statistics
                                      (norm, mean, std, min, max, is_nan, is_inf).
                                      Returns an empty dict if no gradients are found.
        """
        logger.info(f"Starting gradient analysis (norm_type={norm_type})...")
        grad_stats = {}
        found_grads = False

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if param.grad is None:
                continue

            found_grads = True
            grad_data = param.grad.detach().float() # Use float for stable stats

            # Check for NaN/Inf
            has_nan = torch.isnan(grad_data).any().item()
            has_inf = torch.isinf(grad_data).any().item()

            if has_nan or has_inf:
                logger.warning(f"Gradient for '{name}' contains NaN ({has_nan}) or Inf ({has_inf}). Stats might be unreliable.")
                stats = {
                    # Use float('nan') for norm if NaN is present, else float('inf') if Inf is present
                    "norm": float('nan') if has_nan else float('inf'),
                    "mean": float('nan'),
                    "std": float('nan'),
                    "min": float('nan'),
                    "max": float('nan'),
                    "abs_mean": float('nan'),
                    "is_nan": has_nan,
                    "is_inf": has_inf,
                    "shape": tuple(param.grad.shape),
                    "grad_param_norm_ratio": float('nan') # Ratio is undefined with NaN/Inf grads
                }
            else:
                # Calculate statistics only if grads are finite
                 # --- CORRECTED NORM CALCULATION ---
                 grad_norm = torch.linalg.norm(grad_data.flatten(), ord=norm_type).item()
                 # ---------------------------------
                 stats = {
                    "norm": grad_norm,
                    "mean": grad_data.mean().item(),
                    "std": grad_data.std().item(),
                    "min": grad_data.min().item(),
                    "max": grad_data.max().item(),
                    "abs_mean": grad_data.abs().mean().item(), # Mean of absolute values
                    "is_nan": False,
                    "is_inf": False,
                    "shape": tuple(param.grad.shape),
                }
                 # Calculate grad/param norm ratio if param norm > 0
                 # --- CORRECTED PARAM NORM ---
                 param_norm = torch.linalg.norm(param.data.detach().float().flatten(), ord=norm_type).item()
                 # --------------------------
                 if param_norm > 1e-8: # Avoid division by zero
                     stats["grad_param_norm_ratio"] = stats["norm"] / param_norm
                 else:
                     stats["grad_param_norm_ratio"] = None


            grad_stats[name] = stats
            # logger.debug(f"Grad stats for '{name}': {stats}") # Peut être verbeux

        if not found_grads:
            logger.warning("No gradients found in any trainable parameters. Did you run .backward()?")

        logger.info(f"Gradient analysis complete. Analyzed {len(grad_stats)} parameters with gradients.")
        return grad_stats

    def summary(self, norm_type: float = 2.0) -> Dict[str, Any]:
         """Provides a summary of gradient statistics across all layers."""
         stats_per_layer = self.analyze_gradients(norm_type)
         if not stats_per_layer:
             return {"error": "No gradients found to summarize."}

         # Filter out NaN/Inf before calculating summary stats where applicable
         valid_stats = [s for s in stats_per_layer.values() if not (s.get('is_nan') or s.get('is_inf'))]

         all_norms = [s['norm'] for s in valid_stats if s.get('norm') is not None]
         all_means = [s['mean'] for s in valid_stats if s.get('mean') is not None]
         all_stds = [s['std'] for s in valid_stats if s.get('std') is not None]
         all_ratios = [s['grad_param_norm_ratio'] for s in valid_stats if s.get('grad_param_norm_ratio') is not None]
         num_nan = sum(1 for s in stats_per_layer.values() if s.get('is_nan'))
         num_inf = sum(1 for s in stats_per_layer.values() if s.get('is_inf'))

         # Calculate global norm carefully, excluding params without gradients
         param_grads = [p.grad.detach().flatten() for p in self.model.parameters() if p.requires_grad and p.grad is not None]
         global_grad_norm_val = 0.0
         if param_grads:
            try:
                 # Use norm directly on stacked norms is faster than norm on concatenated grads for large models
                 # global_grad_norm_val = torch.linalg.norm(torch.cat(param_grads), ord=norm_type).item() # Alternative but memory intensive
                 individual_norms = torch.stack([torch.linalg.norm(g, ord=norm_type) for g in param_grads])
                 # For L2 norm, global norm is the L2 norm of individual norms
                 if norm_type == 2.0:
                     global_grad_norm_val = torch.linalg.norm(individual_norms, ord=2.0).item()
                 else: # For other norms, it's usually just the sum or max, L2 is most common "global norm"
                      # Provide L2 anyway as it's standard, maybe add others if needed
                      if norm_type != 2.0:
                           logger.warning(f"Calculating standard L2 global gradient norm, even though requested norm was {norm_type}.")
                           l2_norms = torch.stack([torch.linalg.norm(g, ord=2.0) for g in param_grads])
                           global_grad_norm_val = torch.linalg.norm(l2_norms, ord=2.0).item()

            except RuntimeError as e:
                 logger.error(f"Could not compute global gradient norm: {e}. Possibly due to NaN/Inf values not filtered earlier in raw grads.")
                 global_grad_norm_val = float('nan')


         summary = {
             "num_params_with_grads": len(stats_per_layer),
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
         # Add layer with max norm for easier debugging, ensuring it's from valid stats
         if all_norms:
             # Find max norm among valid stats only
             max_norm_item = max(
                 ((name, stats) for name, stats in stats_per_layer.items() if not (stats.get('is_nan') or stats.get('is_inf'))),
                 key=lambda item: item[1]['norm'],
                 default=(None, None)
             )
             if max_norm_item[0] is not None:
                 summary["layer_with_max_grad_norm"] = max_norm_item[0]


         return summary