# TrainSense/visualizer.py
import logging
from typing import Dict, Any, Optional, List # Added List
import math # For histogram binning potentially

# Optional dependency
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import numpy as np # Often useful with matplotlib
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    mtick = None
    np = None
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)

# --- plot_training_step_breakdown function remains the same ---
def plot_training_step_breakdown(
    profile_results: Dict[str, Any],
    title: str = "Average Training Step Time Breakdown",
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> bool:
    """
    Generates a bar chart visualizing the time breakdown of a training step.
    Requires matplotlib (`pip install trainsense[plotting]`).
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("Matplotlib not available. Install with 'pip install trainsense[plotting]'")
        return False
    # Ensure plt and mtick are usable
    if plt is None or mtick is None: return False

    if profile_results.get("profiling_type") != "training_step":
        logger.error("Invalid profile results. Expected 'training_step' type.")
        return False

    # Include detailed data loading breakdown
    phases = {
        'Data Fetch': profile_results.get('percent_time_data_fetch', 0),
        'Data Prep': profile_results.get('percent_time_data_prep', 0),
        'Forward': profile_results.get('percent_time_forward', 0),
        'Loss': profile_results.get('percent_time_loss', 0),
        'Backward': profile_results.get('percent_time_backward', 0),
        'Optimizer': profile_results.get('percent_time_optimizer', 0),
    }

    # Filter phases with negligible time
    phases_filtered = {k: v for k, v in phases.items() if v > 0.1}

    if not phases_filtered:
        logger.warning("No significant time breakdown found. Cannot generate plot.")
        return False

    labels = list(phases_filtered.keys())
    percentages = list(phases_filtered.values())
    total_perc = sum(percentages)

    logger.info(f"Plotting training breakdown. Total percentage accounted for: {total_perc:.1f}%")
    if abs(total_perc - 100.0) > 5.0:
        logger.warning(f"Sum of percentages ({total_perc:.1f}%) is far from 100%. Check profiler timing.")

    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_colors = plt.cm.viridis(np.array(percentages) / 100.0) if np else 'skyblue' # Use numpy for color mapping if available
        bars = ax.bar(labels, percentages, color=bar_colors)

        ax.set_ylabel("Percentage of Step Time (%)")
        ax.set_title(title + f"\n(Avg Step Time: {profile_results.get('avg_step_time_ms', 0):.2f} ms)") # Add avg step time
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100.0))
        ax.set_ylim(0, max(100, max(percentages) * 1.1)) # Ensure y-axis goes to at least 100% or slightly above max
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=9)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path: logger.info(f"Saving plot to: {save_path}"); plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_plot: plt.show()
        else: plt.close(fig)
        return True
    except Exception as e:
        logger.error(f"Failed to generate training breakdown plot: {e}", exc_info=True)
        return False


# --- NEW Gradient Histogram Plot ---
def plot_gradient_histogram(
    grad_stats: Dict[str, Dict[str, Any]], # Takes the detailed stats dict
    num_bins: int = 50,
    log_scale_norm: bool = True, # Log scale for norm axis (often spans orders of magnitude)
    log_scale_counts: bool = True, # Log scale for counts
    title: str = "Histogram of Parameter Gradient Norms (L2)",
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> bool:
    """
    Generates a histogram of the L2 norms of gradients for each parameter.

    Requires matplotlib (`pip install trainsense[plotting]`).

    Args:
        grad_stats (Dict): The detailed statistics dictionary returned by
                           `GradientAnalyzer.analyze_gradients()`.
        num_bins (int): Number of bins for the histogram.
        log_scale_norm (bool): Use logarithmic scale for the gradient norm axis (X-axis).
        log_scale_counts (bool): Use logarithmic scale for the parameter count axis (Y-axis).
        title (str): Title for the plot.
        save_path (Optional[str]): Path to save the plot image.
        show_plot (bool): Whether to display the plot interactively.

    Returns:
        bool: True if the plot was generated successfully, False otherwise.
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("Matplotlib not available. Install with 'pip install trainsense[plotting]'")
        return False
    # Ensure plt and np are usable
    if plt is None or np is None: return False

    if not grad_stats:
        logger.warning("No gradient statistics provided to plot.")
        return False

    # Extract valid norms (exclude None, NaN, Inf)
    norms = np.array([s['norm'] for s in grad_stats.values() if s.get('norm') is not None and not (s.get('is_nan') or s.get('is_inf')) and s['norm'] > 0]) # Filter 0 norms for log scale

    if norms.size == 0:
        logger.warning("No valid positive gradient norms found to plot histogram.")
        return False

    logger.info(f"Plotting histogram for {len(norms)} gradient norms.")

    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Determine bins, potentially logarithmic if log_scale_norm is True
        if log_scale_norm:
            min_log_norm = np.log10(norms.min())
            max_log_norm = np.log10(norms.max())
             # Handle case where min and max are very close or equal
            if np.isclose(min_log_norm, max_log_norm):
                # Create linear bins around the single value if log scale makes no sense
                 bins = np.linspace(norms.min() * 0.9, norms.max() * 1.1, num_bins + 1)
                 log_scale_norm = False # Switch back to linear for X axis
                 logger.warning("Gradient norms have very low variance; using linear scale for X-axis.")
            else:
                 bins = np.logspace(min_log_norm, max_log_norm, num_bins + 1)
        else:
            bins = np.linspace(norms.min(), norms.max(), num_bins + 1)

        # Plot histogram
        counts, bin_edges, patches = ax.hist(norms, bins=bins, log=log_scale_counts, color='teal', alpha=0.8, edgecolor='black')

        ax.set_ylabel(f"Parameter Count {'(Log Scale)' if log_scale_counts else ''}")
        ax.set_xlabel(f"Gradient Norm (L2) {'(Log Scale)' if log_scale_norm else ''}")
        if log_scale_norm:
            ax.set_xscale('log')

        ax.set_title(title)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.grid(axis='x', linestyle=':', alpha=0.5) # Add x grid too


        # Add summary stats text box
        stats_text = (
            f"Num Params: {len(norms)}\n"
            f"Mean Norm: {np.mean(norms):.2e}\n"
            f"Median Norm: {np.median(norms):.2e}\n"
            f"Min Norm: {norms.min():.2e}\n"
            f"Max Norm: {norms.max():.2e}\n"
            f"Std Dev (log10): {np.std(np.log10(norms)):.2f}" if log_scale_norm and len(norms)>1 else "" # Std dev of log is often informative
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.6)
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right', bbox=props)


        plt.tight_layout()

        if save_path: logger.info(f"Saving plot to: {save_path}"); plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_plot: plt.show()
        else: plt.close(fig)
        return True

    except Exception as e:
        logger.error(f"Failed to generate gradient histogram plot: {e}", exc_info=True)
        return False