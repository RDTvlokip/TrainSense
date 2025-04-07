# TrainSense/visualizer.py
import logging
from typing import Dict, Any, Optional

# Optional dependency
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    mtick = None
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)

def plot_training_step_breakdown(
    profile_results: Dict[str, Any],
    title: str = "Average Training Step Time Breakdown",
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> bool:
    """
    Generates a bar chart visualizing the time breakdown of a training step.

    Requires matplotlib to be installed (`pip install matplotlib`).

    Args:
        profile_results (Dict[str, Any]): The results dictionary returned by
                                           ModelProfiler.profile_training_step.
                                           Must contain percentage keys like
                                           'percent_time_data_loading', etc.
        title (str): The title for the plot.
        save_path (Optional[str]): If provided, saves the plot to this file path.
        show_plot (bool): If True, displays the plot interactively.

    Returns:
        bool: True if the plot was generated successfully, False otherwise.
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("Matplotlib is not installed. Cannot generate plot. "
                     "Install with 'pip install matplotlib'")
        return False

    if profile_results.get("profiling_type") != "training_step":
        logger.error("Invalid profile results provided. Expected results from 'profile_training_step'.")
        return False

    phases = {
        'Data Fetch': profile_results.get('percent_time_data_fetch', 0),
        'Data Prep': profile_results.get('percent_time_data_prep', 0),
        'Forward': profile_results.get('percent_time_forward', 0),
        'Loss': profile_results.get('percent_time_loss', 0),
        'Backward': profile_results.get('percent_time_backward', 0),
        'Optimizer': profile_results.get('percent_time_optimizer', 0),
    }

    # Filter out phases with zero time if desired, or keep them
    phases_filtered = {k: v for k, v in phases.items() if v > 0.1} # Threshold to avoid tiny bars

    if not phases_filtered:
        logger.warning("No significant time breakdown found in profile results. Cannot generate plot.")
        return False

    labels = list(phases_filtered.keys())
    percentages = list(phases_filtered.values())

    # Ensure total doesn't wildly exceed 100% due to potential overlaps/measurement nuances
    total_perc = sum(percentages)
    if abs(total_perc - 100.0) > 5.0: # Allow some tolerance
        logger.warning(f"Sum of time percentages ({total_perc:.1f}%) is far from 100%. Plot may be misleading.")
        # Optionally normalize percentages here if desired
        # percentages = [p / total_perc * 100 for p in percentages]


    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(labels, percentages, color=plt.cm.viridis( [p/100 for p in percentages] )) # Color by percentage

        ax.set_ylabel("Percentage of Step Time (%)")
        ax.set_title(title)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100.0))
        ax.set_ylim(0, max(percentages) * 1.1) # Dynamic ylim based on max value

        # Add percentage labels on top of bars
        ax.bar_label(bars, fmt='%.1f%%', padding=3)

        plt.xticks(rotation=45, ha='right') # Rotate labels for better readability
        plt.tight_layout() # Adjust layout

        if save_path:
            logger.info(f"Saving training step breakdown plot to: {save_path}")
            plt.savefig(save_path, dpi=300)

        if show_plot:
            plt.show()
        else:
             plt.close(fig) # Close the figure if not showing interactively

        return True

    except Exception as e:
        logger.error(f"Failed to generate plot: {e}", exc_info=True)
        return False