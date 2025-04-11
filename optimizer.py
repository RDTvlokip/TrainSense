# TrainSense/optimizer.py
import logging
from typing import Tuple, Optional

# Get a logger instance specific to this module
logger = logging.getLogger(__name__)

class OptimizerHelper:
    """
    Provides static helper methods for suggesting optimizers, learning rate schedulers,
    and performing basic learning rate adjustments based on model characteristics and training state.
    These are heuristics and serve as starting points or simple feedback mechanisms.
    """

    # Thresholds used in optimizer suggestions based on model size
    PARAM_THRESHOLD_LARGE = 50_000_000 # Models above this size often benefit from AdamW
    PARAM_THRESHOLD_SMALL = 5_000_000  # Models below this might work well with SGD or Adam

    @staticmethod
    def suggest_optimizer(model_size_params: int,
                          layer_count: int = 0, # Optional layer count for future refinement
                          architecture_type: str = "Unknown") -> str:
        """
        Suggests a suitable optimizer based on model size and architecture type heuristics.

        Args:
            model_size_params (int): Total number of parameters in the model.
            layer_count (int): Total number of layers (currently unused but available for future logic).
            architecture_type (str): The primary architecture type inferred (e.g., "Transformer", "CNN", "RNN", "MLP").

        Returns:
            str: A string suggesting an optimizer and a brief rationale.
        """
        logger.info(f"Suggesting optimizer for model - Size: {model_size_params:,} params, Arch: {architecture_type}")

        # Normalize architecture type for case-insensitive matching
        arch_type_lower = architecture_type.lower()

        # Heuristic Rules:
        # 1. Transformers or very large models: AdamW is often preferred.
        if "transformer" in arch_type_lower or model_size_params > OptimizerHelper.PARAM_THRESHOLD_LARGE:
            recommendation = "AdamW (Recommended for large models/Transformers due to better weight decay handling)"
            logger.debug(f"Recommendation: {recommendation}")
            return recommendation
        # 2. RNNs (LSTM/GRU): Adam or RMSprop are common choices.
        elif "rnn" in arch_type_lower or "lstm" in arch_type_lower or "gru" in arch_type_lower:
             if model_size_params > OptimizerHelper.PARAM_THRESHOLD_SMALL:
                 recommendation = "Adam (Common for RNNs, AdamW also viable)"
             else:
                 recommendation = "RMSprop or Adam (RMSprop sometimes preferred for RNN stability)"
             logger.debug(f"Recommendation: {recommendation}")
             return recommendation
        # 3. Small/Simple models: SGD with momentum or Adam can work well.
        elif model_size_params < OptimizerHelper.PARAM_THRESHOLD_SMALL and layer_count < 50: # Simple MLP/CNN?
            recommendation = "SGD with Momentum or Adam (Adam is often easier to tune; SGD might generalize slightly better with tuning)"
            logger.debug(f"Recommendation: {recommendation}")
            return recommendation
        # 4. Default / Moderate CNNs/MLPs: Adam is a robust general choice.
        else:
             recommendation = "Adam (Good general default; consider AdamW if weight decay is critical)"
             logger.debug(f"Recommendation: {recommendation}")
             return recommendation

    @staticmethod
    def suggest_learning_rate_scheduler(optimizer_name: str) -> str:
        """
        Suggests a learning rate scheduler based on the chosen optimizer name.

        Args:
            optimizer_name (str): The name of the optimizer being used (e.g., "AdamW", "SGD").

        Returns:
            str: A string suggesting a scheduler type and rationale.
        """
        # Normalize optimizer name
        opt_name_lower = optimizer_name.lower()
        logger.info(f"Suggesting scheduler based on optimizer: {optimizer_name}")

        # Heuristic Rules:
        if "adamw" in opt_name_lower:
            recommendation = "CosineAnnealingLR (smooth decay) or ReduceLROnPlateau (adapts to validation metric) or Linear warmup + decay"
            logger.debug(f"Recommendation: {recommendation}")
            return recommendation
        elif "adam" in opt_name_lower:
            recommendation = "StepLR (simple decay) or ReduceLROnPlateau (adapts to validation metric) or CosineAnnealingLR"
            logger.debug(f"Recommendation: {recommendation}")
            return recommendation
        elif "sgd" in opt_name_lower:
             recommendation = "StepLR/MultiStepLR (common with SGD) or CosineAnnealingLR (smooth decay) or ReduceLROnPlateau"
             logger.debug(f"Recommendation: {recommendation}")
             return recommendation
        # Default fallback
        else:
             recommendation = "ReduceLROnPlateau (General purpose, adapts to validation metric)"
             logger.debug(f"Recommendation: {recommendation}")
             return recommendation

    @staticmethod
    def adjust_learning_rate_on_plateau(current_lr: float,
                                        plateau_epochs: int,
                                        min_lr: float = 1e-6,
                                        factor: float = 0.1,
                                        patience: int = 10) -> Tuple[Optional[float], str]:
        """
        Simulates a basic ReduceLROnPlateau logic check.
        Suggests reducing the learning rate if a plateau condition is met.

        Args:
            current_lr (float): The current learning rate.
            plateau_epochs (int): Number of consecutive epochs the monitored metric hasn't improved.
            min_lr (float): The minimum learning rate threshold. Defaults to 1e-6.
            factor (float): Factor by which to reduce the learning rate (lr * factor). Defaults to 0.1.
            patience (int): Number of epochs to wait for improvement before reducing LR. Defaults to 10.

        Returns:
            Tuple[Optional[float], str]: A tuple containing:
                - The potentially new learning rate (or None if min_lr is reached).
                - A message describing the action taken or status.
        """
        logger.info(f"Checking LR adjustment: Current LR={current_lr:.2e}, Plateau Epochs={plateau_epochs}, Patience={patience}, Factor={factor}, Min LR={min_lr:.2e}")

        # Check if patience has been exceeded
        if plateau_epochs >= patience:
            new_lr = current_lr * factor
            # Check if the new LR would fall below the minimum allowed
            if new_lr < min_lr:
                logger.warning(f"Plateau detected ({plateau_epochs} epochs), but reducing LR would go below min_lr ({min_lr:.2e}). No change made.")
                return None, f"Performance plateaued for {plateau_epochs} epochs. Minimum LR ({min_lr:.2e}) reached. Consider stopping or other changes."
            else:
                # Reduce the learning rate
                logger.info(f"Plateau detected ({plateau_epochs} epochs). Reducing learning rate from {current_lr:.2e} to {new_lr:.2e}.")
                return new_lr, f"Performance plateaued for {plateau_epochs} epochs. Reducing learning rate by factor {factor}."
        else:
            # Patience not yet exceeded, no change needed
            logger.debug(f"Performance stable or improving (Plateau duration {plateau_epochs} < Patience {patience}). Learning rate remains {current_lr:.2e}.")
            return current_lr, "Learning rate stable (patience not met)."

    @staticmethod
    def suggest_initial_learning_rate(architecture_type: str = "Unknown",
                                      model_size_params: int = 0) -> float:
         """
         Suggests a heuristic starting learning rate based on architecture and model size.
         These are rough starting points and often require further tuning.

         Args:
             architecture_type (str): The primary architecture type.
             model_size_params (int): Total number of parameters.

         Returns:
             float: A suggested initial learning rate.
         """
         arch_type_lower = architecture_type.lower()
         logger.info(f"Suggesting initial LR for Arch: {architecture_type}, Size: {model_size_params:,} params")

         # Heuristics based on common practices:
         # Transformers often use smaller LRs, especially large ones.
         if "transformer" in arch_type_lower:
              # Smaller LR for larger transformers
              lr = 1e-5 if model_size_params > 1_000_000_000 else \
                   3e-5 if model_size_params > OptimizerHelper.PARAM_THRESHOLD_LARGE else \
                   1e-4
              logger.debug(f"Suggesting LR {lr:.1e} for Transformer.")
              return lr
         # RNNs might benefit from slightly higher LRs than large transformers but lower than CNNs.
         elif "rnn" in arch_type_lower or "lstm" in arch_type_lower or "gru" in arch_type_lower:
              lr = 1e-3
              logger.debug(f"Suggesting LR {lr:.1e} for RNN.")
              return lr
         # CNNs are often trained with LRs around 1e-3, maybe slightly lower for very large ones.
         elif "cnn" in arch_type_lower:
              lr = 5e-4 if model_size_params > OptimizerHelper.PARAM_THRESHOLD_LARGE else 1e-3
              logger.debug(f"Suggesting LR {lr:.1e} for CNN.")
              return lr
         # Default / MLP: 1e-3 is a common starting point.
         else:
              lr = 1e-3
              logger.debug(f"Suggesting LR {lr:.1e} as default.")
              return lr