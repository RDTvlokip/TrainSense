# TrainSense/optimizer.py
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class OptimizerHelper:
    PARAM_THRESHOLD_LARGE = 50_000_000
    PARAM_THRESHOLD_SMALL = 5_000_000

    @staticmethod
    def suggest_optimizer(model_size_params: int,
                          layer_count: int = 0,
                          architecture_type: str = "Unknown") -> str:

        logger.info(f"Suggesting optimizer for model size={model_size_params:,}, layers={layer_count}, arch={architecture_type}")

        arch_type_lower = architecture_type.lower()

        if "transformer" in arch_type_lower or model_size_params > OptimizerHelper.PARAM_THRESHOLD_LARGE:
            logger.debug("Recommending AdamW due to large size or Transformer architecture.")
            return "AdamW (Recommended for large models/Transformers, weight decay handled correctly)"
        elif "rnn" in arch_type_lower or "lstm" in arch_type_lower or "gru" in arch_type_lower:
             if model_size_params > OptimizerHelper.PARAM_THRESHOLD_SMALL:
                 logger.debug("Recommending Adam for moderate/large RNN.")
                 return "Adam (Common for RNNs, consider AdamW as alternative)"
             else:
                 logger.debug("Recommending RMSprop or Adam for small RNN.")
                 return "RMSprop or Adam (RMSprop sometimes preferred for RNNs)"
        elif model_size_params < OptimizerHelper.PARAM_THRESHOLD_SMALL and layer_count < 50:
            logger.debug("Recommending SGD or Adam for small/simple model.")
            return "SGD with Momentum or Adam (Adam is often easier to tune, SGD can generalize better)"
        else: # Moderate CNNs, MLPs, etc.
             logger.debug("Recommending Adam as a general default.")
             return "Adam (Good general default, consider AdamW if weight decay is important)"

    @staticmethod
    def suggest_learning_rate_scheduler(optimizer_name: str) -> str:
        opt_name_lower = optimizer_name.lower()
        logger.info(f"Suggesting scheduler based on optimizer: {optimizer_name}")

        if "adamw" in opt_name_lower:
            logger.debug("Recommending CosineAnnealingLR or ReduceLROnPlateau for AdamW.")
            return "CosineAnnealingLR (Smooth decay) or ReduceLROnPlateau (Adapts to validation metric)"
        elif "adam" in opt_name_lower:
            logger.debug("Recommending StepLR or ReduceLROnPlateau for Adam.")
            return "StepLR (Simple decay) or ReduceLROnPlateau (Adapts to validation metric)"
        elif "sgd" in opt_name_lower:
             logger.debug("Recommending StepLR, MultiStepLR, or CosineAnnealingLR for SGD.")
             return "StepLR/MultiStepLR (Common with SGD) or CosineAnnealingLR (Smooth decay)"
        else:
             logger.debug("Recommending ReduceLROnPlateau as a general fallback scheduler.")
             return "ReduceLROnPlateau (General purpose, adapts to validation metric)"

    @staticmethod
    def adjust_learning_rate_on_plateau(current_lr: float,
                                        plateau_epochs: int,
                                        min_lr: float = 1e-6,
                                        factor: float = 0.1,
                                        patience: int = 10) -> Tuple[Optional[float], str]:

        logger.info(f"Checking LR adjustment: current_lr={current_lr}, plateau_epochs={plateau_epochs}, patience={patience}")

        if plateau_epochs >= patience:
            new_lr = current_lr * factor
            if new_lr < min_lr:
                logger.warning(f"Learning rate reduction below minimum ({min_lr}). Stopping reduction.")
                return None, f"Performance plateaued for {plateau_epochs} epochs. Minimum LR reached, consider stopping."
            else:
                logger.info(f"Performance plateaued for {plateau_epochs} epochs. Reducing LR to {new_lr:.2e}")
                return new_lr, f"Performance plateaued for {plateau_epochs} epochs. Reducing learning rate by factor {factor}."
        else:
            logger.debug("No LR adjustment needed based on plateau duration.")
            return current_lr, "Learning rate stable."

    @staticmethod
    def suggest_initial_learning_rate(architecture_type: str = "Unknown", model_size_params: int = 0) -> float:
         arch_type_lower = architecture_type.lower()
         logger.info(f"Suggesting initial LR for arch={architecture_type}, size={model_size_params:,}")

         if "transformer" in arch_type_lower:
              lr = 1e-4 if model_size_params > OptimizerHelper.PARAM_THRESHOLD_LARGE else 3e-4
              logger.debug(f"Suggesting LR {lr:.1e} for Transformer.")
              return lr
         elif "rnn" in arch_type_lower or "lstm" in arch_type_lower or "gru" in arch_type_lower:
              lr = 1e-3
              logger.debug(f"Suggesting LR {lr:.1e} for RNN.")
              return lr
         elif "cnn" in arch_type_lower:
              lr = 1e-3 if model_size_params < OptimizerHelper.PARAM_THRESHOLD_LARGE else 5e-4
              logger.debug(f"Suggesting LR {lr:.1e} for CNN.")
              return lr
         else: # MLP, Unknown
              lr = 1e-3
              logger.debug(f"Suggesting LR {lr:.1e} as default.")
              return lr