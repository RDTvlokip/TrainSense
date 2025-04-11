# TrainSense/arch_analyzer.py
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional, List # Added List
import logging
from collections import Counter

# Get a logger instance specific to this module
logger = logging.getLogger(__name__)

class ArchitectureAnalyzer:
    """
    Analyzes the architecture of a PyTorch nn.Module.

    Provides information such as parameter counts, layer counts, layer types,
    estimated input shape, inferred architecture type, and complexity assessment.
    """

    # --- Thresholds for Complexity Categorization ---
    PARAM_THRESHOLD_SIMPLE = 1_000_000
    PARAM_THRESHOLD_MODERATE = 50_000_000
    PARAM_THRESHOLD_COMPLEX = 100_000_000 # Models above this start getting very large
    LAYER_THRESHOLD_SIMPLE = 20
    LAYER_THRESHOLD_MODERATE = 100
    LAYER_THRESHOLD_COMPLEX = 200
    # ---------------------------------------------

    # --- Layer Type Classifications ---
    # Sets of common PyTorch layer class names used for architecture inference.
    RNN_TYPES = {"RNN", "LSTM", "GRU"}
    CNN_TYPES = {"Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d"}
    # Added specific Transformer components for better detection
    TRANSFORMER_TYPES = {"Transformer", "TransformerEncoder", "TransformerDecoder",
                         "TransformerEncoderLayer", "TransformerDecoderLayer", "MultiheadAttention"}
    LINEAR_TYPES = {"Linear", "Bilinear"} # Fully connected layers
    POOLING_TYPES = {"MaxPool1d", "MaxPool2d", "MaxPool3d", "AvgPool1d", "AvgPool2d", "AvgPool3d",
                     "AdaptiveMaxPool1d", "AdaptiveMaxPool2d", "AdaptiveMaxPool3d",
                     "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d"}
    NORMALIZATION_TYPES = {"BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "LayerNorm", "GroupNorm",
                           "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d"}
    ACTIVATION_TYPES = {"ReLU", "LeakyReLU", "PReLU", "ReLU6", "ELU", "SELU", "CELU", "GELU",
                        "Sigmoid", "Tanh", "Softmax", "LogSoftmax", "SiLU", "Mish"} # Common activations
    DROPOUT_TYPES = {"Dropout", "Dropout2d", "Dropout3d", "AlphaDropout"}
    EMBEDDING_TYPES = {"Embedding", "EmbeddingBag"} # Added Embedding types
    # ---------------------------------

    def __init__(self, model: nn.Module):
        """
        Initializes the ArchitectureAnalyzer.

        Args:
            model (nn.Module): The PyTorch model to analyze.

        Raises:
            TypeError: If the input 'model' is not an instance of torch.nn.Module.
        """
        if not isinstance(model, nn.Module):
            raise TypeError("Input 'model' must be an instance of torch.nn.Module.")
        self.model = model
        self._analysis_cache: Optional[Dict[str, Any]] = None # Cache for analysis results
        logger.info(f"ArchitectureAnalyzer initialized for model type: {type(model).__name__}")

    def count_parameters(self) -> Tuple[int, int]:
        """
        Counts the total and trainable parameters in the model.

        Returns:
            Tuple[int, int]: A tuple containing (total_parameters, trainable_parameters).
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.debug(f"Parameter count: Total={total_params}, Trainable={trainable_params}")
        return total_params, trainable_params

    def count_layers(self, exclude_containers: bool = True) -> int:
        """
        Counts the number of layers (modules) in the model.

        Args:
            exclude_containers (bool): If True, only counts modules that do not have children
                                       (i.e., excludes containers like nn.Sequential). Defaults to True.

        Returns:
            int: The total number of layers based on the exclusion criteria.
        """
        count = 0
        for module in self.model.modules():
            is_container = len(list(module.children())) > 0
            if exclude_containers and not is_container:
                # Count only leaf modules (no children)
                count += 1
            elif not exclude_containers:
                # Count all modules including containers
                 count +=1
        logger.debug(f"Layer count (exclude_containers={exclude_containers}): {count}")
        return count

    def detect_layer_types(self) -> Dict[str, int]:
        """
        Detects and counts the occurrences of different leaf module types in the model.

        Returns:
            Dict[str, int]: A dictionary mapping layer type names (str) to their counts (int).
        """
        layer_types = Counter()
        # Iterate through all modules in the model
        for module in self.model.modules():
             # Count only leaf modules (modules with no children)
             if len(list(module.children())) == 0:
                layer_types[module.__class__.__name__] += 1
        logger.debug(f"Detected layer types: {dict(layer_types)}")
        return dict(layer_types)

    def _recursive_input_shape_search(self, module: nn.Module) -> Optional[Tuple[int, ...]]:
        """
        Recursively searches for attributes that indicate input dimensionality
        (like 'in_features', 'in_channels', 'embedding_dim') to estimate the model's input shape.
        This is a heuristic and may not always be accurate.

        Args:
            module (nn.Module): The current module being inspected.

        Returns:
            Optional[Tuple[int, ...]]: An estimated input shape tuple (including a batch dimension of 1),
                                      or None if no indicative attribute is found.
        """
        # Check for common attributes indicating input size
        if hasattr(module, 'in_features') and isinstance(module.in_features, int):
            logger.debug(f"Found 'in_features': {module.in_features} in {type(module).__name__}. Estimating shape (1, {module.in_features}).")
            return (1, module.in_features) # Common for Linear layers (Batch, Features)
        if hasattr(module, 'in_channels') and isinstance(module.in_channels, int):
              # Guess spatial dimensions for common CNNs - this is highly approximate
            if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d, nn.BatchNorm2d, nn.InstanceNorm2d)):
                logger.debug(f"Found 'in_channels': {module.in_channels} in 2D layer {type(module).__name__}. Guessing spatial shape (32, 32).")
                return (1, module.in_channels, 32, 32) # Batch, Channels, Height, Width (Common guess)
            if isinstance(module, (nn.Conv1d, nn.MaxPool1d, nn.AvgPool1d, nn.BatchNorm1d, nn.InstanceNorm1d)):
                logger.debug(f"Found 'in_channels': {module.in_channels} in 1D layer {type(module).__name__}. Guessing sequence length 128.")
                return (1, module.in_channels, 128) # Batch, Channels, Sequence Length (Common guess)
            logger.debug(f"Found 'in_channels': {module.in_channels} in layer {type(module).__name__}. Using fallback shape (1, {module.in_channels}).")
            return (1, module.in_channels) # Fallback: Batch, Channels
        if hasattr(module, 'embedding_dim') and isinstance(module, nn.Embedding) and isinstance(module.embedding_dim, int):
            # Embeddings take indices, shape depends on sequence length, difficult to guess accurately
            logger.debug(f"Found 'embedding_dim': {module.embedding_dim} in {type(module).__name__}. Guessing sequence length 10.")
            return (1, 10) # Batch, Sequence Length (Very rough guess for sequence length)

         # Recursively search in children if no direct attribute found
        for child in module.children():
            shape = self._recursive_input_shape_search(child)
            if shape:
                # Return the first shape found in children
                return shape
        # Return None if no shape information found in this branch
        return None

    def estimate_input_shape(self) -> Optional[Tuple[int, ...]]:
        """
        Estimates the model's expected input shape by searching the module tree.
        Calls the recursive helper function starting from the root model.

        Returns:
            Optional[Tuple[int, ...]]: An estimated input shape tuple (Batch=1, ...),
                                      or None if estimation fails.
        """
        logger.info("Attempting to estimate model input shape...")
        estimated_shape = self._recursive_input_shape_search(self.model)
        if estimated_shape:
             logger.info(f"Estimated input shape: {estimated_shape}")
        else:
             logger.warning("Could not estimate input shape from model attributes.")
        return estimated_shape

    def analyze(self, force_recompute: bool = False) -> Dict[str, Any]:
        """
        Performs a comprehensive analysis of the model architecture.
        Results are cached unless `force_recompute` is True.

        Args:
            force_recompute (bool): If True, re-runs the analysis even if cached results exist.
                                    Defaults to False.

        Returns:
            Dict[str, Any]: A dictionary containing various analysis results:
                            - total_parameters (int)
                            - trainable_parameters (int)
                            - non_trainable_parameters (int)
                            - layer_count (int): Count of leaf layers.
                            - layer_types_summary (Dict[str, int]): Counts of each layer type.
                            - estimated_input_shape (Optional[Tuple[int, ...]])
                            - primary_architecture_type (str): Inferred type (e.g., CNN, Transformer).
                            - complexity_category (str): Simple, Moderate, Complex, Very Complex.
                            - recommendations (List[str]): Architecture-specific suggestions.
        """
        # Return cached results if available and not forcing recompute
        if self._analysis_cache is not None and not force_recompute:
            logger.debug("Returning cached architecture analysis.")
            return self._analysis_cache

        logger.info("Starting model architecture analysis.")
        # Perform core analyses
        total_params, trainable_params = self.count_parameters()
        layer_count = self.count_layers(exclude_containers=True) # Count leaf layers
        layer_types = self.detect_layer_types()
        estimated_input_shape = self.estimate_input_shape()

        # Aggregate results into a dictionary
        analysis = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
            "layer_count": layer_count,
            "layer_types_summary": layer_types,
            "estimated_input_shape": estimated_input_shape,
            "primary_architecture_type": self._infer_primary_architecture(layer_types),
            "complexity_category": self._categorize_complexity(total_params, layer_count),
            "recommendations": self._get_architecture_recommendations(total_params, layer_count, layer_types)
        }
        logger.info("Model architecture analysis complete.")
        # Cache the results
        self._analysis_cache = analysis
        return analysis

    def _infer_primary_architecture(self, layer_types: Dict[str, int]) -> str:
        """
        Infers the primary architecture type based on the counts of different layer types.
        Uses predefined sets (CNN_TYPES, RNN_TYPES, TRANSFORMER_TYPES, etc.) and heuristics.

        Args:
            layer_types (Dict[str, int]): Dictionary mapping layer type names to counts.

        Returns:
            str: The inferred primary architecture type (e.g., "Transformer", "CNN", "RNN", "MLP", "Unknown").
        """
        counts = Counter()
        # Count layers belonging to major categories
        for layer_name, count in layer_types.items():
            if layer_name in self.TRANSFORMER_TYPES: counts["Transformer"] += count
            elif layer_name in self.RNN_TYPES: counts["RNN"] += count
            elif layer_name in self.CNN_TYPES: counts["CNN"] += count
            elif layer_name in self.LINEAR_TYPES: counts["MLP"] += count # Multi-Layer Perceptron / Fully Connected
            elif layer_name in self.EMBEDDING_TYPES: counts["EmbeddingBased"] += count # Could be NLP/Transformer

        if not counts:
             # If no indicative layers found, check the model's class name for common patterns
             model_class_name = self.model.__class__.__name__
             logger.debug(f"No standard indicative layers found, checking model class name: {model_class_name}")
             if any(name in model_class_name.upper() for name in ["TRANSFORMER", "GPT", "BERT", "T5", "LLAMA"]):
                 return "Transformer (Pre-trained?)"
             if any(name in model_class_name for name in ["ResNet", "VGG", "Inception", "EfficientNet", "ConvNeXt"]):
                 return "CNN (Pre-trained?)"
             if any(name in model_class_name for name in ["LSTM", "GRU"]):
                 return "RNN (Pre-trained?)"
             logger.warning(f"Could not infer architecture from layer types or model class name '{model_class_name}'.")
             return "Unknown"

        # Define priority for classification (more complex types first)
        priority = ["Transformer", "RNN", "CNN", "MLP", "EmbeddingBased"]
        # Return the highest priority type that has a significant presence
        for arch_type in priority:
            if counts[arch_type] > 0:
                # Heuristic: If a type constitutes >20% of *categorized* layers, consider it primary.
                # This helps differentiate MLP heads in CNNs/Transformers vs. pure MLPs.
                total_categorized_layers = sum(counts.values())
                if total_categorized_layers > 0 and (counts[arch_type] / total_categorized_layers) > 0.2:
                    logger.debug(f"Inferred primary architecture: {arch_type} (dominant type)")
                    return arch_type

        # Fallback: If no type is dominant, return the type with the absolute highest count among prioritized types.
        for arch_type in priority:
             if counts[arch_type] > 0:
                  logger.debug(f"Inferred primary architecture: {arch_type} (highest count fallback)")
                  return arch_type

        # Final fallback if no categorized layers were found (should be rare if model has layers)
        logger.debug("Could not infer architecture based on layer counts.")
        return "Unknown"


    def _categorize_complexity(self, total_params: int, layer_count: int) -> str:
        """
        Categorizes the model's complexity based on parameter and layer counts.

        Args:
            total_params (int): Total number of parameters.
            layer_count (int): Number of leaf layers.

        Returns:
            str: Complexity category ("Simple", "Moderate", "Complex", "Very Complex / Large").
        """
        if total_params >= self.PARAM_THRESHOLD_COMPLEX or layer_count >= self.LAYER_THRESHOLD_COMPLEX:
            return "Very Complex / Large"
        elif total_params >= self.PARAM_THRESHOLD_MODERATE or layer_count >= self.LAYER_THRESHOLD_MODERATE:
            return "Complex"
        elif total_params >= self.PARAM_THRESHOLD_SIMPLE or layer_count >= self.LAYER_THRESHOLD_SIMPLE:
            return "Moderate"
        else:
            return "Simple"

    def _get_architecture_recommendations(self, total_params: int, layer_count: int, layer_types: Dict[str, int]) -> List[str]:
        """
        Generates architecture-specific recommendations based on complexity, type, and layer presence.

        Args:
            total_params (int): Total number of parameters.
            layer_count (int): Number of leaf layers.
            layer_types (Dict[str, int]): Dictionary mapping layer type names to counts.

        Returns:
            List[str]: A list of recommendation strings.
        """
        complexity = self._categorize_complexity(total_params, layer_count)
        primary_arch = self._infer_primary_architecture(layer_types)
        recs = [] # Initialize list for recommendations

        # --- General recommendations based on complexity ---
        if complexity == "Simple":
            recs.append(f"Simple model ({total_params:,} params, {layer_count} layers). If underfitting, consider increasing model capacity or using a more complex architecture. Batch size can likely be increased if memory allows.")
        elif complexity == "Moderate":
            recs.append(f"Moderate model ({total_params:,} params, {layer_count} layers). Standard hyperparameters are often a good starting point. Monitor performance closely.")
        elif complexity == "Complex":
            recs.append(f"Complex model ({total_params:,} params, {layer_count} layers). Ensure sufficient compute resources (GPU memory, time). Monitor for potential bottlenecks (e.g., data loading, specific layers in profiler).")
        else: # Very Complex / Large
            recs.append(f"Very complex/large model ({total_params:,} params, {layer_count} layers). Requires significant compute resources. Consider advanced techniques like distributed training, gradient accumulation, mixed-precision (AMP), model parallelism, or quantization if facing memory/time constraints.")

        # --- Recommendations based on primary architecture type ---
        if primary_arch.startswith("Transformer"):
             recs.append("Transformer architecture detected. AdamW optimizer is strongly recommended. Use learning rate scheduling (warmup/decay). Sensitive to initialization; consider standard initialization schemes. Check attention head performance if possible.")
        elif primary_arch.startswith("RNN"):
             recs.append("RNN (LSTM/GRU) architecture detected. Prone to vanishing/exploding gradients; gradient clipping is highly recommended (e.g., clip norm to 1.0 or 5.0). May benefit from lower learning rates or RMSprop optimizer.")
        elif primary_arch.startswith("CNN"):
             recs.append("CNN architecture detected. Generally robust. Performance heavily depends on kernel sizes, strides, padding, pooling choices, and normalization layers. Adam or SGD w/ momentum are common. Ensure BatchNorm/other norm layers are used appropriately.")
        elif primary_arch == "MLP":
             recs.append("MLP (fully connected) architecture detected. Relatively simple to train. Adam is a common default optimizer. Overfitting can be an issue; consider regularization (dropout, weight decay).")

        # --- Recommendations based on specific layer types ---
        # Normalization
        norm_layers_found = any(layer_name in self.NORMALIZATION_TYPES for layer_name in layer_types)
        if norm_layers_found:
             recs.append("Normalization layers (BatchNorm, LayerNorm, etc.) detected. These generally help stabilize training and improve convergence.")
        else:
             recs.append("No standard normalization layers detected. If training is unstable or slow to converge, consider adding appropriate normalization (e.g., BatchNorm for CNNs, LayerNorm for Transformers/RNNs).")

        # Dropout
        dropout_layers_found = any(layer_name in self.DROPOUT_TYPES for layer_name in layer_types)
        if dropout_layers_found:
             recs.append("Dropout layers detected. Helps prevent overfitting. Ensure dropout is disabled during evaluation/inference (`model.eval()`).")
        else:
             recs.append("No Dropout layers detected. If overfitting is observed on the validation set, consider adding Dropout layers.")

        # Embeddings
        if any(layer_name in self.EMBEDDING_TYPES for layer_name in layer_types):
            recs.append("Embedding layers detected. Common in NLP tasks. Ensure embedding dimensions are appropriate for the vocabulary size and task complexity.")

        logger.debug(f"Generated {len(recs)} architecture recommendations.")
        return recs

    def get_summary(self) -> Dict[str, Any]:
         """
         Returns the complete analysis dictionary. Calls analyze() if not already cached.

         Returns:
             Dict[str, Any]: The dictionary containing analysis results.
         """
         return self.analyze() # Returns the cached or recomputed analysis