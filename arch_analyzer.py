# TrainSense/arch_analyzer.py
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
import logging
from collections import Counter

logger = logging.getLogger(__name__)

class ArchitectureAnalyzer:
    PARAM_THRESHOLD_SIMPLE = 1_000_000
    PARAM_THRESHOLD_MODERATE = 50_000_000
    PARAM_THRESHOLD_COMPLEX = 100_000_000
    LAYER_THRESHOLD_SIMPLE = 20
    LAYER_THRESHOLD_MODERATE = 100
    LAYER_THRESHOLD_COMPLEX = 200

    RNN_TYPES = {"RNN", "LSTM", "GRU"}
    CNN_TYPES = {"Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d"}
    TRANSFORMER_TYPES = {"Transformer", "TransformerEncoder", "TransformerDecoder", "TransformerEncoderLayer", "TransformerDecoderLayer", "MultiheadAttention"}
    LINEAR_TYPES = {"Linear", "Bilinear"}
    POOLING_TYPES = {"MaxPool1d", "MaxPool2d", "MaxPool3d", "AvgPool1d", "AvgPool2d", "AvgPool3d", "AdaptiveMaxPool1d", "AdaptiveMaxPool2d", "AdaptiveMaxPool3d", "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d"}
    NORMALIZATION_TYPES = {"BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "LayerNorm", "GroupNorm", "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d"}
    ACTIVATION_TYPES = {"ReLU", "LeakyReLU", "PReLU", "ReLU6", "ELU", "SELU", "CELU", "GELU", "Sigmoid", "Tanh", "Softmax", "LogSoftmax", "SiLU", "Mish"}
    DROPOUT_TYPES = {"Dropout", "Dropout2d", "Dropout3d", "AlphaDropout"}

    def __init__(self, model: nn.Module):
        if not isinstance(model, nn.Module):
            raise TypeError("Input 'model' must be an instance of torch.nn.Module.")
        self.model = model
        self._analysis_cache: Optional[Dict[str, Any]] = None
        logger.info(f"ArchitectureAnalyzer initialized for model type: {type(model).__name__}")

    def count_parameters(self) -> Tuple[int, int]:
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total_params, trainable_params

    def count_layers(self, exclude_containers: bool = True) -> int:
        count = 0
        for module in self.model.modules():
            is_container = len(list(module.children())) > 0
            if exclude_containers and not is_container:
                count += 1
            elif not exclude_containers:
                 count +=1
        return count

    def detect_layer_types(self) -> Dict[str, int]:
        layer_types = Counter()
        for module in self.model.modules():
             if len(list(module.children())) == 0: # Count only leaf modules
                layer_types[module.__class__.__name__] += 1
        return dict(layer_types)

    def _recursive_input_shape_search(self, module: nn.Module) -> Optional[Tuple[int, ...]]:
         if hasattr(module, 'in_features') and isinstance(module.in_features, int):
             return (1, module.in_features) # Common for Linear
         if hasattr(module, 'in_channels') and isinstance(module.in_channels, int):
              # Guess spatial dimensions for common CNNs
              if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
                   return (1, module.in_channels, 32, 32) # Common guess
              if isinstance(module, (nn.Conv1d, nn.MaxPool1d, nn.AvgPool1d)):
                   return (1, module.in_channels, 128) # Common guess
              return (1, module.in_channels) # Fallback
         if hasattr(module, 'embedding_dim') and isinstance(module.embedding_dim, int):
             return (1, 10) # Common sequence length guess for embeddings

         for child in module.children():
              shape = self._recursive_input_shape_search(child)
              if shape:
                  return shape
         return None

    def estimate_input_shape(self) -> Optional[Tuple[int, ...]]:
        return self._recursive_input_shape_search(self.model)

    def analyze(self, force_recompute: bool = False) -> Dict[str, Any]:
        if self._analysis_cache is not None and not force_recompute:
            return self._analysis_cache

        logger.info("Starting model architecture analysis.")
        total_params, trainable_params = self.count_parameters()
        layer_count = self.count_layers()
        layer_types = self.detect_layer_types()
        estimated_input_shape = self.estimate_input_shape()

        analysis = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
            "layer_count": layer_count,
            "layer_types_summary": layer_types,
            "estimated_input_shape": estimated_input_shape,
            "primary_architecture_type": self._infer_primary_architecture(layer_types),
            "complexity_category": self._categorize_complexity(total_params, layer_count),
            "recommendation": self._get_architecture_recommendation(total_params, layer_count, layer_types)
        }
        logger.info("Model architecture analysis complete.")
        self._analysis_cache = analysis
        return analysis

    def _infer_primary_architecture(self, layer_types: Dict[str, int]) -> str:
        counts = Counter()
        for layer_name, count in layer_types.items():
            if layer_name in self.TRANSFORMER_TYPES:
                counts["Transformer"] += count
            elif layer_name in self.RNN_TYPES:
                counts["RNN"] += count
            elif layer_name in self.CNN_TYPES:
                counts["CNN"] += count
            elif layer_name in self.LINEAR_TYPES:
                 counts["MLP"] += count # Multi-Layer Perceptron / Fully Connected

        if not counts:
             # Check for common library specific models if direct layers aren't informative
             model_class_name = self.model.__class__.__name__
             if "GPT" in model_class_name or "BERT" in model_class_name or "T5" in model_class_name:
                 return "Transformer (Pre-trained)"
             if "ResNet" in model_class_name or "VGG" in model_class_name or "Inception" in model_class_name:
                 return "CNN (Pre-trained)"
             return "Unknown"

        # Return the type with the highest count, prioritizing complex types
        priority = ["Transformer", "RNN", "CNN", "MLP"]
        for arch_type in priority:
            if counts[arch_type] > 0:
                # Simple heuristic: if multiple types exist, check dominance
                total_layers = sum(counts.values())
                if counts[arch_type] / total_layers > 0.3: # If > 30% of layers are of this type
                     return arch_type
        # Fallback to highest count if no clear dominance
        return counts.most_common(1)[0][0] if counts else "Unknown"


    def _categorize_complexity(self, total_params: int, layer_count: int) -> str:
        if total_params < self.PARAM_THRESHOLD_SIMPLE and layer_count < self.LAYER_THRESHOLD_SIMPLE:
            return "Simple"
        elif total_params < self.PARAM_THRESHOLD_MODERATE and layer_count < self.LAYER_THRESHOLD_MODERATE:
            return "Moderate"
        elif total_params < self.PARAM_THRESHOLD_COMPLEX and layer_count < self.LAYER_THRESHOLD_COMPLEX:
            return "Complex"
        else:
            return "Very Complex / Large"

    def _get_architecture_recommendation(self, total_params: int, layer_count: int, layer_types: Dict[str, int]) -> str:
        complexity = self._categorize_complexity(total_params, layer_count)
        primary_arch = self._infer_primary_architecture(layer_types)
        recs = []

        if complexity == "Simple":
            recs.append(f"Simple model ({total_params:,} params, {layer_count} layers). Consider increasing batch size or model capacity if underfitting.")
        elif complexity == "Moderate":
            recs.append(f"Moderate model ({total_params:,} params, {layer_count} layers). Standard hyperparameters likely suitable; monitor performance.")
        elif complexity == "Complex":
            recs.append(f"Complex model ({total_params:,} params, {layer_count} layers). Ensure sufficient compute resources. Monitor for bottlenecks.")
        else: # Very Complex
            recs.append(f"Very complex/large model ({total_params:,} params, {layer_count} layers). Requires significant compute (GPU memory/time). Consider distributed training, gradient accumulation, or mixed precision.")

        if primary_arch == "Transformer":
             recs.append("Transformer architecture detected. Often benefits from AdamW optimizer and learning rate scheduling (warmup/decay). Sensitive to initialization.")
        elif primary_arch == "RNN":
             recs.append("RNN (LSTM/GRU) architecture detected. Prone to vanishing/exploding gradients; consider gradient clipping. May benefit from lower learning rates.")
        elif primary_arch == "CNN":
             recs.append("CNN architecture detected. Generally robust. Performance depends on kernel sizes, strides, pooling, and normalization choices.")

        if layer_types.get("LayerNorm", 0) > 0 or layer_types.get("BatchNorm1d", 0) > 0 or layer_types.get("BatchNorm2d", 0) > 0:
             recs.append("Normalization layers present, which generally helps stabilize training.")
        else:
             recs.append("No standard normalization layers detected. Consider adding BatchNorm or LayerNorm if training is unstable.")

        if layer_types.get("Dropout", 0) > 0:
             recs.append("Dropout layers present. Helps prevent overfitting; ensure it's disabled during evaluation.")
        else:
             recs.append("No Dropout layers detected. Consider adding if overfitting is observed.")

        return " | ".join(recs)

    def get_summary(self) -> Dict[str, Any]:
         return self.analyze()