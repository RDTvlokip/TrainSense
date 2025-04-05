import time
import torch

class PerformanceTuner:
    def __init__(self, model, device="cpu"):
        self.model = model.to(device)
        self.device = device
    def benchmark(self, input_shape, iterations=50):
        self.model.eval()
        dummy_input = torch.randn(*input_shape).to(self.device)
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)
            start = time.time()
            for _ in range(iterations):
                _ = self.model(dummy_input)
            end = time.time()
        avg_time = (end - start) / iterations
        throughput = 1 / avg_time if avg_time > 0 else 0
        return {"avg_inference_time": avg_time, "throughput": throughput}
    def tune_learning_rate(self, current_lr, throughput, target_throughput):
        if throughput < target_throughput:
            new_lr = current_lr * 0.9
            message = "Réduction du learning rate pour stabiliser l'entraînement"
        elif throughput > target_throughput * 1.1:
            new_lr = current_lr * 1.1
            message = "Augmentation du learning rate pour accélérer l'entraînement"
        else:
            new_lr = current_lr
            message = "Learning rate adapté"
        return new_lr, message