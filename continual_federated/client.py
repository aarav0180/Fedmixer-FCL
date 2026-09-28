import torch.nn.functional as F

from .drift import MeanDriftDetector
from .memory import ReplayMemory


class ContinualClient:
    """Client-side continual training wrapper around an unchanged forecasting model."""

    def __init__(self, model, optimizer, device, memory_size=200,
                 memory_batch_size=16, memory_weight=0.1, distill_weight=0.1,
                 drift_threshold=1.0):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.memory = ReplayMemory(memory_size)
        self.memory_batch_size = memory_batch_size
        self.memory_weight = memory_weight
        self.distill_weight = distill_weight
        self.drift_detector = MeanDriftDetector(drift_threshold)
        self.last_drift_distance = 0.0

    def observe(self, inputs):
        drift, distance = self.drift_detector.update(inputs)
        self.last_drift_distance = distance
        return drift

    def train_batch(self, inputs, targets, forward_fn, distillation_inputs=None,
                    distillation_targets=None, use_replay=True):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        self.model.train()
        self.optimizer.zero_grad()
        predictions = forward_fn(self.model, inputs)
        loss = F.mse_loss(predictions, targets)

        if use_replay:
            replay = self.memory.sample(self.memory_batch_size, self.device)
            if replay is not None:
                replay_inputs, replay_targets, _ = replay
                replay_predictions = forward_fn(self.model, replay_inputs)
                loss = loss + self.memory_weight * F.mse_loss(replay_predictions, replay_targets)

        if distillation_inputs is not None and distillation_targets is not None:
            distill_predictions = forward_fn(self.model, distillation_inputs.to(self.device))
            loss = loss + self.distill_weight * F.mse_loss(
                distill_predictions, distillation_targets.to(self.device))

        loss.backward()
        self.optimizer.step()
        if use_replay:
            self.memory.add_batch(inputs, targets, predictions)
        return loss.detach().item()