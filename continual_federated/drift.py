import torch


class MeanDriftDetector:
    """Detect a shift in the mean of consecutive time-series chunks."""

    def __init__(self, threshold=1.0):
        if threshold < 0:
            raise ValueError('threshold must be non-negative')
        self.threshold = threshold
        self.previous_mean = None

    def update(self, current_chunk):
        current_mean = current_chunk.detach().float().mean(dim=tuple(range(current_chunk.ndim - 1)))
        drift = False
        distance = 0.0
        if self.previous_mean is not None:
            distance = torch.linalg.vector_norm(current_mean - self.previous_mean).item()
            drift = distance > self.threshold
        self.previous_mean = current_mean.cpu()
        return drift, distance

    def reset(self):
        self.previous_mean = None