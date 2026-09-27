from collections import deque

import torch


class ReplayMemory:
    """Bounded CPU replay memory for input windows, targets, and old outputs."""

    def __init__(self, capacity=200):
        if capacity < 1:
            raise ValueError('capacity must be positive')
        self.capacity = capacity
        self._items = deque(maxlen=capacity)

    def __len__(self):
        return len(self._items)

    def add_batch(self, inputs, targets, logits=None):
        inputs = inputs.detach().cpu()
        targets = targets.detach().cpu()
        logits = None if logits is None else logits.detach().cpu()
        for index in range(inputs.shape[0]):
            item = (inputs[index].clone(), targets[index].clone())
            if logits is not None:
                item += (logits[index].clone(),)
            self._items.append(item)

    def sample(self, batch_size, device):
        if not self._items:
            return None
        count = min(batch_size, len(self._items))
        indices = torch.randperm(len(self._items))[:count].tolist()
        items = [self._items[index] for index in indices]
        inputs = torch.stack([item[0] for item in items]).to(device)
        targets = torch.stack([item[1] for item in items]).to(device)
        logits = None
        if len(items[0]) == 3:
            logits = torch.stack([item[2] for item in items]).to(device)
        return inputs, targets, logits