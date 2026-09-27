import torch
from sklearn.cluster import SpectralClustering


class FedMixerServer:
    """Server utilities for clustered averaging and prediction-space distillation."""

    def __init__(self, cluster_count=None, random_state=2021,
                 recluster_loss_fraction=0.2, recluster_similarity=0.8):
        self.cluster_count = cluster_count
        self.random_state = random_state
        self.recluster_loss_fraction = recluster_loss_fraction
        self.recluster_similarity = recluster_similarity
        self.clusters = None

    @staticmethod
    def _flatten_state(state):
        return torch.cat([value.detach().float().reshape(-1).cpu()
                          for value in state.values()])

    def cluster_clients(self, states, losses=None):
        if not states:
            self.clusters = []
            return self.clusters
        if len(states) == 1:
            self.clusters = [[0]]
            return self.clusters
        vectors = torch.stack([self._flatten_state(state) for state in states])
        distances = torch.cdist(vectors, vectors)
        scale = distances[distances > 0].median().clamp_min(1e-8)
        affinity = torch.exp(-(distances ** 2) / (2 * scale ** 2)).numpy()
        count = self.cluster_count or min(3, len(states))
        count = max(1, min(count, len(states)))
        labels = SpectralClustering(
            n_clusters=count,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=self.random_state,
        ).fit_predict(affinity)
        if losses is not None:
            del losses
        self.clusters = [[index for index, label in enumerate(labels) if label == cluster]
                         for cluster in range(count)]
        return self.clusters

    def should_recluster(self, states, losses):
        if self.clusters is None or not losses:
            return True
        loss_values = torch.tensor(losses, dtype=torch.float32)
        loss_std = loss_values.std(unbiased=False).clamp_min(1e-8)
        loss_mean = loss_values.mean()
        outliers = (torch.abs(loss_values - loss_mean) > loss_std).float().mean()
        if outliers > self.recluster_loss_fraction:
            return True

        vectors = [self._flatten_state(state) for state in states]
        for indices in self.clusters:
            centroid = torch.stack([vectors[index] for index in indices]).mean(dim=0)
            for index in indices:
                similarity = torch.cosine_similarity(vectors[index], centroid, dim=0)
                if similarity < self.recluster_similarity:
                    return True
        return False

    @staticmethod
    def average_cluster(states, client_indices, sample_counts=None):
        if not client_indices:
            raise ValueError('client_indices must not be empty')
        result = {}
        for name in states[client_indices[0]]:
            tensors = [states[index][name].detach().float() for index in client_indices]
            if sample_counts is None:
                weights = torch.ones(len(client_indices), dtype=torch.float32)
            else:
                weights = torch.tensor(
                    [sample_counts[index] for index in client_indices], dtype=torch.float32)
            weights = weights / weights.sum().clamp_min(1e-8)
            result[name] = sum(tensor * weight for tensor, weight in zip(tensors, weights))
        return result

    def aggregate(self, states, losses=None, sample_counts=None, force_recluster=False):
        if force_recluster or self.should_recluster(states, losses):
            self.cluster_clients(states, losses)
        return [self.average_cluster(states, indices, sample_counts)
                for indices in self.clusters]

    @staticmethod
    @torch.no_grad()
    def mutual_distillation(models, public_batch, forward_fn):
        if not models:
            raise ValueError('models must not be empty')
        predictions = [forward_fn(model, public_batch) for model in models]
        ensemble = torch.stack(predictions).mean(dim=0)
        return [ensemble.detach().clone() for _ in models]