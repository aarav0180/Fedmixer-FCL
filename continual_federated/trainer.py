"""Temporal-round orchestration for continual federated forecasting."""

import torch

from .server import FedMixerServer


class ContinualFedMixer:
    """Run local replay, clustered averaging, and mutual distillation per round.

    ``round_batches`` is a list of rounds. Each round contains one list of
    ``(inputs, targets)`` batches per client. The forecasting model and its
    unchanged forward path are supplied by the caller.
    """

    def __init__(self, clients, forward_fn, public_inputs=None, server=None):
        if not clients:
            raise ValueError('clients must not be empty')
        self.clients = clients
        self.forward_fn = forward_fn
        self.public_inputs = public_inputs
        self.server = server or FedMixerServer()

    def fit(self, round_batches):
        history = []
        for round_index, client_batches in enumerate(round_batches, start=1):
            if len(client_batches) != len(self.clients):
                raise ValueError('each round must contain one entry per client')

            if self.public_inputs is not None and self.server.clusters is not None:
                public_targets = self._distillation_targets()
            else:
                public_targets = [None] * len(self.clients)

            losses = []
            sample_counts = []
            drifted = []
            for client_index, (client, batches) in enumerate(zip(self.clients, client_batches)):
                client_loss = []
                sample_count = 0
                client_drift = False
                for inputs, targets in batches:
                    client_drift = client.observe(inputs) or client_drift
                    client_loss.append(client.train_batch(
                        inputs, targets, self.forward_fn,
                        self.public_inputs, public_targets[client_index]))
                    sample_count += len(inputs)
                losses.append(sum(client_loss) / max(1, len(client_loss)))
                sample_counts.append(sample_count)
                drifted.append(client_drift)

            states = [{name: value.detach().clone()
                       for name, value in client.model.state_dict().items()}
                      for client in self.clients]
            force_recluster = round_index == 1 or any(drifted)
            cluster_states = self.server.aggregate(
                states, losses, sample_counts, force_recluster=force_recluster)
            for cluster_index, indices in enumerate(self.server.clusters):
                for client_index in indices:
                    self.clients[client_index].model.load_state_dict(cluster_states[cluster_index])

            history.append({
                'round': round_index,
                'clusters': [list(indices) for indices in self.server.clusters],
                'losses': losses,
                'drifted': drifted,
                'memory': [len(client.memory) for client in self.clients],
            })
        return history

    def _distillation_targets(self):
        representatives = [self.clients[indices[0]].model
                            for indices in self.server.clusters]
        with torch.no_grad():
            predictions = [self.forward_fn(model, self.public_inputs)
                           for model in representatives]
            ensemble = torch.stack(predictions).mean(dim=0).detach()
        return [ensemble.clone() for _ in self.clients]