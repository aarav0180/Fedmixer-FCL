"""CPU-only smoke demo for continual FedMixer components."""

import torch
from torch import nn

from .client import ContinualClient
from .server import FedMixerServer


def windows(values):
    values = torch.tensor(values, dtype=torch.float32).view(-1, 1, 1)
    return values[:-1], values[1:]


def forward(model, inputs):
    return model(inputs).view_as(inputs)


def main():
    torch.manual_seed(7)
    values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    streams = [values[:5], values[5:]]
    clients = []
    for _ in streams:
        model = nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        clients.append(ContinualClient(
            model, optimizer, torch.device('cpu'), memory_size=4,
            memory_batch_size=2, memory_weight=0.2, drift_threshold=1.5))

    server = FedMixerServer(cluster_count=2)
    for round_index, chunk_start in enumerate((0, 3), start=1):
        states = []
        for client, stream in zip(clients, streams):
            chunk = stream[chunk_start:]
            inputs, targets = windows(chunk)
            if len(inputs) == 0:
                continue
            client.observe(inputs)
            client.train_batch(inputs, targets, forward)
            states.append({name: value.detach().clone() for name, value in client.model.state_dict().items()})

        clusters = server.cluster_clients(states)
        cluster_states = server.aggregate(states)
        for cluster_index, indices in enumerate(clusters):
            for index in indices:
                clients[index].model.load_state_dict(cluster_states[cluster_index])

        public_inputs, _ = windows(values[:5])
        distillation = server.mutual_distillation(
            [client.model for client in clients], public_inputs, forward)
        print('round={} clusters={} memory={} distill_shape={}'.format(
            round_index, clusters, [len(client.memory) for client in clients],
            tuple(distillation[0].shape)))


if __name__ == '__main__':
    main()