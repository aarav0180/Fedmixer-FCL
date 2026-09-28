"""Compare FedMixer with and without continual replay/drift on Electricity clients."""

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from continual_federated import ContinualClient, ContinualFedMixer
from continual_federated.csv_data import load_csv_series, make_rounds
from models.TimeMixer import Model


def make_model(device):
    configs = SimpleNamespace(
        task_name='long_term_forecast', seq_len=96, label_len=48, pred_len=96,
        down_sampling_window=2, down_sampling_layers=1, channel_independence=1,
        e_layers=2, moving_avg=25, enc_in=1, c_out=1, use_future_temporal_feature=0,
        d_model=16, embed='timeF', freq='h', dropout=0.1, use_norm=1,
        decomp_method='moving_avg', d_ff=32, top_k=5, factor=1,
        activation='gelu', down_sampling_method='avg',
    )
    return Model(configs).float().to(device)


def build_rounds(data_dir, client_count, rounds, limit):
    client_batches = []
    for client_index in range(client_count):
        path = Path(data_dir) / 'clients' / 'client_{:03d}.csv'.format(client_index)
        values = load_csv_series(path, target='load', limit=limit)
        client_batches.append(make_rounds(values, 1, rounds, 96, 96))
    normalized_rounds = []
    for round_index in range(rounds):
        normalized_clients = []
        for client_index in range(client_count):
            batches = client_batches[client_index][round_index]
            if len(batches) != 1:
                raise ValueError(
                    'client {} round {} must contain exactly one complete batch'.format(
                        client_index, round_index + 1))
            normalized_clients.append(batches[0])
        normalized_rounds.append(normalized_clients)
    return normalized_rounds


def run_variant(round_batches, device, continual, seed):
    torch.manual_seed(seed)
    clients = []
    for _ in round_batches[0]:
        model = make_model(device)
        clients.append(ContinualClient(
            model, torch.optim.Adam(model.parameters(), lr=0.001), device,
            memory_size=200, memory_batch_size=16,
            memory_weight=0.1, distill_weight=0.1, drift_threshold=1.0))

    def forward(model, inputs):
        return model(inputs, None, None, None)

    first_client_round = round_batches[0][0]
    if isinstance(first_client_round, list):
        first_client_round = first_client_round[0]
    public_inputs = first_client_round[0].to(device)
    trainer = ContinualFedMixer(clients, forward, public_inputs)
    history = trainer.fit(round_batches, continual=continual)

    mse_values = []
    mae_values = []
    with torch.no_grad():
        for client, (inputs, targets) in zip(clients, round_batches[-1]):
            predictions = forward(client.model, inputs.to(device))
            expected = targets.to(device)
            mse_values.append(F.mse_loss(predictions, expected).item())
            mae_values.append(F.l1_loss(predictions, expected).item())
    return history, sum(mse_values) / len(mse_values), sum(mae_values) / len(mae_values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='./data/electricity_federated')
    parser.add_argument('--clients', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--limit', type=int, default=1000)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--mode', choices=['baseline', 'continual', 'both'], default='both')
    parser.add_argument('--output', default='./output/output.log')
    args = parser.parse_args()
    device = torch.device(args.device)
    round_batches = build_rounds(args.data_dir, args.clients, args.rounds, args.limit)

    variants = {
        'baseline': [('FedMixer baseline', False)],
        'continual': [('Continual FedMixer', True)],
        'both': [('FedMixer baseline', False), ('Continual FedMixer', True)],
    }
    output_lines = []
    for name, continual in variants[args.mode]:
        history, mse, mae = run_variant(round_batches, device, continual, seed=2021)
        output_lines.append('{}: MSE={:.6f} MAE={:.6f} final_clusters={} memory={}'.format(
            name, mse, mae, history[-1]['clusters'], history[-1]['memory']))

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'a', encoding='utf-8') as handle:
        handle.write('\n=== mode: {} ===\n'.format(args.mode))
        handle.write('\n'.join(output_lines) + '\n')
    print('\n'.join(output_lines))
    print('results saved to {}'.format(args.output))


if __name__ == '__main__':
    main()