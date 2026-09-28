"""Small, dataset-agnostic helpers for numeric CSV time series."""

import csv

import torch


def load_csv_series(path, target=None, limit=None):
    with open(path, newline='', encoding='utf-8') as handle:
        rows = csv.DictReader(handle)
        fieldnames = rows.fieldnames or []
        numeric_fields = [field for field in fieldnames if field != 'date']
        target = target or numeric_fields[-1]
        if target not in fieldnames:
            raise ValueError('target column not found: {}'.format(target))
        values = []
        for row in rows:
            try:
                values.append(float(row[target]))
            except (TypeError, ValueError):
                continue
            if limit and len(values) >= limit:
                break
    if len(values) < 2:
        raise ValueError('CSV must contain at least two numeric target values')
    return torch.tensor(values, dtype=torch.float32)


def make_rounds(values, client_count=2, rounds=2, seq_len=4, pred_len=1,
                batch_size=32):
    if len(values) < client_count * (seq_len + pred_len):
        raise ValueError('dataset is too small for the requested clients and windows')
    streams = list(torch.tensor_split(values, client_count))
    result = []
    for round_index in range(rounds):
        client_round = []
        for stream in streams:
            start = 0 if round_index == 0 else round_index * len(stream) // rounds
            end = len(stream) if round_index == rounds - 1 else (round_index + 1) * len(stream) // rounds
            chunk = stream[start:end]
            batches = []
            window_count = len(chunk) - seq_len - pred_len + 1
            for batch_start in range(0, max(0, window_count), batch_size):
                batch_indices = range(batch_start, min(batch_start + batch_size, window_count))
                inputs = torch.stack([chunk[index:index + seq_len]
                                      for index in batch_indices]).unsqueeze(-1)
                targets = torch.stack([chunk[index + seq_len:index + seq_len + pred_len]
                                       for index in batch_indices]).unsqueeze(-1)
                batches.append((inputs, targets))
            client_round.append(batches)
        result.append(client_round)
    return result