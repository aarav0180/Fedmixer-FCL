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
            if limit is not None and len(values) >= limit:
                break
    if len(values) < 2:
        raise ValueError('CSV must contain at least two numeric target values')
    return torch.tensor(values, dtype=torch.float32)


def make_rounds(values, client_count=2, rounds=2, seq_len=4, pred_len=1):
    if len(values) < client_count * (seq_len + pred_len):
        raise ValueError('dataset is too small for the requested clients and windows')
    streams = list(torch.tensor_split(values, client_count))
    result = []
    for round_index in range(rounds):
        client_round = []
        for stream in streams:
            start = round_index * (seq_len + pred_len)
            chunk = stream[start:start + seq_len + pred_len]
            batches = []
            if len(chunk) == seq_len + pred_len:
                inputs = torch.stack([chunk[index:index + seq_len]
                                      for index in range(pred_len)]).unsqueeze(-1)
                targets = torch.stack([chunk[index + seq_len:index + seq_len + pred_len]
                                       for index in range(pred_len)]).unsqueeze(-1)
                batches.append((inputs, targets))
            client_round.append(batches)
        result.append(client_round)
    return result