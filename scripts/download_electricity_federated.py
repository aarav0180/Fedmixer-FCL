"""Download and prepare the Electricity Consumption paper dataset.

This script is non-interactive and uses only the Python standard library.
It creates one CSV per household client plus a continual-learning task manifest.
"""

import argparse
import csv
import json
import os
import tempfile
import urllib.request
import zipfile


URL = (
    'https://archive.ics.uci.edu/static/public/321/'
    'electricityloaddiagrams20112014.zip'
)


def parse_value(raw):
    return float(raw.replace(',', '.'))


def read_source(path, max_rows=None):
    with zipfile.ZipFile(path) as archive:
        member = next(name for name in archive.namelist() if name.endswith('.txt'))
        with archive.open(member) as stream:
            reader = csv.reader((line.decode('utf-8') for line in stream), delimiter=';')
            header = next(reader)
            client_ids = header[1:]
            rows = []
            for row in reader:
                if len(row) != len(header):
                    continue
                rows.append((row[0], row[1:]))
                if max_rows and len(rows) >= max_rows:
                    break
    return client_ids, rows


def prepare(output, client_limit, max_rows, rounds, seq_len, pred_len):
    os.makedirs(output, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = os.path.join(temp_dir, 'electricity.zip')
        print('downloading {}'.format(URL))
        urllib.request.urlretrieve(URL, archive_path)
        client_ids, rows = read_source(archive_path, max_rows)

    complete = []
    for index, client_id in enumerate(client_ids):
        values = []
        valid = True
        for timestamp, row_values in rows:
            try:
                values.append((timestamp, parse_value(row_values[index])))
            except (IndexError, ValueError):
                valid = False
                break
        if valid and values:
            complete.append((client_id, values))
        if len(complete) >= client_limit:
            break

    if len(complete) < client_limit:
        raise RuntimeError('only {} complete clients found'.format(len(complete)))

    clients_dir = os.path.join(output, 'clients')
    os.makedirs(clients_dir, exist_ok=True)
    manifest = {
        'dataset': 'Electricity Consumption Dataset',
        'source': URL,
        'sampling': '15 minutes',
        'client_count': len(complete),
        'sequence_length': seq_len,
        'prediction_length': pred_len,
        'round_count': rounds,
        'clients': [],
    }

    for client_number, (client_id, values) in enumerate(complete):
        filename = 'client_{:03d}.csv'.format(client_number)
        path = os.path.join(clients_dir, filename)
        with open(path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['timestamp', 'load'])
            writer.writerows(values)

        task_size = len(values) // rounds
        tasks = []
        for round_index in range(rounds):
            start = round_index * task_size
            end = len(values) if round_index == rounds - 1 else (round_index + 1) * task_size
            windows = max(0, end - start - seq_len - pred_len + 1)
            tasks.append({'round': round_index + 1, 'start': start, 'end': end, 'windows': windows})
        manifest['clients'].append({
            'client_id': client_id,
            'file': os.path.join('clients', filename).replace('\\', '/'),
            'tasks': tasks,
        })

    manifest_path = os.path.join(output, 'manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)
    print('prepared {} clients and {} rounds in {}'.format(len(complete), rounds, output))
    print('manifest: {}'.format(manifest_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='./data/electricity_federated')
    parser.add_argument('--clients', type=int, default=10,
                        help='number of complete household clients to prepare')
    parser.add_argument('--max-rows', type=int, default=1000,
                        help='rows to keep per client; use 0 for the full archive')
    parser.add_argument('--rounds', type=int, default=4,
                        help='chronological continual-learning tasks per client')
    parser.add_argument('--seq-len', type=int, default=96)
    parser.add_argument('--pred-len', type=int, default=96)
    args = parser.parse_args()
    prepare(args.output, args.clients, args.max_rows or None,
            args.rounds, args.seq_len, args.pred_len)


if __name__ == '__main__':
    main()