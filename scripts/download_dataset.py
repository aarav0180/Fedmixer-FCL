"""Download a small, public ETTh1 CSV for local CPU experiments."""

import argparse
import os
from urllib.request import urlopen


URL = 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='./data/ETT/ETTh1_tiny.csv')
    parser.add_argument('--max-rows', type=int, default=64)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with urlopen(URL, timeout=30) as response:
        lines = response.read().decode('utf-8').splitlines()
    if args.max_rows > 0:
        lines = lines[:args.max_rows + 1]
    with open(args.output, 'w', encoding='utf-8', newline='\n') as handle:
        handle.write('\n'.join(lines) + '\n')
    print('saved {} rows to {}'.format(max(0, len(lines) - 1), args.output))


if __name__ == '__main__':
    main()