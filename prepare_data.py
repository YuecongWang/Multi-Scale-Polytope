"""Build a documented CIFAR-10 research split from the official training set."""
import argparse
from pathlib import Path

from experiment_recorder import digest, save_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='datasets')
    parser.add_argument('--output', default='datasets/CIFAR10_TRAIN_Split.pth')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--clean-per-class', type=int, default=4800)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.clean_per_class < 5000:
        parser.error('clean-per-class must lie within 1..4999')
    output = Path(args.output)
    if output.exists() or output.with_suffix('.json').exists():
        parser.error('Output exists; select another --output to preserve the split identity')
    import numpy as np
    import torch
    from torchvision.datasets import CIFAR10
    dataset = CIFAR10(args.root, train=True, download=args.download)
    labels = np.asarray(dataset.targets)
    rng = np.random.RandomState(args.seed)
    ids = {'clean_train': [], 'others': []}
    for label in range(10):
        candidates = np.flatnonzero(labels == label)
        if len(candidates) != 5000:
            raise ValueError('Expected official CIFAR-10 training data: 5000 images/class')
        selected = rng.permutation(candidates).tolist()
        ids['clean_train'].extend(selected[:args.clean_per_class])
        ids['others'].extend(selected[args.clean_per_class:])
    if set(ids['clean_train']) & set(ids['others']):
        raise ValueError('Split source IDs overlap')
    splits = {name: [dataset[index] for index in indices] for name, indices in ids.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    torch.save(splits, temporary)
    temporary.replace(output)
    save_json(output.with_suffix('.json'), dict(
        schema_version=1, source='torchvision CIFAR10 train=True; official test set excluded',
        protocol='new deterministic research split, not the upstream authors original split',
        seed=args.seed, clean_per_class=args.clean_per_class,
        others_per_class=5000-args.clean_per_class, source_ids=ids,
        split_sha256=digest(output)))
    print(f'Saved {output}: clean_train={len(splits["clean_train"])}, others={len(splits["others"])}')


if __name__ == '__main__':
    main()
