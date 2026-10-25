"""Summarize paired victim evaluations without reading training logs or running models."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

from experiment_recorder import save_json


def summarize(root):
    grouped = defaultdict(list)
    seen = set()
    shared_protocol = None
    bases_by_target = {}
    for path in sorted(Path(root).rglob('evaluation.json')):
        payload = json.loads(path.read_text(encoding='utf-8'))
        method = payload['protocol']['method']
        if 'parameters' in payload['protocol']:
            protocol = payload['protocol']
            common = dict(parameters={key: value for key, value in protocol['parameters'].items()
                                      if key not in ('mode', 'feature_mode', 'target_index')},
                          data=protocol['data'], substitutes=protocol['substitutes'], victims=protocol['victims'])
            if shared_protocol is not None and common != shared_protocol:
                raise ValueError('Different training/data protocols in one summary root: ' + str(path))
            shared_protocol = common
        for result in payload['results']:
            identity = (method, result['victim'], result['seed'], result['target_index'])
            if identity in seen:
                raise ValueError('Duplicate evaluation identity: ' + str(identity))
            seen.add(identity)
            if 'base_indices' in payload:
                target = result['target_index']
                if target in bases_by_target and bases_by_target[target] != payload['base_indices']:
                    raise ValueError('Methods used different base indices for target ' + str(target))
                bases_by_target[target] = payload['base_indices']
            grouped[identity[:3]].append(result)
    rows = []
    for (method, victim, seed), results in sorted(grouped.items()):
        count = len(results)
        eligible = [result for result in results if result['clean_target_correct']]
        clean = sum(result['clean']['clean_accuracy_percent'] for result in results) / count
        poisoned = sum(result['poisoned']['clean_accuracy_percent'] for result in results) / count
        rows.append(dict(method=method, victim=victim, seed=seed, targets=count,
            eligible_targets=len(eligible), successes=sum(result['attack_success'] for result in results),
            asr_percent=100 * sum(result['attack_success'] for result in results) / count,
            conditional_asr_percent=(100 * sum(result['attack_success'] for result in eligible) / len(eligible)
                                     if eligible else None),
            mean_clean_accuracy_percent=clean, mean_poisoned_accuracy_percent=poisoned,
            clean_accuracy_drop_points=clean-poisoned))
    return rows, seen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', help='One experiment output_root; do not mix different protocols')
    parser.add_argument('--config', help='Require all configured method/target/victim evaluations')
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args(argv)
    try:
        rows, seen = summarize(args.root)
        if not rows:
            raise ValueError('No evaluation.json files found')
        missing = set()
        unexpected = set()
        if args.config:
            from workflow import load_config
            config = load_config(args.config)
            expected = {(method, victim, config.get('seed', 2026), target)
                        for method in config['attack']['methods'] for victim in config['victim']['nets']
                        for target in config['attack']['targets']}
            missing, unexpected = expected-seen, seen-expected
        output = Path(args.root)
        save_json(output / 'summary.json', dict(rows=rows, missing=sorted(missing), unexpected=sorted(unexpected)))
        with (output / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for row in rows:
            conditional = row['conditional_asr_percent']
            value = 'N/A (zero eligible targets)' if conditional is None else f'{conditional:.2f}%'
            print(f'{row["method"]} / {row["victim"]} / seed{row["seed"]}: '
                  f'ASR={row["asr_percent"]:.2f}%, conditional={value}, n={row["targets"]}')
        if (missing or unexpected) and not args.allow_incomplete:
            raise ValueError(f'Incomplete/mixed evaluation: {len(missing)} missing, {len(unexpected)} unexpected; see summary.json')
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
