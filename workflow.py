"""Plan and run a consistent pretrain -> craft -> evaluate research workflow."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from experiment_protocol import METHODS, checkpoint_path, method_name, resolve_training_options, validate_attack
from train_cifar10_models import MODEL_NAMES

ROOT = Path(__file__).resolve().parent


def method_training_options(attack, victim, method):
    """Preserve method defaults while allowing independent victim fine-tuning."""
    mode, feature, default_blocks = METHODS[method]
    settings = resolve_training_options(
        None, feature,
        attack_representation=attack.get('representation', 'blocks' if default_blocks else 'features'),
        victim_training=victim.get('training', 'end2end' if default_blocks else 'linear'),
        victim_model_mode=victim.get('model_mode', 'eval'))
    canonical = method_name(mode, feature, settings['attack_end2end'])
    if canonical != method:
        raise ValueError(f'attack.representation changes {method} into {canonical}; '
                         f'use the matching method label {canonical}')
    return settings


def load_config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    allowed = {'schema_version', 'name', 'dataset', 'model_directory', 'output_root', 'device',
               'gpu', 'seed', 'pretrain', 'attack', 'victim'}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError('Unknown configuration keys: ' + ', '.join(sorted(unknown)))
    if config.get('schema_version') != 1:
        raise ValueError('schema_version must be 1')
    sections = {
        'dataset': {'split', 'root'},
        'pretrain': {'epochs', 'lr', 'batch_size', 'seed', 'sidx', 'eidx', 'workers', 'milestones'},
        'attack': {'methods', 'targets', 'substitute_nets', 'checkpoints', 'dropouts', 'net_repeat',
                   'target_label', 'poison_label', 'target_start', 'poison_num', 'iterations', 'lr',
                   'epsilon', 'optimizer', 'decay_iterations', 'decay_ratio', 'subset_group', 'representation'},
        'victim': {'nets', 'checkpoint', 'epochs', 'lr', 'optimizer', 'momentum', 'weight_decay',
                   'batch_size', 'num_per_class', 'milestones', 'training', 'model_mode'},
    }
    for section, keys in sections.items():
        if section not in config or not isinstance(config[section], dict):
            raise ValueError('Missing configuration object: ' + section)
        extra = set(config[section]) - keys
        if extra:
            raise ValueError(f'Unknown {section} keys: ' + ', '.join(sorted(extra)))
    a, v, p = config['attack'], config['victim'], config['pretrain']
    for key in ('methods', 'targets', 'substitute_nets', 'checkpoints', 'dropouts'):
        if not isinstance(a.get(key), list) or not a[key]:
            raise ValueError('attack.' + key + ' must be a nonempty list')
    if not isinstance(v.get('nets'), list) or not v['nets']:
        raise ValueError('victim.nets must be nonempty')
    if len(set(a['methods'])) != len(a['methods']) or len(set(a['targets'])) != len(a['targets']):
        raise ValueError('Methods and target indices must be unique')
    if len(a['checkpoints']) != len(a['dropouts']):
        raise ValueError('Each checkpoint template needs one dropout value')
    if any(not 0 <= value < 1 for value in a['dropouts']):
        raise ValueError('Dropout must lie within [0, 1)')
    if config.get('device', 'cuda') not in ('cuda', 'cpu'):
        raise ValueError('device must be cuda or cpu')
    uses_blocks = False
    canonical_methods = set()
    for method in a['methods']:
        if method not in METHODS:
            raise ValueError('Unknown method: ' + method)
        mode, feature, _ = METHODS[method]
        settings = method_training_options(a, v, method)
        canonical = method_name(mode, feature, settings['attack_end2end'])
        if canonical in canonical_methods:
            raise ValueError('Method labels select the same canonical algorithm: ' + canonical)
        canonical_methods.add(canonical)
        validate_attack(mode, feature, settings['attack_end2end'],
                        a.get('net_repeat', 1), a.get('poison_num', 5))
        uses_blocks = uses_blocks or settings['attack_end2end']
    names = a['substitute_nets'] + v['nets']
    if set(names) - set(MODEL_NAMES):
        raise ValueError('Unknown architecture: ' + ', '.join(sorted(set(names)-set(MODEL_NAMES))))
    if uses_blocks:
        unsupported = [name for name in a['substitute_nets']
                       if not name.startswith(('ResNet', 'ResNeXt', 'DPN', 'SENet'))]
        if unsupported:
            raise ValueError('Original E2E block interface unavailable for: ' + ', '.join(unsupported))
    if not 0 <= a.get('epsilon', .1) <= 1:
        raise ValueError('epsilon must lie within [0, 1] in pixel coordinates')
    if a.get('poison_num', 5) > v.get('num_per_class', 50):
        raise ValueError('Poison count exceeds per-class training budget')
    for value in a['targets']:
        if not isinstance(value, int) or value < 0:
            raise ValueError('Target indices must be nonnegative integers')
        position = a.get('target_start', 50) + value
        start = a.get('subset_group', 0) * v.get('num_per_class', 50)
        if start <= position < start + v.get('num_per_class', 50):
            raise ValueError('Target overlaps the victim training subset')
    if not 0 <= a.get('target_label', 6) < 10 or not 0 <= a.get('poison_label', 8) < 10:
        raise ValueError('CIFAR-10 labels must lie within 0..9')
    if a.get('target_label', 6) == a.get('poison_label', 8):
        raise ValueError('Target and poison classes must differ')
    for section, fields in ((a, ('iterations', 'poison_num', 'net_repeat')),
                            (v, ('epochs', 'batch_size', 'num_per_class')), (p, ('epochs', 'batch_size'))):
        for field in fields:
            if field in section and (not isinstance(section[field], int) or section[field] < 1):
                raise ValueError(field + ' must be a positive integer')
    if p.get('sidx', 0) < 0 or p.get('sidx', 0) >= p.get('eidx', 4800):
        raise ValueError('Pretraining requires 0 <= sidx < eidx')
    if a.get('subset_group', 0) < 0 or a.get('target_start', 50) < 0:
        raise ValueError('subset_group and target_start must be nonnegative')
    for section in (a, v, p):
        if section.get('lr', .1) <= 0:
            raise ValueError('Learning rates must be positive')
    if a.get('optimizer', 'adam') not in ('adam', 'sgd', 'signedadam'):
        raise ValueError('Unknown attack optimizer')
    if v.get('optimizer', 'adam') not in ('adam', 'sgd'):
        raise ValueError('Unknown victim optimizer')
    # Resolve project-relative paths once, regardless of the calling shell's directory.
    for section, key in ((config['dataset'], 'split'), (config['dataset'], 'root'),
                         (config, 'model_directory'), (config, 'output_root')):
        if not section.get(key):
            raise ValueError('Missing path: ' + key)
        section[key] = str((ROOT / section[key]).resolve())
    return config


def add_option(command, name, value):
    if isinstance(value, list):
        command.extend(['--' + name, *map(str, value)])
    else:
        command.extend(['--' + name, str(value)])


def build_plan(config):
    a, v, p, data = (config[key] for key in ('attack', 'victim', 'pretrain', 'dataset'))
    plan = {'pretrain': [], 'craft': [], 'evaluate': []}
    models = {}
    requests = [(net, template, dropout) for template, dropout in zip(a['checkpoints'], a['dropouts'])
                for net in a['substitute_nets']]
    requests += [(net, v['checkpoint'], 0.0) for net in v['nets']]
    for net, template, dropout in requests:
        destination = checkpoint_path(config['model_directory'], template, net)
        definition = (net, dropout)
        if str(destination) in models and models[str(destination)] != definition:
            raise ValueError('Different models/dropout values would overwrite checkpoint: ' + str(destination))
        if str(destination) in models:
            continue
        models[str(destination)] = definition
        cmd = [sys.executable, str(ROOT / 'train_cifar10_models.py')]
        options = {'net': net, 'train-dp': dropout, 'checkpoint-name': destination.name,
                   'output-dir': str(destination.parent), 'data-path': data['split'], 'dset-path': data['root'],
                   'device': config.get('device', 'cuda'), 'gpu': config.get('gpu', '0'),
                   'epochs': p.get('epochs', 300), 'lr': p.get('lr', .1), 'batchsize': p.get('batch_size', 128),
                   'seed': p.get('seed', 1226), 'sidx': p.get('sidx', 0), 'eidx': p.get('eidx', 4800),
                   'workers': p.get('workers', 0), 'milestones': p.get('milestones', [150, 225])}
        for key, value in options.items():
            add_option(cmd, key, value)
        plan['pretrain'].append(cmd)
    for target in a['targets']:
        for method in a['methods']:
            mode, feature, _ = METHODS[method]
            settings = method_training_options(a, v, method)
            options = {'gpu': config.get('gpu', '0'), 'device': config.get('device', 'cuda'),
                'seed': config.get('seed', 2026), 'mode': mode, 'feature-mode': feature,
                'attack-representation': settings['attack_representation'],
                'victim-training': settings['victim_training'],
                'victim-model-mode': settings['victim_model_mode'],
                'substitute-nets': a['substitute_nets'],
                'subs-chk-name': a['checkpoints'], 'subs-dp': a['dropouts'], 'target-net': v['nets'],
                'test-chk-name': v['checkpoint'], 'model-resume-path': config['model_directory'],
                'train-data-path': data['split'], 'dset-path': data['root'],
                'target-index': target, 'target-start': a.get('target_start', 50),
                'target-label': a.get('target_label', 6), 'poison-label': a.get('poison_label', 8),
                'poison-num': a.get('poison_num', 5), 'poison-ites': a.get('iterations', 800),
                'poison-lr': a.get('lr', .04), 'poison-epsilon': a.get('epsilon', .1),
                'poison-opt': a.get('optimizer', 'adam'), 'net-repeat': a.get('net_repeat', 1),
                'poison-decay-ites': a.get('decay_iterations', []), 'poison-decay-ratio': a.get('decay_ratio', .1),
                'subset-group': a.get('subset_group', 0), 'retrain-epochs': v.get('epochs', 60),
                'retrain-lr': v.get('lr', .1), 'retrain-opt': v.get('optimizer', 'adam'),
                'retrain-momentum': v.get('momentum', .9), 'retrain-wd': v.get('weight_decay', 0),
                'retrain-bsize': v.get('batch_size', 64), 'num-per-class': v.get('num_per_class', 50),
                'lr-decay-epoch': v.get('milestones', [30, 45]),
                'chk-path': str(Path(config['output_root']) / method)}
            for stage in ('craft', 'evaluate'):
                cmd = [sys.executable, str(ROOT / 'craft_poisons_transfer.py'), '--stage', stage]
                for key, value in options.items():
                    add_option(cmd, key, value)
                plan[stage].append(cmd)
    return plan


def preflight(config, stage):
    paths = [Path(config['dataset']['split'])]
    if stage in ('craft', 'evaluate'):
        a, v = config['attack'], config['victim']
        paths.extend(checkpoint_path(config['model_directory'], template, net)
                     for template in a['checkpoints'] for net in a['substitute_nets'])
        paths.extend(checkpoint_path(config['model_directory'], v['checkpoint'], net) for net in v['nets'])
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing required files:\n' + '\n'.join(missing))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs' / 'smoke.json'))
    parser.add_argument('--stage', choices=['plan', 'pretrain', 'craft', 'evaluate', 'all'], default='plan')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without loading data/training')
    parser.add_argument('--json', action='store_true', help='Print the complete machine-readable plan')
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        plan = build_plan(config)
        stages = ['pretrain', 'craft', 'evaluate'] if args.stage in ('all', 'plan') else [args.stage]
        if args.json:
            print(json.dumps({stage: plan[stage] for stage in stages}, ensure_ascii=False, indent=2))
        else:
            print(config.get('name', 'MSP experiment'))
            for stage in stages:
                print(f'{stage}: {len(plan[stage])} jobs')
                commands = plan[stage] if args.dry_run else plan[stage][:1]
                for command in commands:
                    print(subprocess.list2cmdline(command) if sys.platform == 'win32' else shlex.join(command))
        if args.stage == 'plan' or args.dry_run:
            return
        for stage in stages:
            preflight(config, stage)
            for index, command in enumerate(plan[stage], 1):
                print(f'[{stage} {index}/{len(plan[stage])}]', flush=True)
                subprocess.run(command, cwd=ROOT, check=True)
    except (ValueError, KeyError, FileNotFoundError, subprocess.CalledProcessError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
