"""Single-target CP/BP/MSP experiments: craft, then paired clean/poisoned evaluation."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import random
import sys
import time

from experiment_protocol import (checkpoint_path, ensure_protocol, file_identity,
                                 method_name, parse_bool, resolve_training_options, validate_attack)
from experiment_recorder import save_json


class Logger:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['craft', 'evaluate', 'all'], default='all')
    p.add_argument('--gpu', default='0')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--end2end', type=parse_bool, nargs='?', const=True, default=None,
                   help='Legacy shorthand; prefer independent attack/victim options')
    p.add_argument('--attack-representation', choices=['features', 'blocks'], default=None,
                   help='Feature-mode objective or the original BP/CP block objective')
    p.add_argument('--victim-training', choices=['linear', 'end2end'], default=None,
                   help='Train the classifier head or all victim parameters')
    p.add_argument('--victim-model-mode', choices=['eval', 'train'], default='eval',
                   help='Victim module mode during optimization; independent of parameter selection')
    p.add_argument('--substitute-nets', nargs='+', default=['ResNet50', 'ResNet18'])
    p.add_argument('--target-net', nargs='+', default=['DenseNet121'])
    p.add_argument('--model-resume-path', default='model-chks')
    p.add_argument('--subs-chk-name', nargs='+', default=['ckpt-%s-4800.t7'])
    p.add_argument('--test-chk-name', default='ckpt-%s-4800.t7')
    p.add_argument('--subs-dp', nargs='+', type=float, default=[0.0])
    p.add_argument('--net-repeat', type=int, default=1)
    p.add_argument('--target-dset', choices=['cifar10', '102flowers'], default='cifar10')
    p.add_argument('--target-label', type=int, default=6)
    p.add_argument('--target-index', type=int, default=0)
    p.add_argument('--target-start', type=int, default=50,
                   help='Per-class offset in others, outside the victim training window')
    p.add_argument('--poison-label', '-plabel', type=int, default=8)
    p.add_argument('--poison-num', type=int, default=5)
    p.add_argument('--poison-lr', '-plr', type=float, default=0.04)
    p.add_argument('--poison-momentum', '-pm', type=float, default=0.9)
    p.add_argument('--poison-ites', type=int, default=4000)
    p.add_argument('--poison-decay-ites', nargs='*', type=int, default=[])
    p.add_argument('--poison-decay-ratio', type=float, default=0.1)
    p.add_argument('--poison-epsilon', '-peps', type=float, default=0.1)
    p.add_argument('--poison-opt', choices=['adam', 'sgd', 'signedadam'], default='adam')
    p.add_argument('--nearest', action='store_true')
    p.add_argument('--subset-group', type=int, default=0)
    p.add_argument('--original-grad', type=parse_bool, default=True,
                   help='Deprecated compatibility option; attack uses autograd')
    p.add_argument('--tol', type=float, default=1e-6)
    p.add_argument('--retrain-lr', '-rlr', type=float, default=0.1)
    p.add_argument('--retrain-opt', choices=['adam', 'sgd'], default='adam')
    p.add_argument('--retrain-momentum', '-rm', type=float, default=0.9)
    p.add_argument('--lr-decay-epoch', nargs='*', type=int, default=[30, 45])
    p.add_argument('--retrain-epochs', type=int, default=60)
    p.add_argument('--retrain-bsize', type=int, default=64)
    p.add_argument('--retrain-wd', type=float, default=0.0)
    p.add_argument('--num-per-class', type=int, default=50)
    p.add_argument('--chk-path', default='runs/attack')
    p.add_argument('--chk-subdir', default='poisons', help='Deprecated output-layout option')
    p.add_argument('--eval-poison-path', default='', help='Explicit historical poison checkpoint')
    p.add_argument('--resume-poison-ite', type=int, default=None,
                   help='Saved zero-based iteration; warm start with a fresh optimizer')
    p.add_argument('--train-data-path', default='datasets/CIFAR10_TRAIN_Split.pth')
    p.add_argument('--dset-path', default='datasets')
    p.add_argument('--mode', default='convex')
    p.add_argument('--feature-mode', choices=['original', 'pool', 'flat', 'flat-layerwise', 'pool-layerwise'],
                   default='original')
    p.add_argument('--seed', type=int, default=2026)
    return p


def validate_args(a):
    options = resolve_training_options(a.end2end, a.feature_mode, a.attack_representation,
                                       a.victim_training, a.victim_model_mode)
    for name, value in options.items():
        setattr(a, name, value)
    validate_attack(a.mode, a.feature_mode, a.attack_end2end, a.net_repeat, a.poison_num)
    if len(a.subs_chk_name) != len(a.subs_dp):
        raise ValueError('Each substitute checkpoint template needs one --subs-dp value')
    if any(not 0 <= dp < 1 for dp in a.subs_dp):
        raise ValueError('Dropout must be within [0, 1)')
    if min(a.target_index, a.target_start, a.subset_group) < 0:
        raise ValueError('Target index/offset and subset group must be nonnegative')
    if not 0 <= a.target_label < 10 or not 0 <= a.poison_label < 10 or a.target_label == a.poison_label:
        raise ValueError('Choose different target/poison labels within 0..9')
    if not 0 <= a.poison_epsilon <= 1:
        raise ValueError('poison-epsilon is a pixel-space budget within [0, 1]')
    if min(a.poison_ites, a.retrain_epochs, a.retrain_bsize, a.num_per_class) < 1:
        raise ValueError('Iterations, epochs, batch size and per-class count must be positive')
    if a.poison_num > a.num_per_class:
        raise ValueError('Poison count exceeds the victim per-class training budget')
    if min(a.poison_lr, a.retrain_lr, a.tol) <= 0 or a.retrain_wd < 0:
        raise ValueError('Learning rates/tolerance must be positive and weight decay nonnegative')
    if a.attack_end2end:
        unsupported = [name for name in a.substitute_nets
                       if not name.startswith(('ResNet', 'ResNeXt', 'DPN', 'SENet'))]
        if unsupported:
            raise ValueError('Original E2E block interface unavailable for: ' + ', '.join(unsupported))
    position = a.target_start + a.target_index
    if (a.target_dset == 'cifar10' and a.subset_group * a.num_per_class <= position
            < (a.subset_group + 1) * a.num_per_class):
        raise ValueError('Target is inside victim training subset; change --target-start/subset-group')
    if a.resume_poison_ite is not None and not 0 <= a.resume_poison_ite < a.poison_ites - 1:
        raise ValueError('Resume iteration must precede the final iteration')


def run(a):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    import numpy as np
    import torch
    from torchvision import transforms
    from utils import load_pretrained_net, fetch_target, fetch_nearest_poison_bases, fetch_poison_bases
    from trainer import make_convex_polytope_poisons, train_network_with_poison

    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu for a small smoke run')

    def seed():
        random.seed(a.seed)
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(a.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    seed()
    objective = a.mode + (f'-{a.net_repeat}Repeat' if a.net_repeat > 1 else '')
    output = Path(a.chk_path) / a.feature_mode / objective / str(a.poison_ites) / str(a.target_index)
    final_path = output / f'poison_{a.poison_ites - 1:05d}.pth'
    parameters = {key: value for key, value in vars(a).items()
                  if key not in {'stage', 'resume_poison_ite', 'eval_poison_path', 'chk_path', 'chk_subdir'}}
    protocol = dict(schema_version=2, method=method_name(a.mode, a.feature_mode, a.attack_end2end),
        parameters=parameters, data=file_identity(a.train_data_path),
        substitutes=[file_identity(checkpoint_path(a.model_resume_path, template, name))
                     for template in a.subs_chk_name for name in a.substitute_nets],
        victims=[file_identity(checkpoint_path(a.model_resume_path, a.test_chk_name, name))
                 for name in a.target_net],
        attack_representation=a.attack_representation, victim_training=a.victim_training,
        victim_model_mode=a.victim_model_mode,
        training=('all parameters' if a.end2end else 'classifier head only') + ' in ' + a.victim_model_mode + ' mode',
        overlap='unknown unless established by checkpoint training provenance')
    if a.eval_poison_path:
        protocol['imported_poison'] = file_identity(a.eval_poison_path)
    poison_path = Path(a.eval_poison_path) if a.eval_poison_path else final_path
    if a.stage == 'evaluate' and not poison_path.is_file():
        raise FileNotFoundError(poison_path)
    ensure_protocol(output / 'protocol.json', protocol)
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if a.target_dset == 'cifar10':
        target = fetch_target(a.target_label, a.target_index, a.target_start,
                              path=a.train_data_path, subset='others', transforms=transform)
    else:
        from utils import fetch_target_102flower_dset
        target = fetch_target_102flower_dset(a.target_index, transform)

    with (output / 'log.txt').open('a', encoding='utf-8', buffering=1) as log:
        with contextlib.redirect_stdout(Logger(sys.stdout, log)):
            print(json.dumps(protocol, ensure_ascii=False, indent=2))
            started = time.perf_counter()
            if a.stage in ('craft', 'all') and not final_path.exists():
                if a.device == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                proxies = [load_pretrained_net(name, template, a.model_resume_path, test_dp=dp, device=a.device)
                           for template, dp in zip(a.subs_chk_name, a.subs_dp)
                           for name in a.substitute_nets]
                diagnostic = load_pretrained_net(a.target_net[0], a.test_chk_name, a.model_resume_path, device=a.device)
                offset = a.subset_group * a.num_per_class
                if a.nearest:
                    bases, indices = fetch_nearest_poison_bases(proxies, target, a.poison_num, a.poison_label,
                        a.num_per_class, 'others', a.train_data_path, transform, start_idx=offset, device=a.device)
                else:
                    bases, indices = fetch_poison_bases(a.poison_label, a.poison_num, subset='others',
                        path=a.train_data_path, transforms=transform, start_idx=offset)
                bases = [base.to(a.device) for base in bases]
                initial, start = bases, 0
                if a.resume_poison_ite is not None:
                    state = torch.load(output / f'poison_{a.resume_poison_ite:05d}.pth',
                                       map_location=a.device, weights_only=False)
                    if state['idx'] != indices:
                        raise ValueError('Resume base indices differ from protocol')
                    initial = [image.to(a.device) for image, _ in state['poison']]
                    start = a.resume_poison_ite + 1
                    print('Warm start: optimizer/dropout restart; not a bitwise continuation')
                make_convex_polytope_poisons(proxies, diagnostic, bases, target,
                    device=a.device, opt_method=a.poison_opt, lr=a.poison_lr,
                    momentum=a.poison_momentum, iterations=a.poison_ites, epsilon=a.poison_epsilon,
                    decay_ites=a.poison_decay_ites, decay_ratio=a.poison_decay_ratio,
                    mean=torch.tensor(mean).view(1, 3, 1, 1), std=torch.tensor(std).view(1, 3, 1, 1),
                    chk_path=str(output), poison_idxes=indices, poison_label=a.poison_label,
                    tol=a.tol, end2end=a.attack_end2end, start_ite=start, poison_init=initial,
                    mode=a.mode, net_repeat=a.net_repeat, feature_mode=a.feature_mode)
                save_json(output / 'craft.json', dict(seconds=time.perf_counter()-started,
                    timing_scope='model loading, target features, optimization, diagnostics and checkpoints',
                    peak_allocated_bytes=torch.cuda.max_memory_allocated() if a.device == 'cuda' else None,
                    warm_start_iteration=a.resume_poison_ite, iterations=a.poison_ites,
                    method=protocol['method'], attack_representation=a.attack_representation,
                    victim_training=a.victim_training, final_checkpoint=str(final_path)))
                del proxies, diagnostic
                if a.device == 'cuda':
                    torch.cuda.empty_cache()
            if a.stage == 'craft':
                print('Final poison checkpoint:', final_path)
                return
            state = torch.load(poison_path, map_location='cpu', weights_only=False)
            poisons, indices = state['poison'], state['idx']
            if len(poisons) != a.poison_num or any(int(label) != a.poison_label for _, label in poisons):
                raise ValueError('Saved poison count/labels differ from requested evaluation')
            # Historical checkpoint end2end described attack blocks, not victim parameter updates.
            for key, expected in (('feature_mode', a.feature_mode), ('mode', a.mode),
                                  ('end2end', a.attack_end2end), ('attack_representation', a.attack_representation)):
                if key in state and state[key] != expected:
                    raise ValueError('Saved poison method metadata differs: ' + key)
            raw = torch.load(a.train_data_path, map_location='cpu', weights_only=False)['others']
            if len(indices) != len(poisons) or len(set(indices)) != len(indices):
                raise ValueError('Each poison must replace one unique base')
            actual_linf = 0.0
            for (image, label), index in zip(poisons, indices):
                if not 0 <= index < len(raw) or int(raw[index][1]) != label:
                    raise ValueError('Saved poison base index/label is invalid')
                base = transform(raw[index][0])
                if image.shape != base.shape or not torch.isfinite(image).all():
                    raise ValueError('Saved poison has invalid shape or nonfinite pixels')
                deviation = float(((image-base) * torch.tensor(std).view(3, 1, 1)).abs().max())
                pixels = image * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
                if deviation > a.poison_epsilon + 1e-6 or pixels.min() < -1e-6 or pixels.max() > 1+1e-6:
                    raise ValueError('Saved poison violates the requested pixel-space budget/range')
                actual_linf = max(actual_linf, deviation)
            results = []
            for name in a.target_net:
                measurements = {}
                for condition in ('clean', 'poisoned'):
                    net = load_pretrained_net(name, a.test_chk_name, a.model_resume_path, device=a.device)
                    seed()  # independent of poison generation, paired victim-training RNG
                    measured_start = time.perf_counter()
                    measurements[condition] = train_network_with_poison(net, target,
                        poisons if condition == 'poisoned' else [], indices if condition == 'poisoned' else [],
                        str(output), a, save_state=False, return_metrics=True)
                    measurements[condition]['seconds'] = time.perf_counter()-measured_start
                    del net
                eligible = measurements['clean']['prediction'] == a.target_label
                success = measurements['poisoned']['prediction'] == a.poison_label
                result = dict(victim=name, seed=a.seed, target_index=a.target_index,
                    victim_training=a.victim_training, victim_model_mode=a.victim_model_mode,
                    target_label=a.target_label, poison_label=a.poison_label,
                    clean_target_correct=eligible, attack_success=success,
                    eligible_attack_success=success if eligible else None, **measurements)
                save_json(output / f'evaluation-{name}-seed{a.seed}.json', result)
                results.append(result)
            save_json(output / 'evaluation.json',
                      dict(protocol=protocol, poison=file_identity(poison_path),
                           base_indices=indices, actual_linf=actual_linf,
                           craft=json.loads((output / 'craft.json').read_text(encoding='utf-8'))
                                 if (output / 'craft.json').exists() else None,
                           results=results))
            print('------SUMMARY------')
            print('TIME ELAPSED (mins):', (time.perf_counter() - started) / 60)
            print('TARGET INDEX:', a.target_index)
            for result in results:
                print(result['victim'], int(result['attack_success']))


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.eval_poison_path:
        args.stage = 'evaluate'
    try:
        validate_args(args)
    except ValueError as error:
        parser.error(str(error))
    run(args)


if __name__ == '__main__':
    main()
