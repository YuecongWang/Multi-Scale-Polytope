"""Pretrain CIFAR-10 feature extractors and resume interrupted training.

The best checkpoint uses the ``net`` field expected by the poison-crafting and
evaluation scripts. A separate ``*-latest`` checkpoint also retains optimizer,
scheduler and random-number-generator state for continuing the next epoch.
"""

import argparse
import inspect
import os
from pathlib import Path
import random
import time
import warnings


MODEL_NAMES = (
    'DenseNet105', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'DenseNet161',
    'DPN26', 'DPN92', 'GoogLeNet', 'MobileNetV2', 'ResNet18', 'ResNet34',
    'ResNet50', 'ResNet101', 'ResNet152', 'ResNeXt29_2x64d',
    'ResNeXt29_4x64d', 'ResNeXt29_8x64d', 'ResNeXt29_32x4d', 'SENet18',
)


def build_parser():
    """Build the CLI without importing the optional training dependencies."""
    parser = argparse.ArgumentParser(description='Pretrain CIFAR-10 feature extractors')
    parser.add_argument('--lr', default=0.1, type=float, help='initial SGD learning rate')
    parser.add_argument('--resume', '-r', default='', help='checkpoint to continue or evaluate')
    parser.add_argument('--evaluate-only', action='store_true',
                        help='evaluate --resume on the official CIFAR-10 test set and exit')
    parser.add_argument('--net', default='DenseNet121', choices=MODEL_NAMES)
    parser.add_argument('--gpu', default='0', help='CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1')
    parser.add_argument('--device', default='auto',
                        help='auto, cpu, cuda, or cuda:N (after applying --gpu)')
    parser.add_argument('--train-dp', default=0.0, type=float)
    parser.add_argument('--droplayer', default=0.0, type=float)
    parser.add_argument('--batchsize', '--batch-size', default=128, type=int)
    parser.add_argument('--seed', default=1226, type=int)
    parser.add_argument('--sidx', default=0, type=int,
                        help='inclusive per-class start index in clean_train')
    parser.add_argument('--eidx', default=4800, type=int,
                        help='exclusive per-class end index in clean_train')
    parser.add_argument('--epochs', default=300, type=int,
                        help='total epochs, including epochs already completed by --resume')
    parser.add_argument('--milestones', default=[150, 225], nargs='*', type=int,
                        help='epoch indices at which SGD learning rate is multiplied by --gamma')
    parser.add_argument('--gamma', '--lr-decay', default=0.1, type=float)
    parser.add_argument('--workers', default=0 if os.name == 'nt' else 2, type=int,
                        help='data-loader workers; defaults to 0 on Windows and 2 elsewhere')
    parser.add_argument('--data-path', default='datasets/CIFAR10_TRAIN_Split.pth',
                        help='split file containing the clean_train image/label list')
    parser.add_argument('--dset-path', default='datasets',
                        help='directory containing the official CIFAR-10 test dataset')
    parser.add_argument('--no-download', action='store_true',
                        help='require the official CIFAR-10 test dataset to exist locally')
    parser.add_argument('--output-dir', default='model-chks', help='checkpoint output directory')
    parser.add_argument('--checkpoint-name', default='ckpt-{net}-{eidx}.t7',
                        help='best-checkpoint filename; supports {net}/{eidx}/{sidx}/{seed} '
                             'or the legacy model placeholder %%s')
    return parser


def checkpoint_paths(args):
    """Resolve one best filename and the corresponding latest filename."""
    name = args.checkpoint_name.format(**vars(args))
    if '%s' in name:
        name = name % args.net
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('--checkpoint-name must be a filename inside --output-dir')
    best = Path(args.output_dir) / name
    latest = best.with_name(best.stem + '-latest' + best.suffix)
    return best, latest


def _model_factory(name, train_dp, droplayer):
    # Explicit imports avoid eval() and names leaked by wildcard imports.
    from models.densenet_msp import DenseNet105, DenseNet121, DenseNet169, DenseNet201, DenseNet161
    from models.dpn_msp import DPN26, DPN92
    from models.googlenet_msp import GoogLeNet
    from models.mobilenetv2_msp import MobileNetV2
    from models.resnet_msp import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
    from models.resnext_msp import ResNeXt29_2x64d, ResNeXt29_4x64d, ResNeXt29_8x64d, ResNeXt29_32x4d
    from models.senet_msp import SENet18

    constructors = {
        constructor.__name__: constructor for constructor in (
            DenseNet105, DenseNet121, DenseNet169, DenseNet201, DenseNet161,
            DPN26, DPN92, GoogLeNet, MobileNetV2, ResNet18, ResNet34, ResNet50,
            ResNet101, ResNet152, ResNeXt29_2x64d, ResNeXt29_4x64d,
            ResNeXt29_8x64d, ResNeXt29_32x4d, SENet18,
        )
    }
    constructor = constructors[name]
    parameters = inspect.signature(constructor).parameters
    kwargs = {}
    for key, value in (('train_dp', train_dp), ('droplayer', droplayer)):
        if key in parameters:
            kwargs[key] = value
        elif value != 0:
            raise ValueError('{} does not support --{}'.format(name, key.replace('_', '-')))
    return constructor(**kwargs)


def _load(torch, path):
    # Split files contain PIL images, and training checkpoints contain NumPy RNG
    # state; both are trusted local project artifacts rather than weights alone.
    kwargs = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False
    return torch.load(path, **kwargs)


def seed_worker(worker_id):
    """Top-level function so Windows multiprocessing can pickle it."""
    import numpy as np
    import torch

    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def train_one_epoch(net, loader, optimizer, criterion, device, epoch):
    net.train()
    loss_sum, correct, total = 0.0, 0, 0
    for batch_index, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total += targets.size(0)
        loss_sum += loss.item() * targets.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        if batch_index % 100 == 0:
            print('{} Epoch {}, iteration {}, Loss: {:.3f} | Acc: {:.3f}'.format(
                time.strftime('%Y-%m-%d %H:%M:%S'), epoch, batch_index,
                loss_sum / total, 100.0 * correct / total))
    return {'loss': loss_sum / total, 'accuracy': 100.0 * correct / total}


def evaluate(net, loader, criterion, device, epoch):
    import torch

    net.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            total += targets.size(0)
            loss_sum += criterion(outputs, targets).item() * targets.size(0)
            correct += outputs.argmax(1).eq(targets).sum().item()
    metrics = {'loss': loss_sum / total, 'accuracy': 100.0 * correct / total}
    print('* CIFAR-10 test, epoch {}, Loss: {:.3f} | Acc: {:.3f}'.format(
        epoch, metrics['loss'], metrics['accuracy']))
    return metrics


def _rng_state(torch, np, train_generator):
    state = {
        'python': random.getstate(), 'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(), 'train_loader': train_generator.get_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(torch, np, state, train_generator):
    if not state:
        warnings.warn('Checkpoint has no RNG state; exact continuation is unavailable.')
        return
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    train_generator.set_state(state['train_loader'])
    if 'cuda' in state and torch.cuda.is_available():
        if len(state['cuda']) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(state['cuda'])
        else:
            warnings.warn('CUDA device count changed; CUDA RNG state was not restored.')


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.evaluate_only and not args.resume:
        parser.error('--evaluate-only requires --resume')
    if args.sidx < 0 or args.eidx <= args.sidx:
        parser.error('require 0 <= --sidx < --eidx')
    if args.epochs < 1 or args.batchsize < 1 or args.workers < 0:
        parser.error('--epochs and --batchsize must be positive; --workers must be nonnegative')
    if args.lr <= 0 or not 0 < args.gamma <= 1:
        parser.error('--lr must be positive and --gamma must be in (0, 1]')
    if any(value < 1 for value in args.milestones) or args.milestones != sorted(set(args.milestones)):
        parser.error('--milestones must contain strictly increasing positive epoch indices')
    if not 0 <= args.train_dp < 1 or not 0 <= args.droplayer < 1:
        parser.error('--train-dp and --droplayer must be in [0, 1)')
    try:
        best_path, latest_path = checkpoint_paths(args)
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    if args.resume and not Path(args.resume).is_file():
        parser.error('resume checkpoint does not exist: {}'.format(args.resume))
    if not args.evaluate_only and not Path(args.data_path).is_file():
        parser.error('training split does not exist: {}'.format(args.data_path))

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import numpy as np
    import torch
    from torch import nn
    from torch.optim import SGD
    from torch.optim.lr_scheduler import MultiStepLR
    from torch.utils.data import DataLoader
    from torchvision.datasets import CIFAR10
    from torchvision import transforms
    from dataloader import SubsetOfList

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        try:
            device = torch.device(args.device)
        except (RuntimeError, ValueError) as exc:
            parser.error(str(exc))
    if device.type not in ('cuda', 'cpu'):
        parser.error('--device must be auto, cpu, cuda, or cuda:N')
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA was requested but is unavailable; use --device cpu')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    train_generator = torch.Generator().manual_seed(args.seed)
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    testset = CIFAR10(root=args.dset_path, train=False,
                     download=not args.no_download, transform=test_transform)
    testloader = DataLoader(testset, batch_size=1000, shuffle=False,
                            num_workers=args.workers, worker_init_fn=seed_worker,
                            pin_memory=device.type == 'cuda')

    net = _model_factory(args.net, args.train_dp, args.droplayer).to(device)
    if device.type == 'cuda' and device.index in (None, 0) and torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    model = net.module if isinstance(net, nn.DataParallel) else net
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    start_epoch, best_acc = 0, float('-inf')
    if args.resume:
        checkpoint = _load(torch, args.resume)
        state_dict = checkpoint['net']
        # Accept original DataParallel checkpoints and ordinary state dicts.
        state_dict = {key[7:] if key.startswith('module.') else key: value
                      for key, value in state_dict.items()}
        model.load_state_dict(state_dict)
        start_epoch = checkpoint.get('next_epoch', checkpoint.get('epoch', -1) + 1)
        best_acc = checkpoint.get('best_acc', checkpoint.get('acc', best_acc))
        if args.evaluate_only:
            return evaluate(net, testloader, criterion, device, checkpoint.get('epoch', -1))
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            warnings.warn('Legacy checkpoint has no optimizer state; SGD momentum starts fresh.')
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            # Old checkpoints saved only weights/epoch/accuracy. Reconstruct the
            # next epoch's learning rate while preserving the original schedule.
            decay_count = sum(milestone <= start_epoch for milestone in args.milestones)
            for group in optimizer.param_groups:
                group['lr'] = args.lr * args.gamma ** decay_count
            scheduler.last_epoch = start_epoch
            scheduler._last_lr = [group['lr'] for group in optimizer.param_groups]
        _restore_rng(torch, np, checkpoint.get('rng_state'), train_generator)
        print('Resuming at epoch {} of {} (best test accuracy {:.3f}).'.format(
            start_epoch, args.epochs, best_acc))
    if start_epoch >= args.epochs:
        parser.error('checkpoint has completed {} epochs; increase --epochs to continue'.format(start_epoch))

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize,
    ])
    split = _load(torch, args.data_path)
    trainset = SubsetOfList(split['clean_train'], transform=train_transform,
                            start_idx=args.sidx, end_idx=args.eidx)
    expected = 10 * (args.eidx - args.sidx)
    if len(trainset) != expected:
        parser.error('split selection contains {} images; expected {} for indices [{}, {}) per class'.format(
            len(trainset), expected, args.sidx, args.eidx))
    trainloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True,
                            num_workers=args.workers, worker_init_fn=seed_worker,
                            generator=train_generator, pin_memory=device.type == 'cuda')
    best_path.parent.mkdir(parents=True, exist_ok=True)
    print('Training {} on {}: {} images, epochs [{}, {}), best checkpoint {}.'.format(
        args.net, device, len(trainset), start_epoch, args.epochs, best_path))
    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_one_epoch(net, trainloader, optimizer, criterion, device, epoch)
        test_metrics = evaluate(net, testloader, criterion, device, epoch)
        scheduler.step()
        improved = test_metrics['accuracy'] > best_acc
        best_acc = max(best_acc, test_metrics['accuracy'])
        state = {
            'net': model.state_dict(), 'acc': test_metrics['accuracy'],
            'best_acc': best_acc, 'epoch': epoch, 'next_epoch': epoch + 1,
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'rng_state': _rng_state(torch, np, train_generator),
            'config': dict(vars(args)), 'train_metrics': train_metrics,
            'test_metrics': test_metrics, 'format_version': 2,
        }
        if improved:
            torch.save(state, best_path)
        torch.save(state, latest_path)
    return {'best_accuracy': best_acc, 'checkpoint': str(best_path), 'latest': str(latest_path)}


if __name__ == '__main__':
    main()
