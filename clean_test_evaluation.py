"""Measure pretrained CIFAR-10 checkpoints; does not train or generate poisons."""
import argparse
import os
from experiment_recorder import save_json


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nets', '--target-net', nargs='+', default=['ResNet18'])
    p.add_argument('--model-resume-path', default='model-chks')
    p.add_argument('--test-chk-name', default='ckpt-%s-4800.t7')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--gpu', default='0')
    p.add_argument('--dset-path', default='datasets')
    p.add_argument('--batch-size', type=int, default=500)
    p.add_argument('--download', action='store_true')
    p.add_argument('--output', default='runs/pretrained-clean-evaluation.json')
    a = p.parse_args(argv)
    if a.batch_size < 1:
        p.error('batch-size must be positive')
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    import torch
    from torchvision import transforms
    from torchvision.datasets import CIFAR10
    from utils import load_pretrained_net
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        (.4914, .4822, .4465), (.2023, .1994, .2010))])
    data = CIFAR10(a.dset_path, train=False, download=a.download, transform=transform)
    loader = torch.utils.data.DataLoader(data, batch_size=a.batch_size, shuffle=False)
    results = []
    with torch.no_grad():
        for name in a.nets:
            net = load_pretrained_net(name, a.test_chk_name, a.model_resume_path, device=a.device)
            correct, total = 0, 0
            for images, labels in loader:
                predictions = net(images.to(a.device)).argmax(1).cpu()
                correct += int((predictions == labels).sum())
                total += len(labels)
            result = dict(model=name, correct=correct, samples=total, accuracy_percent=100*correct/total)
            print(result)
            results.append(result)
            del net
    save_json(a.output, dict(checkpoint_template=a.test_chk_name, device=a.device, results=results))


if __name__ == '__main__':
    main()
