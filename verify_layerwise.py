"""CPU validation; run before launching GPU attacks."""
import ast
from pathlib import Path
import torch
from torch import nn

# Load only the two pure functions, avoiding torchvision/dataset dependencies.
tree = ast.parse(Path(__file__).with_name('trainer.py').read_text(encoding='utf-8'))
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in ('extract_attack_features', 'loss_from_center')]
namespace = {'torch': torch, 'nn': nn}
exec(compile(ast.Module(body=functions, type_ignores=[]), 'trainer.py', 'exec'), namespace)
extract = namespace['extract_attack_features']
loss_fn = namespace['loss_from_center']

class Toy(nn.Module):
    def _msp_last_three_features(self, x):
        return [x, x * 2, x * 3]
    def penultimate2(self, x):
        return torch.cat([f.flatten(1) for f in self._msp_last_three_features(x)], 1)
    def penultimate1(self, x):
        return torch.cat([f.mean((2, 3)) for f in self._msp_last_three_features(x)], 1)

net = Toy()
x = torch.ones(5, 2, 2, 2, requires_grad=True)
target = torch.full((1, 2, 2, 2), 2.)
for representation in ('flat', 'pool'):
    mode = representation + '-layerwise'
    features = extract(net, x, mode)
    assert torch.allclose(torch.cat(features, 1), extract(net, x, representation))
    centers = [f.detach() for f in extract(net, target, mode)]
    for repeats in (1, 3):
        loss = loss_fn([net], [centers], lambda: x, repeats, False, mode)
        assert torch.allclose(loss, torch.tensor(0.5))
        gradient = torch.autograd.grad(loss, x)[0]
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0

# Verify actual architectures use identical stage tensors and preserve gradients.
from models import DPN92, SENet18, ResNet50, ResNeXt29_2x64d, GoogLeNet, MobileNetV2, DenseNet121
torch.set_num_threads(1)
for factory in (DPN92, SENet18, ResNet50, ResNeXt29_2x64d, GoogLeNet, MobileNetV2, DenseNet121):
    model = factory().eval()
    sample = torch.randn(1, 3, 32, 32, requires_grad=True)
    for representation in ('flat', 'pool'):
        features = extract(model, sample, representation + '-layerwise')
        with torch.no_grad():
            reference = extract(model, sample, representation)
        assert torch.allclose(torch.cat(features, 1), reference, rtol=1e-5, atol=1e-6)
        gradient = torch.autograd.grad(sum(f.square().mean() for f in features), sample)[0]
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        print(factory.__name__, representation, [f.shape[1] for f in features], 'PASS')
print('All checks passed. GPU checkpoints/data and the full attack still require Target 0 validation.')
