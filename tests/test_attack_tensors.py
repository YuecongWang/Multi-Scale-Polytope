"""Real PyTorch attack regressions; explicitly skipped when PyTorch is absent."""
import ast
import contextlib
import importlib.util
import inspect
import io
import os
from pathlib import Path
import random
import sys
import tempfile
import time
import unittest

from experiment_protocol import validate_attack

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from torch import nn

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(HAS_TORCH, 'PyTorch is not installed; real tensor/gradient tests were not run')
class TensorTests(unittest.TestCase):
    def setUp(self):
        names = {'extract_attack_features', 'loss_from_center', 'PoisonBatch', 'make_convex_polytope_poisons'}
        tree = ast.parse((ROOT / 'trainer.py').read_text(encoding='utf-8'))
        nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        self.ns = dict(torch=torch, nn=nn, inspect=inspect, validate_attack=validate_attack,
                       time=time, sys=sys, os=os, random=random)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'trainer.py', 'exec'), self.ns)

    def test_layerwise_is_distinct_from_energy_weighted_concatenation(self):
        class Toy(nn.Module):
            def _msp_last_three_features(self, x):
                return [x, x*2, x*3]
            def penultimate2(self, x):
                return torch.cat([part.flatten(1) for part in self._msp_last_three_features(x)], 1)
        model = Toy()
        x = torch.ones(5, 1, 2, 2, requires_grad=True)
        extract, loss_fn = self.ns['extract_attack_features'], self.ns['loss_from_center']
        centers = [part.detach() * scale for part, scale in zip(
            extract(model, x[:1], 'flat-layerwise'), (2, 5, 10))]
        loss = loss_fn([model], [centers], lambda: x, 1, False, 'flat-layerwise')
        concat = loss_fn([model], [torch.cat(centers, 1)], lambda: x, 1, False, 'flat')
        self.assertAlmostEqual(float(loss.detach()), (.5+.8+.9)/3, places=6)
        self.assertFalse(torch.allclose(loss, concat))
        gradient = torch.autograd.grad(loss, x)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0)

    def test_nondefault_epsilon_projects_pixels_and_freezes_networks(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.ones(()))
            def forward(self, x, penu=False):
                return x.flatten(1)*self.scale
        model = Toy()
        self.ns['get_CP_loss'] = lambda *args, **kwargs: (torch.tensor(0.), [], 0)
        self.ns['get_poison_tuples'] = lambda batch, label: [(image.detach().cpu(), label) for image in batch()]
        base = [torch.full((3, 2, 2), .5) for _ in range(2)]
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            result = self.ns['make_convex_polytope_poisons']([model], model, base,
                torch.ones(1, 3, 2, 2), device='cpu', mode='mean', iterations=2, epsilon=.02,
                lr=.2, mean=torch.zeros(1, 3, 1, 1), std=torch.ones(1, 3, 1, 1),
                chk_path=directory, poison_idxes=[0, 1], poison_label=8)
        self.assertLessEqual(max(float((image-base[i]).abs().max()) for i, (image, _) in enumerate(result)), .020001)
        self.assertFalse(model.scale.requires_grad)
        self.assertIsNone(model.scale.grad)


if __name__ == '__main__':
    unittest.main()
