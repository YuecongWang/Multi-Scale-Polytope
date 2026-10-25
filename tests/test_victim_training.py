"""Real CPU tensor tests for victim training, with synthetic data and no torchvision.

Run: python -m unittest discover -s tests -p test_victim_training.py -v
Production training/crafting definitions are loaded through AST so these tests do
not import trainer.py's optional image dependencies or download CIFAR-10. The
optimizers, gradients, feature cache, model weights and BatchNorm are real PyTorch.
When PyTorch is absent the tests explicitly skip; a skip is not tensor validation.
"""
import ast
from contextlib import redirect_stdout
import importlib.util
import inspect
import io
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest


HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import numpy as np
    import torch
    from torch import nn


ROOT = Path(__file__).resolve().parents[1]


def load_definitions(filename, names, namespace):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError(f"Missing production definitions in {filename}: {names}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / filename), "exec"), namespace)
    return namespace


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed; real victim tensor/gradient tests were not run")
class VictimTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(307)
        generator = torch.Generator().manual_seed(193)
        self.images = torch.rand(8, 3, 2, 2, generator=generator) * .7 + .15
        self.labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        self.test_images = torch.rand(6, 3, 2, 2, generator=generator) * .7 + .15
        self.test_labels = torch.tensor([0, 1, 0, 1, 1, 0])
        self.target = torch.full((1, 3, 2, 2), .82)
        self.feature_cache_calls = 0
        self.dataset_requests = []
        self.test_requests = []

        class TinyVictim(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Linear(12, 4, bias=False)
                self.bn = nn.BatchNorm1d(4)
                self.linear = nn.Linear(4, 2)
                self.penultimate_modes = []

            def penultimate(self, x):
                self.penultimate_modes.append(self.training)
                return torch.tanh(self.bn(self.backbone(x.flatten(1))))

            def forward(self, x, penu=False):
                features = self.penultimate(x)
                return features if penu else self.linear(features)

            def get_penultimate_params_list(self):
                return list(self.linear.parameters())

            def _msp_last_three_features(self, x):
                features = self.backbone(x.flatten(1)).unsqueeze(-1).unsqueeze(-1)
                return [features, features.square() + .2, torch.tanh(features) + .4]

        self.model_type = TinyVictim
        cache_namespace = load_definitions("dataloader.py", {"FeatureSet"}, {
            "torch": torch, "data": torch.utils.data,
        })
        production_cache = cache_namespace["FeatureSet"]

        def feature_cache(loader, model, device):
            self.feature_cache_calls += 1
            return production_cache(loader, model, device)

        def poisoned_dataset(path, **kwargs):
            self.dataset_requests.append((path, kwargs))
            images, labels = self.images.clone(), self.labels.clone()
            for (image, label), index in zip(kwargs["poison_tuple_list"], kwargs["poison_indices"]):
                images[index] = image
                labels[index] = label
            return torch.utils.data.TensorDataset(images, labels)

        def cifar10(**kwargs):
            self.test_requests.append(kwargs)
            return torch.utils.data.TensorDataset(self.test_images, self.test_labels)

        # Images are already tensors; identity transforms keep these tests focused
        # on training contracts instead of torchvision preprocessing behavior.
        transforms = SimpleNamespace(Compose=lambda steps: nn.Sequential(*steps),
                                     ToTensor=nn.Identity, Normalize=lambda *args: nn.Identity())
        self.namespace = load_definitions("trainer.py", {
            "AverageMeter", "accuracy", "train_network_with_poison",
        }, dict(torch=torch, nn=nn, time=time, sys=sys, os=os,
                transforms=transforms, PoisonedDataset=poisoned_dataset, FeatureSet=feature_cache,
                torchvision=SimpleNamespace(datasets=SimpleNamespace(CIFAR10=cifar10))))
        self.train = self.namespace["train_network_with_poison"]

    def args(self, victim_training="linear", victim_model_mode="eval", end2end=False):
        return SimpleNamespace(device="cpu", end2end=end2end, attack_end2end=False,
                               victim_training=victim_training, victim_model_mode=victim_model_mode,
                               retrain_opt="sgd", retrain_lr=.15, retrain_wd=0., retrain_momentum=0.,
                               retrain_epochs=3, retrain_bsize=4, lr_decay_epoch=[],
                               train_data_path="synthetic-split", dset_path="synthetic-test",
                               num_per_class=4, subset_group=0, target_label=0, poison_label=1)

    def run_training(self, model, args, poisons=None, indices=None):
        poisons = [] if poisons is None else poisons
        indices = [] if indices is None else indices
        with redirect_stdout(io.StringIO()):
            return self.train(model, self.target, poisons, indices, "unused", args,
                              save_state=False, return_metrics=True)

    def assert_metrics(self, model, metrics, poison_count=0):
        self.assertFalse(model.training, "Final target/test evaluation must use eval mode")
        with torch.no_grad():
            expected_prediction = int(model(self.target).argmax(1)[0])
            expected_accuracy = 100. * float((model(self.test_images).argmax(1) == self.test_labels).float().mean())
        self.assertEqual(metrics["prediction"], expected_prediction)
        self.assertAlmostEqual(metrics["clean_accuracy_percent"], expected_accuracy, places=4)
        self.assertEqual(metrics["train_samples"], len(self.images))
        self.assertEqual(metrics["poison_count"], poison_count)
        self.assertEqual(len(metrics["logits"]), 2)
        self.assertAlmostEqual(sum(metrics["probabilities"]), 1., places=6)
        self.assertEqual(metrics["trainable_parameters"],
                         sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
        self.assertTrue(self.dataset_requests[-1][1]["preserve_order"])

    def assert_backbone_delta(self, model, before, metrics, should_change):
        actual = float((model.backbone.weight.detach() - before).abs().max())
        self.assertEqual(metrics["backbone_parameter"], "backbone.weight")
        self.assertAlmostEqual(metrics["backbone_max_change"], actual, places=10)
        if should_change:
            self.assertGreater(actual, 1e-8, "The real backbone weight must update, not just requires_grad")
        else:
            self.assertEqual(actual, 0.)

    def test_linear_caches_features_and_updates_only_classifier(self):
        model = self.model_type()
        backbone = model.backbone.weight.detach().clone()
        head = model.linear.weight.detach().clone()
        mean, variance = model.bn.running_mean.clone(), model.bn.running_var.clone()
        args = self.args(victim_training="linear")
        args.attack_end2end = True  # Attack representation must not select victim training.
        metrics = self.run_training(model, args)
        self.assert_backbone_delta(model, backbone, metrics, should_change=False)
        self.assertFalse(torch.equal(model.linear.weight, head))
        self.assertTrue(torch.equal(model.bn.running_mean, mean))
        self.assertTrue(torch.equal(model.bn.running_var, variance))
        self.assertFalse(model.backbone.weight.requires_grad)
        self.assertTrue(model.linear.weight.requires_grad)
        self.assertEqual(self.feature_cache_calls, 1)
        self.assertEqual(metrics["victim_training"], "linear")
        self.assert_metrics(model, metrics)

    def test_end2end_updates_real_backbone_with_batchnorm_in_eval_mode(self):
        model = self.model_type()
        backbone = model.backbone.weight.detach().clone()
        mean, variance = model.bn.running_mean.clone(), model.bn.running_var.clone()
        metrics = self.run_training(model, self.args(victim_training="end2end", victim_model_mode="eval"))
        self.assert_backbone_delta(model, backbone, metrics, should_change=True)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        self.assertTrue(torch.equal(model.bn.running_mean, mean))
        self.assertTrue(torch.equal(model.bn.running_var, variance))
        self.assertEqual(int(model.bn.num_batches_tracked), 0)
        self.assertEqual(self.feature_cache_calls, 0)
        self.assertEqual(metrics["victim_training"], "end2end")
        self.assertEqual(metrics["victim_model_mode"], "eval")
        self.assert_metrics(model, metrics)

    def test_end2end_train_mode_updates_batchnorm_but_finishes_with_eval_metrics(self):
        model = self.model_type()
        backbone = model.backbone.weight.detach().clone()
        mean, variance = model.bn.running_mean.clone(), model.bn.running_var.clone()
        metrics = self.run_training(model, self.args(victim_training="end2end", victim_model_mode="train"))
        self.assert_backbone_delta(model, backbone, metrics, should_change=True)
        self.assertFalse(torch.equal(model.bn.running_mean, mean))
        self.assertFalse(torch.equal(model.bn.running_var, variance))
        self.assertEqual(int(model.bn.num_batches_tracked), 6)
        self.assertIn(True, model.penultimate_modes)
        self.assertEqual(metrics["victim_model_mode"], "train")
        self.assert_metrics(model, metrics)

    def test_legacy_end2end_remains_a_fallback_when_new_option_is_absent(self):
        model = self.model_type()
        backbone = model.backbone.weight.detach().clone()
        args = self.args(end2end=True)
        del args.victim_training
        metrics = self.run_training(model, args)
        self.assertEqual(metrics["victim_training"], "end2end")
        self.assert_backbone_delta(model, backbone, metrics, should_change=True)

    def test_linear_train_module_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "linear"):
            self.run_training(self.model_type(), self.args(victim_training="linear", victim_model_mode="train"))
        self.assertEqual(self.feature_cache_calls, 0)
        self.assertEqual(self.dataset_requests, [])

    def test_msp_feature_poisons_flow_into_full_parameter_victim_training(self):
        from experiment_protocol import validate_attack

        namespace = load_definitions("trainer.py", {
            "PoisonBatch", "proj_onto_simplex", "least_squares_simplex", "extract_attack_features",
            "loss_from_center", "get_CP_loss", "make_convex_polytope_poisons",
        }, dict(torch=torch, nn=nn, np=np, time=time, sys=sys, os=os, random=random,
                inspect=inspect, validate_attack=validate_attack))
        load_definitions("utils.py", {"get_poison_tuples"}, namespace)
        proxy, victim = self.model_type().eval(), self.model_type().eval()
        indices = [1]
        bases = [self.images[index].clone() for index in indices]
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            poisons = namespace["make_convex_polytope_poisons"](
                [proxy], victim, bases, self.target, device="cpu", mode="mean",
                feature_mode="flat-layerwise", end2end=False, iterations=3, lr=.02, epsilon=.05,
                mean=torch.zeros(1, 3, 1, 1), std=torch.ones(1, 3, 1, 1),
                chk_path=directory, poison_idxes=indices, poison_label=1)
            self.assertTrue((Path(directory) / "poison_00002.pth").is_file())
        delta = float((poisons[0][0] - bases[0]).abs().max())
        self.assertGreater(delta, 1e-6, "MSP input gradients must produce an actual poison perturbation")
        self.assertLessEqual(delta, .050001)
        self.assertFalse(victim.backbone.weight.requires_grad, "Crafting freezes the monitoring victim")
        before = victim.backbone.weight.detach().clone()
        metrics = self.run_training(victim, self.args(victim_training="end2end"), poisons, indices)
        self.assert_backbone_delta(victim, before, metrics, should_change=True)
        self.assertTrue(victim.backbone.weight.requires_grad, "Full training must unfreeze a monitored victim")
        self.assertEqual(self.feature_cache_calls, 0)
        self.assert_metrics(victim, metrics, poison_count=1)


if __name__ == "__main__":
    unittest.main()
