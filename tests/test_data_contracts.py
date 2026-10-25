"""Dependency-free data/checkpoint contract tests using AST and small mocks.

Run: python -m unittest discover -s tests -p test_data_contracts.py -v
These tests execute the production Python selection/loading logic. They do not
exercise PyTorch tensors, model forward passes, autograd, CUDA or real files.
Run verify_layerwise.py separately in a configured PyTorch environment.
"""
import ast
from collections import Counter
from contextlib import redirect_stdout
import inspect
import io
import operator
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]


def load_definitions(filename, names, namespace):
    """Use real definitions while avoiding optional ML dependency imports."""
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    found = {node.name for node in nodes}
    if found != set(names):
        raise AssertionError(f"Missing production definitions: {set(names) - found}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / filename), "exec"), namespace)
    return namespace


class DatasetContracts(unittest.TestCase):
    def setUp(self):
        self.samples = [(f"image-{ordinal}-{label}", label)
                        for ordinal in range(6) for label in (0, 1)]
        self.samples.append(("excluded-class", 2))
        self.load = Mock(return_value={"others": self.samples})
        namespace = load_definitions("dataloader.py", {"PoisonedDataset"}, {
            "torch": SimpleNamespace(load=self.load),
            "data": SimpleNamespace(Dataset=object),
            "Counter": Counter, "operator": operator,
        })
        self.dataset_type = namespace["PoisonedDataset"]

    def dataset(self, **kwargs):
        options = dict(subset="others", class_labels=[0, 1])
        options.update(kwargs)
        return self.dataset_type("trusted-local-split.pth", **options)

    def test_budget_replaces_bases_and_preserves_class_counts(self):
        dataset = self.dataset(num_per_label=3, poison_tuple_list=[("poison", 1)], poison_indices=[1])
        self.assertEqual(len(dataset), 6)
        self.assertEqual(dataset.class_counts, {0: 3, 1: 3})
        self.assertEqual(dataset.valid_indices, [0, 2, 3, 4, 5])
        self.assertEqual(dataset[0], ("poison", 1))
        self.load.assert_called_once_with("trusted-local-split.pth", map_location="cpu", weights_only=False)

    def test_whole_split_still_replaces_bases_and_filters_classes(self):
        dataset = self.dataset(poison_tuple_list=[("poison", 1)], poison_indices=[1])
        self.assertEqual(len(dataset), 12)
        self.assertNotIn(1, dataset.valid_indices)
        self.assertNotIn(12, dataset.valid_indices)
        self.assertEqual(dataset.class_counts, {0: 6, 1: 6})

    def test_group_one_uses_next_class_window(self):
        dataset = self.dataset(num_per_label=3, subset_group=1,
                               poison_tuple_list=[("poison", 1)], poison_indices=[7])
        self.assertEqual(len(dataset), 6)
        self.assertEqual(dataset.valid_indices, [6, 8, 9, 10, 11])
        self.assertEqual(dataset.class_counts, {0: 3, 1: 3})

    def test_clean_transform_does_not_transform_saved_poison_again(self):
        dataset = self.dataset(num_per_label=3, transform=lambda image: f"transformed:{image}",
                               poison_tuple_list=[("poison", 1)], poison_indices=[1])
        self.assertEqual(dataset[0], ("poison", 1))
        self.assertEqual(dataset[1], ("transformed:image-0-0", 0))

    def test_preserve_order_replaces_base_at_its_original_position(self):
        dataset = self.dataset(num_per_label=3, preserve_order=True,
                               poison_tuple_list=[("poison", 1)], poison_indices=[1])
        expected = self.samples[:6].copy()
        expected[1] = ("poison", 1)
        self.assertEqual([dataset[index] for index in range(len(dataset))], expected)
        self.assertEqual(dataset.class_counts, {0: 3, 1: 3})

    def test_preserve_order_zero_perturbation_matches_clean_dataset_itemwise(self):
        normalize = lambda image: f"normalized:{image}"
        clean = self.dataset(num_per_label=3, preserve_order=True, transform=normalize)
        # Saved poisons are already transformed. At epsilon=0 they represent the
        # exact normalized bases, with matching labels and original positions.
        indices = [1, 4]
        poisons = [(normalize(self.samples[index][0]), self.samples[index][1]) for index in indices]
        poisoned = self.dataset(num_per_label=3, preserve_order=True, transform=normalize,
                                poison_tuple_list=poisons, poison_indices=indices)
        self.assertEqual(len(clean), len(poisoned))
        self.assertEqual([clean[index] for index in range(len(clean))],
                         [poisoned[index] for index in range(len(poisoned))])

    def test_invalid_labels_and_indices_fail_before_training(self):
        invalid = {
            "duplicate_bases": dict(poison_tuple_list=[("a", 1), ("b", 1)], poison_indices=[1, 1]),
            "missing_base": dict(poison_tuple_list=[("a", 1)], poison_indices=[]),
            "changed_label": dict(poison_tuple_list=[("a", 0)], poison_indices=[1]),
            "negative_index": dict(poison_tuple_list=[("a", 1)], poison_indices=[-1]),
            "out_of_range": dict(poison_tuple_list=[("a", 1)], poison_indices=[13]),
            "fractional_index": dict(poison_tuple_list=[("a", 1)], poison_indices=[1.5]),
            "excluded_poison_class": dict(poison_tuple_list=[("a", 2)], poison_indices=[12]),
            "malformed_poison": dict(poison_tuple_list=[("a",)], poison_indices=[1]),
        }
        for name, options in invalid.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.dataset(**options)

    def test_insufficient_or_inconsistent_class_budgets_fail(self):
        invalid = {
            "insufficient_samples": dict(num_per_label=7),
            "bases_outside_group": dict(num_per_label=3, subset_group=1,
                                        poison_tuple_list=[("a", 1)], poison_indices=[1]),
            "too_many_poisons": dict(num_per_label=1,
                                     poison_tuple_list=[("a", 1), ("b", 1)], poison_indices=[1, 3]),
            "group_without_budget": dict(num_per_label=-1, subset_group=1),
            "negative_group": dict(num_per_label=3, subset_group=-1),
            "duplicate_classes": dict(class_labels=[0, 0]),
        }
        for name, options in invalid.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.dataset(**options)


class BaseSelectionContracts(unittest.TestCase):
    def setUp(self):
        self.load = Mock(return_value={"others": [
            (f"image-{ordinal}-{label}", label)
            for ordinal in range(6) for label in (0, 1)
        ]})
        self.namespace = load_definitions("utils.py", {
            "_nonnegative_index", "_load_split", "fetch_poison_bases", "fetch_all_target_cls",
        }, {"torch": SimpleNamespace(load=self.load, stack=tuple), "operator": operator})

    def test_base_offset_is_per_class_but_indices_are_split_relative(self):
        images, indices = self.namespace["fetch_poison_bases"](
            1, 2, "others", "trusted.pth", str, start_idx=3)
        self.assertEqual(indices, [7, 9])
        self.assertEqual(images, ["image-3-1", "image-4-1"])
        self.load.assert_called_once_with("trusted.pth", map_location="cpu", weights_only=False)

    def test_nearest_candidate_window_respects_same_offset(self):
        images, indices = self.namespace["fetch_all_target_cls"](
            1, 3, "others", "trusted.pth", str, start_idx=3)
        self.assertEqual(indices, [7, 9, 11])
        self.assertEqual(len(images), 3)

    def test_candidate_shortage_and_invalid_count_are_rejected(self):
        for name in ("fetch_poison_bases", "fetch_all_target_cls"):
            for count, offset in ((4, 3), (0, 0), (1, -1)):
                with self.subTest(function=name, count=count, offset=offset), self.assertRaises(ValueError):
                    self.namespace[name](1, count, "others", "trusted.pth", str, start_idx=offset)


class CheckpointContracts(unittest.TestCase):
    def setUp(self):
        class Device:
            def __init__(self, value):
                self.type, _, index = str(value).partition(":")
                self.index = int(index) if index else None

        class Module:
            def __init__(self):
                self.training = True

            def eval(self):
                self.training = False
                return self

        class DataParallel(Module):
            def __init__(self, model, **kwargs):
                super().__init__()
                self.module = model
                self.device_ids = kwargs.get("device_ids")
                self.output_device = kwargs.get("output_device")

        self.model = SimpleNamespace(load_state_dict=Mock(), to=Mock())

        def factory(test_dp=0, bdp=0):
            return self.model

        self.load = Mock(return_value={"net": {"module.linear.weight": "mock-weight"}})
        self.cuda_available = Mock(return_value=True)
        namespace = load_definitions("utils.py", {"_CPUDataParallel", "load_pretrained_net"}, {
            "MODEL_FACTORIES": {"ResNet18": factory}, "inspect": inspect, "Path": Path,
            "torch": SimpleNamespace(device=Device, load=self.load,
                                     nn=SimpleNamespace(Module=Module, DataParallel=DataParallel),
                                     cuda=SimpleNamespace(is_available=self.cuda_available, current_device=lambda: 0)),
        })
        self.loader = namespace["load_pretrained_net"]
        self.directory = Path(tempfile.gettempdir()) / "msp-contract-models"

    def call_loader(self, filename, **kwargs):
        with redirect_stdout(io.StringIO()):
            return self.loader("ResNet18", filename, self.directory, **kwargs)

    def test_absolute_directory_template_and_absolute_checkpoint(self):
        cases = [("ckpt-%s.t7", self.directory / "ckpt-ResNet18.t7"),
                 ("fixed.t7", self.directory / "fixed.t7"),
                 (str(self.directory.parent / "explicit.t7"), self.directory.parent / "explicit.t7")]
        for filename, expected in cases:
            with self.subTest(filename=filename):
                self.call_loader(filename, device="cpu")
                args, kwargs = self.load.call_args
                self.assertEqual(args[0], expected)
                self.assertEqual(kwargs["map_location"].type, "cpu")
                self.assertIs(kwargs["weights_only"], False)

    def test_checkpoint_formats_and_dataparallel_prefix(self):
        for checkpoint in ({"net": {"module.linear.weight": "weight"}},
                           {"state_dict": {"linear.weight": "weight"}},
                           {"linear.weight": "weight"}):
            with self.subTest(checkpoint=checkpoint):
                self.load.return_value = checkpoint
                self.call_loader("fixed.t7", device="cpu")
                self.model.load_state_dict.assert_called_with({"linear.weight": "weight"})

    def test_explicit_cpu_ignores_visible_cuda(self):
        network = self.call_loader("fixed.t7", device="cpu")
        self.assertEqual(network.device_ids, [])
        self.assertFalse(network.training)
        self.assertEqual(self.model.to.call_args.args[0].type, "cpu")

    def test_cuda_index_is_used_for_map_location_and_primary_device(self):
        network = self.call_loader("fixed.t7", device="cuda:1")
        self.assertEqual(network.device_ids, [1])
        self.assertEqual(network.output_device, 1)
        mapped = self.load.call_args.kwargs["map_location"]
        self.assertEqual((mapped.type, mapped.index), ("cuda", 1))

    def test_unavailable_cuda_is_rejected(self):
        self.cuda_available.return_value = False
        with self.assertRaises(RuntimeError):
            self.call_loader("fixed.t7", device="cuda")
        self.load.assert_not_called()

    def test_model_input_is_registry_lookup_not_python_expression(self):
        with self.assertRaises(ValueError):
            self.loader("__import__('os').system('bad')", "fixed.t7", self.directory, device="cpu")
        self.load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
