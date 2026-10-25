"""Dependency-free regression tests for experiment planning and result accounting."""
import json
from pathlib import Path
import tempfile
import unittest

from craft_poisons_transfer import build_parser as attack_parser, validate_args
from experiment_protocol import ensure_protocol, parse_bool, validate_attack
from summarize_results import summarize
from train_cifar10_models import build_parser as training_parser
from workflow import ROOT, build_plan, load_config


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / 'configs/smoke.json')

    def test_every_smoke_command_parses_and_method_reaches_trainer(self):
        plan = build_plan(self.config)
        self.assertEqual(len(plan['pretrain']), 2)
        self.assertEqual(len(plan['craft']), 5)
        for command in plan['pretrain']:
            args = training_parser().parse_args(command[2:])
            self.assertEqual(args.data_path, self.config['dataset']['split'])
            self.assertEqual(args.epochs, 300)
        modes = set()
        for stage in ('craft', 'evaluate'):
            for command in plan[stage]:
                args = attack_parser().parse_args(command[2:])
                validate_args(args)
                self.assertEqual(args.stage, stage)
                self.assertFalse(args.end2end)
                modes.add((args.mode, args.feature_mode))
        self.assertEqual(len(modes), 5)

    def test_benchmark_uses_18_proxies_and_300_targets_methods(self):
        config = load_config(ROOT / 'configs/benchmark.json')
        plan = build_plan(config)
        self.assertEqual(len(plan['pretrain']), 20)
        self.assertEqual(len(plan['craft']), 300)
        args = attack_parser().parse_args(plan['craft'][0][2:])
        self.assertEqual(len(args.substitute_nets) * len(args.subs_chk_name), 18)

    def test_empty_schedules_stay_empty(self):
        self.config['pretrain']['milestones'] = []
        self.config['victim']['milestones'] = []
        plan = build_plan(self.config)
        self.assertEqual(training_parser().parse_args(plan['pretrain'][0][2:]).milestones, [])
        self.assertEqual(attack_parser().parse_args(plan['craft'][0][2:]).lr_decay_epoch, [])

    def test_checkpoint_collisions_rejected(self):
        self.config['attack']['checkpoints'] = ['one-file.t7']
        self.config['attack']['substitute_nets'] = ['ResNet18', 'ResNet50']
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            build_plan(self.config)

    def test_false_really_means_false(self):
        self.assertFalse(parse_bool('False'))
        self.assertFalse(attack_parser().parse_args(['--end2end', 'False']).end2end)
        self.assertTrue(attack_parser().parse_args(['--end2end']).end2end)

    def test_msp_e2e_and_convex_layerwise_rejected(self):
        with self.assertRaises(ValueError):
            validate_attack('mean', 'flat', True)
        with self.assertRaises(ValueError):
            validate_attack('convex', 'flat-layerwise', False)
        with self.assertRaises(ValueError):
            validate_attack('coeffs_fixed_type_99', 'original', False)

    def test_target_in_training_window_rejected(self):
        args = attack_parser().parse_args(['--target-start', '0'])
        with self.assertRaisesRegex(ValueError, 'inside'):
            validate_args(args)

    def test_e2e_unsupported_architecture_fails_before_loading_weights(self):
        args = attack_parser().parse_args(['--end2end', 'true'])
        with self.assertRaisesRegex(ValueError, 'block interface'):
            validate_args(args)

    def test_configuration_typos_rejected(self):
        original = json.loads((ROOT / 'configs/smoke.json').read_text())
        original['attack']['poison_epislon'] = .05
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, 'Unknown attack'):
                load_config(path)

    def test_manifest_refuses_changed_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'protocol.json'
            protocol = {'method': 'MSP-Flat', 'epsilon': .1}
            ensure_protocol(path, protocol)
            ensure_protocol(path, protocol)
            with self.assertRaisesRegex(ValueError, 'changed'):
                ensure_protocol(path, dict(protocol, epsilon=.05))

    def test_manifest_refuses_unrecorded_historical_poison(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'poison_00001.pth').touch()
            with self.assertRaisesRegex(ValueError, 'historical'):
                ensure_protocol(Path(directory, 'protocol.json'), {})


class SummaryTests(unittest.TestCase):
    def payload(self):
        def result(index, eligible, success):
            return dict(victim='ResNet50', seed=2026, target_index=index,
                clean_target_correct=eligible, attack_success=success,
                clean={'clean_accuracy_percent': 90}, poisoned={'clean_accuracy_percent': 88})
        return dict(protocol={'method': 'MSP-Flat'}, results=[
            result(0, True, True), result(1, True, False), result(2, False, True)])

    def test_conditional_denominator_and_accuracy_drop(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'evaluation.json').write_text(json.dumps(self.payload()))
            rows, seen = summarize(directory)
            self.assertEqual(rows[0]['eligible_targets'], 2)
            self.assertEqual(rows[0]['conditional_asr_percent'], 50)
            self.assertAlmostEqual(rows[0]['asr_percent'], 200/3)
            self.assertEqual(rows[0]['clean_accuracy_drop_points'], 2)
            self.assertEqual(len(seen), 3)

    def test_zero_eligible_is_missing_not_zero_success(self):
        payload = self.payload()
        for result in payload['results']:
            result['clean_target_correct'] = False
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'evaluation.json').write_text(json.dumps(payload))
            self.assertIsNone(summarize(directory)[0][0]['conditional_asr_percent'])

    def test_duplicates_are_not_double_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'a').mkdir()
            (root / 'b').mkdir()
            for child in ('a', 'b'):
                (root / child / 'evaluation.json').write_text(json.dumps(self.payload()))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                summarize(root)

    def test_different_targets_and_methods_share_only_control_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, method in enumerate(('BP-Original', 'MSP-Flat')):
                payload = self.payload()
                payload['results'] = [payload['results'][index]]
                payload['protocol'] = dict(method=method, data={}, substitutes=[], victims=[],
                    parameters=dict(target_index=index, mode='mean',
                                    feature_mode='original' if index == 0 else 'flat', retrain_lr=.1))
                child = Path(directory) / method
                child.mkdir()
                (child / 'evaluation.json').write_text(json.dumps(payload))
            self.assertEqual(len(summarize(directory)[0]), 2)
            payload['protocol']['parameters']['retrain_lr'] = .01
            (child / 'evaluation.json').write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, 'Different training'):
                summarize(directory)

    def test_different_base_selection_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, method in enumerate(('BP-Original', 'MSP-Flat')):
                payload = self.payload()
                payload['protocol']['method'] = method
                payload['base_indices'] = [index]
                child = Path(directory) / method
                child.mkdir()
                (child / 'evaluation.json').write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, 'base indices'):
                summarize(directory)


if __name__ == '__main__':
    unittest.main()
