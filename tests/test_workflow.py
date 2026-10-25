"""Dependency-free regression tests for experiment planning and result accounting."""
import json
from pathlib import Path
import tempfile
import unittest

from craft_poisons_transfer import build_parser as attack_parser, validate_args
from experiment_protocol import ensure_protocol, parse_bool, resolve_training_options, validate_attack
from summarize_results import summarize
from train_cifar10_models import build_parser as training_parser
from workflow import ROOT, build_plan, load_config, method_training_options


class TrainingOptionTests(unittest.TestCase):
    def test_default_features_and_linear_victim(self):
        self.assertEqual(resolve_training_options(None, 'original'), dict(
            attack_representation='features', attack_end2end=False,
            victim_training='linear', victim_model_mode='eval', end2end=False))

    def test_legacy_true_keeps_original_blocks_but_preserves_msp_features(self):
        original = resolve_training_options(True, 'original')
        self.assertEqual(original['attack_representation'], 'blocks')
        self.assertTrue(original['attack_end2end'])
        self.assertTrue(original['end2end'])
        for feature in ('pool', 'flat', 'flat-layerwise', 'pool-layerwise'):
            with self.subTest(feature=feature):
                settings = resolve_training_options(True, feature)
                self.assertEqual(settings['attack_representation'], 'features')
                self.assertFalse(settings['attack_end2end'])
                self.assertTrue(settings['end2end'])

    def test_victim_training_and_attack_representation_are_independent(self):
        linear_blocks = resolve_training_options(None, 'original', 'blocks', 'linear')
        self.assertTrue(linear_blocks['attack_end2end'])
        self.assertFalse(linear_blocks['end2end'])
        full_features = resolve_training_options(None, 'flat', 'features', 'end2end')
        self.assertFalse(full_features['attack_end2end'])
        self.assertTrue(full_features['end2end'])
        self.assertEqual(resolve_training_options(True, 'original', 'features')['attack_representation'],
                         'features')

    def test_legacy_and_explicit_victim_conflicts_rejected(self):
        for legacy, victim in ((True, 'linear'), (False, 'end2end')):
            with self.subTest(legacy=legacy, victim=victim):
                with self.assertRaisesRegex(ValueError, 'contradicts'):
                    resolve_training_options(legacy, 'original', victim_training=victim)

    def test_block_representation_requires_original_features(self):
        with self.assertRaisesRegex(ValueError, 'blocks require'):
            resolve_training_options(None, 'flat', 'blocks', 'end2end')
        with self.assertRaisesRegex(ValueError, 'blocks require'):
            validate_attack('mean', 'flat', True)
        validate_attack('mean', 'flat', False)

    def test_train_mode_requires_full_parameter_victim(self):
        with self.assertRaisesRegex(ValueError, 'backbone frozen'):
            resolve_training_options(None, 'flat', victim_training='linear', victim_model_mode='train')
        settings = resolve_training_options(None, 'flat', victim_training='end2end', victim_model_mode='train')
        self.assertEqual(settings['victim_model_mode'], 'train')

    def test_invalid_explicit_routes_rejected(self):
        for options in ({'attack_representation': 'unknown'}, {'victim_training': 'unknown'},
                        {'victim_model_mode': 'unknown'}):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    resolve_training_options(None, 'original', **options)


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
                self.assertEqual(args.attack_representation, 'features')
                self.assertEqual(args.victim_training, 'linear')
                self.assertEqual(args.victim_model_mode, 'eval')
                self.assertNotIn('--end2end', command)
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

    def test_msp_blocks_and_convex_layerwise_rejected(self):
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

    def test_blocks_unsupported_proxy_fails_before_loading_weights(self):
        args = attack_parser().parse_args(['--end2end', 'true', '--substitute-nets', 'DenseNet121'])
        with self.assertRaisesRegex(ValueError, 'block interface'):
            validate_args(args)

    def test_full_parameter_configs_keep_feature_attacks_and_all_architectures(self):
        for stem, jobs, epochs in (('smoke', 5, 2), ('benchmark', 300, 60)):
            with self.subTest(config=stem):
                config = load_config(ROOT / 'configs' / (stem + '_end2end.json'))
                original = load_config(ROOT / 'configs' / (stem + '.json'))
                self.assertEqual(config['attack']['methods'], original['attack']['methods'])
                self.assertNotEqual(config['output_root'], original['output_root'])
                self.assertEqual(config['victim']['nets'], original['victim']['nets'])
                plan = build_plan(config)
                self.assertEqual(len(plan['craft']), jobs)
                for stage in ('craft', 'evaluate'):
                    for command in plan[stage]:
                        self.assertNotIn('--end2end', command)
                        args = attack_parser().parse_args(command[2:])
                        validate_args(args)
                        self.assertEqual(args.attack_representation, 'features')
                        self.assertEqual(args.victim_training, 'end2end')
                        self.assertEqual(args.victim_model_mode, 'eval')
                        self.assertTrue(args.end2end)
                        self.assertEqual(args.retrain_lr, 1e-4)
                        self.assertEqual(args.retrain_wd, 0)
                        self.assertEqual(args.retrain_epochs, epochs)

    def test_legacy_method_default_and_explicit_routes(self):
        self.assertEqual(method_training_options({}, {}, 'BP-E2E-Original'), dict(
            attack_representation='blocks', attack_end2end=True,
            victim_training='end2end', victim_model_mode='eval', end2end=True))
        self.assertFalse(method_training_options({}, {'training': 'linear'},
                                                'BP-E2E-Original')['end2end'])
        with self.assertRaisesRegex(ValueError, 'matching method label BP-Original'):
            method_training_options({'representation': 'features'}, {}, 'BP-E2E-Original')

    def test_workflow_representation_cannot_relabel_the_attack(self):
        for method, representation, actual in (
                ('BP-Original', 'blocks', 'BP-E2E-Original'),
                ('BP-E2E-Original', 'features', 'BP-Original'),
                ('CP-Original', 'blocks', 'CP-E2E-Original')):
            with self.subTest(method=method, representation=representation):
                with self.assertRaisesRegex(ValueError, 'matching method label ' + actual):
                    method_training_options({'representation': representation}, {}, method)
        settings = method_training_options({'representation': 'blocks'}, {'training': 'linear'},
                                           'BP-E2E-Original')
        self.assertTrue(settings['attack_end2end'])
        self.assertFalse(settings['end2end'])
        self.assertEqual(method_training_options({'representation': 'features'}, {}, 'MSP-Flat')
                         ['victim_training'], 'linear')

    def test_two_labels_cannot_collapse_into_one_algorithm(self):
        config = json.loads((ROOT / 'configs/smoke.json').read_text())
        config['attack']['methods'] = ['BP-Original', 'BP-E2E-Original']
        config['attack']['representation'] = 'features'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'matching method label BP-Original'):
                load_config(path)

    def test_workflow_blocks_limit_proxies_but_not_diagnostic_victim(self):
        config = json.loads((ROOT / 'configs/smoke.json').read_text())
        config['attack']['methods'] = ['BP-E2E-Original']
        config['attack']['substitute_nets'] = ['ResNet18']
        config['victim']['nets'] = ['DenseNet121']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps(config))
            settings = load_config(path)
            args = attack_parser().parse_args(build_plan(settings)['craft'][0][2:])
            validate_args(args)
            self.assertEqual(args.attack_representation, 'blocks')
            self.assertEqual(args.victim_training, 'end2end')
            config['attack']['substitute_nets'] = ['DenseNet121']
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'block interface'):
                load_config(path)

    def test_workflow_rejects_linear_train_mode(self):
        config = json.loads((ROOT / 'configs/smoke.json').read_text())
        config['victim']['model_mode'] = 'train'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'backbone frozen'):
                load_config(path)

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
