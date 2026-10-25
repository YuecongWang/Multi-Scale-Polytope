"""Prevent linear results from being reported as full-parameter MSP evaluations."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from summarize_results import main, summarize

ROOT = Path(__file__).resolve().parents[1]


class SummaryTrainingTests(unittest.TestCase):
    def result(self, training='end2end', index=0):
        return dict(victim='ResNet50', seed=2026, target_index=index, victim_training=training,
                    victim_model_mode='eval', clean_target_correct=True, attack_success=False,
                    clean={'clean_accuracy_percent': 90}, poisoned={'clean_accuracy_percent': 89})

    def write(self, directory, results):
        Path(directory, 'evaluation.json').write_text(json.dumps(
            dict(protocol={'method': 'MSP-Flat'}, results=results)), encoding='utf-8')

    def test_linear_evaluation_cannot_pass_as_end2end_even_when_incomplete_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write(directory, [self.result('linear')])
            error = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(error), self.assertRaises(SystemExit) as raised:
                main([directory, '--config', str(ROOT/'configs/smoke_end2end.json'), '--allow-incomplete'])
            self.assertEqual(raised.exception.code, 1)
            self.assertIn('training mode does not match', error.getvalue())

    def test_full_training_result_is_preserved_in_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write(directory, [self.result()])
            with redirect_stdout(io.StringIO()):
                main([directory, '--config', str(ROOT/'configs/smoke_end2end.json'), '--allow-incomplete'])
            row = json.loads(Path(directory, 'summary.json').read_text())['rows'][0]
            self.assertEqual(row['victim_training'], 'end2end')
            self.assertEqual(row['victim_model_mode'], 'eval')

    def test_mixed_training_modes_are_never_averaged(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write(directory, [self.result('linear'), self.result('end2end', index=1)])
            with self.assertRaisesRegex(ValueError, 'Mixed victim training'):
                summarize(directory)


if __name__ == '__main__':
    unittest.main()
