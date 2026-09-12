import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    'evaluate_vlm_quality',
    Path(__file__).resolve().parents[1] / 'tools/evaluate_vlm_quality.py')
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


class QualityTest(unittest.TestCase):
    def test_clean_english_answer_with_expected_concepts(self):
        checks = quality.inspect(
            'A gray office chair is beside a desk.', 'en',
            [['chair', 'seat'], ['desk', 'table']], 2)
        self.assertTrue(all(checks.values()))

    def test_repetition_and_script_leakage_fail(self):
        checks = quality.inspect('사진에는 插座 반복 반복 반복', 'ko', [], 2)
        self.assertFalse(checks['expected_script'])
        self.assertFalse(checks['max_consecutive_token_repeat'])

    def test_missing_image_concept_fails(self):
        checks = quality.inspect(
            'There is food on a yellow table.', 'en', [['chair', 'seat']], 2)
        self.assertFalse(checks['required_concepts'])


if __name__ == '__main__':
    unittest.main()
