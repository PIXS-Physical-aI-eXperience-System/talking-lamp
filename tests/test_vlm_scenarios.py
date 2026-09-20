import importlib.util
import unittest
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "bench_vlm_scenarios",
    Path(__file__).resolve().parents[1] / "tools/bench_vlm_scenarios.py",
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class ScenarioEvaluationTest(unittest.TestCase):
    def test_valid_answer_passes(self):
        text = ('{"observation":"A hand adjusts a robot lamp.",'
                '"speech_ko":"램프를 조절하고 있네요.","motion":"curious"}')
        case = {
            "required_any": [["hand"], ["lamp"]],
            "expected_motions": ["curious"],
        }
        self.assertTrue(bench.evaluate(text, case)["passed"])

    def test_markdown_json_can_be_diagnosed_but_wrong_motion_fails(self):
        text = ('```json\n{"observation":"A robot lamp.",'
                '"speech_ko":"램프예요.","motion":"dance"}\n```')
        result = bench.evaluate(text, {
            "required_any": [["lamp"]],
            "forbidden_motions": ["dance"],
        })
        self.assertIsNotNone(result["parsed"])
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["allowed_motion"])

    def test_missing_visual_concept_fails(self):
        text = ('{"observation":"A desk.","speech_ko":"책상이네요.",'
                '"motion":"nod"}')
        result = bench.evaluate(text, {"required_any": [["lamp", "robot"]]})
        self.assertFalse(result["checks"]["visual_concepts"])


if __name__ == "__main__":
    unittest.main()
