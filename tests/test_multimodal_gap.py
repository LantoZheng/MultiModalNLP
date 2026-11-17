import json
import tempfile
import unittest
from pathlib import Path

from multimodal_gap import analyze, load_records


class AnalyzeTests(unittest.TestCase):
    def test_compares_model_families_on_shared_tasks(self):
        records = [
            {"task_id": "vision", "model_family": "LLM", "expected": "cat", "prediction": "cat"},
            {"task_id": "vision", "model_family": "VLM", "expected": "cat", "prediction": "dog"},
            {"task_id": "audio", "model_family": "LLM", "expected": "yes", "prediction": "no"},
            {"task_id": "audio", "model_family": "ALM", "expected": "yes", "prediction": "yes"},
            {"task_id": "text-only", "model_family": "VLM", "expected": "ok", "prediction": "ok"},
        ]

        report = analyze(records)

        self.assertEqual(report["family_accuracy"]["LLM"]["accuracy"], 0.5)
        self.assertEqual(report["family_accuracy"]["VLM"]["total"], 2)
        self.assertEqual(report["gaps_vs_llm"]["VLM"]["tasks_compared"], 1)
        self.assertEqual(report["gaps_vs_llm"]["VLM"]["mean_accuracy_difference"], -1.0)
        self.assertEqual(report["gaps_vs_llm"]["ALM"]["mean_accuracy_difference"], 1.0)

    def test_normalizes_family_and_answer_case_and_whitespace(self):
        report = analyze([
            {"task_id": "task", "model_family": " vlm ", "expected": " A CAT ", "prediction": "a cat"}
        ])

        self.assertEqual(report["family_accuracy"]["VLM"]["accuracy"], 1.0)
        self.assertEqual(report["gaps_vs_llm"], {})

    def test_gap_weights_shared_tasks_equally(self):
        records = []
        for _ in range(10):
            records.extend([
                {"task_id": "large", "model_family": "LLM", "expected": "yes", "prediction": "no"},
                {"task_id": "large", "model_family": "VLM", "expected": "yes", "prediction": "yes"},
            ])
        records.extend([
            {"task_id": "small", "model_family": "LLM", "expected": "yes", "prediction": "yes"},
            {"task_id": "small", "model_family": "VLM", "expected": "yes", "prediction": "no"},
        ])

        report = analyze(records)

        self.assertAlmostEqual(report["gaps_vs_llm"]["VLM"]["mean_accuracy_difference"], 0.0)


class LoadRecordsTests(unittest.TestCase):
    def test_rejects_missing_fields_with_line_number(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            path.write_text(json.dumps({"task_id": "task"}) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "line 1: missing required fields"):
                load_records(path)

    def test_rejects_invalid_json_with_line_number(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            path.write_text("{bad json}\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "line 1: invalid JSON"):
                load_records(path)


if __name__ == "__main__":
    unittest.main()
