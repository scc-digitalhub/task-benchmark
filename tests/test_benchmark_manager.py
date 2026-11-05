from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from jsonschema import Draft7Validator

from task_benchmark import BenchmarkManager
from task_benchmark.impl.data_loader.data_loader import DataLoader
from task_benchmark.impl.evaluator.base_evaluator import Evaluator
from task_benchmark.impl.invoker.base_invoker import Invoker


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ListDataLoader(DataLoader):
    def __init__(self, items: list[tuple[str, str]]) -> None:
        super().__init__("test-dataset", "")
        self._items = iter(items)

    def next(self) -> tuple[str, str]:
        return next(self._items)


class UppercaseInvoker(Invoker[str, str]):
    def invoke(self, test_data: str) -> str:
        return test_data.upper()


class SampleEvaluator(Evaluator[str]):
    def __init__(self) -> None:
        self.contexts: list[str] = []

    def _evaluate(self, test_context: str, service_response: str) -> float:
        self.contexts.append(test_context)
        return {"sample-1": 0.75, "sample-2": 0.5}[test_context]


class BenchmarkManagerTests(unittest.TestCase):
    def test_runs_dataset_and_returns_schema_valid_metrics(self) -> None:
        evaluator = SampleEvaluator()
        manager = BenchmarkManager(
            ListDataLoader([("sample-1", "first"), ("sample-2", "second")]),
            UppercaseInvoker(),
            evaluator,
            service_id="service-test",
        )
        schema = json.loads(
            (PROJECT_ROOT / "json" / "metrics_schema.json").read_text(encoding="utf-8")
        )

        with (
            patch(
                "task_benchmark.benchmark_manager.time.time",
                side_effect=[1000, 1001, 1002, 1003, 1004, 1005],
            ),
            patch(
                "task_benchmark.benchmark_manager.time.perf_counter",
                side_effect=[1.0, 1.25, 2.0, 2.5],
            ),
            patch("builtins.print") as output,
        ):
            result = manager.run_benchmark()

        Draft7Validator.check_schema(schema)
        Draft7Validator(schema).validate(result)
        self.assertEqual(evaluator.contexts, ["sample-1", "sample-2"])
        self.assertEqual(output.call_args_list[0].args[0],
                         "sample-1: inference_time=0.250000000s, quality_score=0.7500")
        self.assertEqual(output.call_args_list[1].args[0],
                         "sample-2: inference_time=0.500000000s, quality_score=0.5000")
        self.assertEqual((result["window_from"], result["window_to"]), (1000, 1005))
        self.assertEqual(
            [metric["name"] for metric in result["metrics"]],
            ["inference_time", "quality_score"],
        )
        inference_time, quality_score = result["metrics"]
        self.assertEqual(inference_time["unit"], "seconds")
        self.assertEqual(inference_time["values"], [
            {"timestamp": 1001, "value": 0.25},
            {"timestamp": 1003, "value": 0.5},
        ])
        self.assertEqual(quality_score["unit"], "ratio")
        self.assertEqual(quality_score["values"], [
            {"timestamp": 1002, "value": 0.75},
            {"timestamp": 1004, "value": 0.5},
        ])
        self.assertEqual(
            {item["name"]: item["value"] for item in quality_score["summary"]},
            {"avg": 0.625, "max": 0.75, "min": 0.5},
        )


if __name__ == "__main__":
    unittest.main()