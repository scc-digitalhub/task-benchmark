from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from task_benchmark.implementations.metrics_manager import (
    MetricsManager,
    MockupMetricsManager,
)


class MetricsManagerTests(unittest.TestCase):
    def test_base_class_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            MetricsManager()

    def test_mock_returns_example_unchanged_for_any_service(self) -> None:
        metrics_path = Path(__file__).resolve().parents[1] / "examples" / "json" / "dh_core_metrics.json"
        expected = json.loads(metrics_path.read_text(encoding="utf-8"))
        manager = MockupMetricsManager(metrics_path)

        self.assertIsInstance(manager, MetricsManager)
        self.assertEqual(manager.collect_metrics("pod-1"), expected)
        self.assertEqual(manager.collect_metrics("service-2"), expected)
        params = {"start": 1790812923, "filters": {"metric": "cpu"}}
        self.assertEqual(manager.collect_metrics("pod-1", params=params), expected)
        self.assertEqual(params, {"start": 1790812923, "filters": {"metric": "cpu"}})
        self.assertEqual(json.loads(json.dumps(manager.collect_metrics("pod-1"))), expected)

    def test_responses_are_independent_and_fixture_is_loaded_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metrics_path = Path(directory) / "metrics.json"
            metrics_path.write_text('{"metrics": [{"value": 0.5}]}', encoding="utf-8")
            manager = MockupMetricsManager(metrics_path)
            metrics_path.unlink()
            response = manager.collect_metrics("pod-1")
            response["metrics"][0]["value"] = 100

            self.assertEqual(
                manager.collect_metrics("pod-1"),
                {"metrics": [{"value": 0.5}]},
            )

    def test_rejects_empty_identifier(self) -> None:
        metrics_path = Path(__file__).resolve().parents[1] / "examples" / "json" / "dh_core_metrics.json"
        manager = MockupMetricsManager(metrics_path)
        with self.assertRaises(ValueError):
            manager.collect_metrics(" ")


if __name__ == "__main__":
    unittest.main()