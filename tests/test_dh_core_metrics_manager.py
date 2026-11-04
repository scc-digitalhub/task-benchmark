from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft7Validator

from task_benchmark.implementations.metrics_manager import DHCoreMetricsManager, MetricsManager


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DHCoreMetricsManagerTests(unittest.TestCase):
    def test_converts_real_export(self) -> None:
        path = PROJECT_ROOT / "examples" / "json" / "dh_core_metrics.json"
        entries = json.loads(path.read_text(encoding="utf-8"))
        pod = next(
            label.removeprefix("pod:")
            for label in entries[0]["metadata"]["labels"]
            if label.startswith("pod:")
        )
        manager = DHCoreMetricsManager(path)
        result = manager.collect_metrics(pod)

        schema = json.loads(
            (PROJECT_ROOT / "json" / "metrics_schema.json").read_text(encoding="utf-8")
        )
        Draft7Validator.check_schema(schema)
        Draft7Validator(schema).validate(result)

        self.assertIsInstance(manager, MetricsManager)
        self.assertEqual(result["service_id"], pod)
        self.assertEqual(set(result), {"service_id", "window_from", "window_to", "metrics"})
        all_timestamps = []
        expected_metrics = [metric for entry in entries for metric in entry["metrics"]]
        self.assertEqual(len(result["metrics"]), len(expected_metrics))
        for item, source in zip(result["metrics"], expected_metrics):
            metric = item
            self.assertEqual(set(metric), {"name", "unit", "values", "summary"})
            self.assertEqual(metric["name"], source["name"])
            self.assertEqual(metric["unit"], source["unit"])
            self.assertEqual(metric["values"], source["metrics"])
            self.assertEqual(metric["summary"], source["summary"])
            all_timestamps.extend(point["timestamp"] for point in metric["values"])
        self.assertEqual(result["window_from"], min(all_timestamps))
        self.assertEqual(result["window_to"], max(all_timestamps))

    def test_selects_service_and_strips_extra_fields(self) -> None:
        entry = {
            "id": "entry-1",
            "metadata": {"labels": ["service:service-1"]},
            "metrics": [{
                "name": "cpu", "unit": "seconds", "quota": None,
                "metrics": [{"timestamp": 20, "value": 0.5, "extra": True},
                            {"timestamp": 10, "value": 0.1}],
                "summary": [{"name": "avg", "value": 0.3, "extra": True}],
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            other = {"id": "other", "metadata": {}, "metrics": []}
            path.write_text(json.dumps([entry, other]), encoding="utf-8")
            manager = DHCoreMetricsManager(path)
            result = manager.collect_metrics("service-1")
            self.assertEqual(result["window_from"], 10)
            self.assertEqual(result["window_to"], 20)
            self.assertEqual(result["metrics"][0]["summary"],
                             [{"name": "avg", "value": 0.3}])
            self.assertNotIn("extra", result["metrics"][0]["values"][0])
            self.assertEqual(len(result["metrics"]), 1)
            with self.assertRaisesRegex(ValueError, "No DHCore metrics"):
                manager.collect_metrics("unknown")
            self.assertEqual(manager.collect_metrics("entry-1")["metrics"], result["metrics"])

    def test_empty_series_do_not_invent_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            path.write_text(json.dumps([{"id": "service-1", "metrics": [{
                "name": "network", "unit": "bytes", "metrics": [], "summary": []
            }]}]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "without timestamps"):
                DHCoreMetricsManager(path).collect_metrics("service-1")

    def test_invalid_values_are_rejected(self) -> None:
        for timestamp, value in [(True, 1), (10.5, 1), (10, True), (10, float("nan"))]:
            with self.subTest(timestamp=timestamp, value=value):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "metrics.json"
                    path.write_text(json.dumps([{"id": "service-1", "metrics": [{
                        "name": "cpu", "unit": "seconds", "summary": [],
                        "metrics": [{"timestamp": timestamp, "value": value}],
                    }]}]), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        DHCoreMetricsManager(path).collect_metrics("service-1")

    def test_filters_inclusive_window_and_recomputes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            path.write_text(json.dumps([{"id": "service-1", "metrics": [{
                "name": "cpu", "unit": "seconds",
                "metrics": [{"timestamp": timestamp, "value": value}
                            for timestamp, value in [(10, 1), (20, 2), (30, 4), (40, 100)]],
                "summary": [{"name": "sum", "value": 107}],
            }]}]), encoding="utf-8")
            manager = DHCoreMetricsManager(path)
            result = manager.collect_metrics("service-1", window_from=20, window_to=30)
            metric = result["metrics"][0]
            self.assertEqual([point["timestamp"] for point in metric["values"]], [20, 30])
            self.assertEqual({item["name"]: item["value"] for item in metric["summary"]},
                             {"sum": 6, "avg": 3, "max": 4, "min": 2})
            self.assertEqual((result["window_from"], result["window_to"]), (20, 30))
            for bounds, expected in [({"window_from": 30}, [30, 40]),
                                     ({"window_to": 20}, [10, 20]),
                                     ({"window_from": 20, "window_to": 20}, [20])]:
                with self.subTest(bounds=bounds):
                    filtered = manager.collect_metrics("service-1", **bounds)
                    self.assertEqual(
                        [point["timestamp"] for point in filtered["metrics"][0]["values"]],
                        expected,
                    )
            empty = manager.collect_metrics("service-1", window_from=50, window_to=60)
            self.assertEqual(empty["metrics"][0]["values"], [])
            self.assertEqual(empty["metrics"][0]["summary"], [])
            schema = json.loads((PROJECT_ROOT / "json" / "metrics_schema.json").read_text())
            Draft7Validator(schema).validate(result)
            Draft7Validator(schema).validate(empty)

    def test_rejects_invalid_time_bounds(self) -> None:
        manager = DHCoreMetricsManager()
        for bounds in ({"window_from": True}, {"window_to": 1.5}, {"window_from": "10"}):
            with self.subTest(bounds=bounds):
                with self.assertRaises(TypeError):
                    manager.collect_metrics("service-1", **bounds)
        with self.assertRaises(ValueError):
            manager.collect_metrics("service-1", window_from=20, window_to=10)


if __name__ == "__main__":
    unittest.main()