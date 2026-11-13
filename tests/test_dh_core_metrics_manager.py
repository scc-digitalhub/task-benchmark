from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from jsonschema import Draft7Validator

from task_benchmark.impl.metrics_manager import DHCoreMetricsManager, MetricsManager


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_URL = "https://core.dev.atlas.fbk.eu/api/v1/-/test-hf/resource_metrics/runs"


class DHCoreMetricsManagerTests(unittest.TestCase):
    @staticmethod
    @contextmanager
    def mock_api_response(entries):
        response_body = json.dumps(entries).encode("utf-8")
        with patch.dict(os.environ, {"DHCORE_ACCESS_TOKEN": "test-token"}):
            with patch(
                "task_benchmark.impl.metrics_manager.dh_core.urlopen",
                side_effect=(BytesIO(response_body) for _ in range(20)),
            ) as urlopen_mock:
                yield urlopen_mock

    def test_converts_real_export(self) -> None:
        path = PROJECT_ROOT / "examples" / "json" / "dh_core_metrics.json"
        entries = json.loads(path.read_text(encoding="utf-8"))
        run_id = next(
            label.removeprefix("dhcore_run:")
            for label in entries[0]["metadata"]["labels"]
            if label.startswith("dhcore_run:")
        )
        manager = DHCoreMetricsManager()
        with patch.dict(os.environ, {"DHCORE_ACCESS_TOKEN": "test-token"}):
            with patch(
                "task_benchmark.impl.metrics_manager.dh_core.urlopen",
                return_value=BytesIO(path.read_bytes()),
            ) as urlopen_mock:
                result = manager.collect_metrics(run_id)
        urlopen_mock.assert_called_once()
        request = urlopen_mock.call_args.args[0]
        self.assertEqual(request.full_url, f"{RUNS_URL}/{run_id}")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-token")
        self.assertEqual(urlopen_mock.call_args.kwargs, {"timeout": 30})

        schema = json.loads(
            (PROJECT_ROOT / "json" / "metrics_schema.json").read_text(encoding="utf-8")
        )
        Draft7Validator.check_schema(schema)
        Draft7Validator(schema).validate(result)

        self.assertIsInstance(manager, MetricsManager)
        self.assertEqual(result["service_id"], run_id)
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
        other = {"id": "other", "metadata": {}, "metrics": []}
        manager = DHCoreMetricsManager("https://example.test/runs")
        with self.mock_api_response([entry, other]):
            result = manager.collect_metrics("service-1")
            with self.assertRaisesRegex(ValueError, "No DHCore metrics"):
                manager.collect_metrics("unknown")
            self.assertEqual(
                manager.collect_metrics("entry-1")["metrics"],
                result["metrics"],
            )
        self.assertEqual(result["window_from"], 10)
        self.assertEqual(result["window_to"], 20)
        self.assertEqual(result["metrics"][0]["summary"],
                         [{"name": "avg", "value": 0.3}])
        self.assertNotIn("extra", result["metrics"][0]["values"][0])
        self.assertEqual(len(result["metrics"]), 1)

    def test_filters_requested_metric_names(self) -> None:
        entries = [{"id": "service-1", "metrics": [
            {
                "name": "cpu",
                "unit": "seconds",
                "metrics": [{"timestamp": 10, "value": 0.5}],
                "summary": [],
            },
            {
                "name": "memory",
                "unit": "bytes",
                "metrics": [{"timestamp": 10, "value": 1024}],
                "summary": [],
            },
        ]}]
        with self.mock_api_response(entries):
            result = DHCoreMetricsManager().collect_metrics(
                "service-1",
                window_from=10,
                window_to=10,
                metric_names=["memory"],
            )
        self.assertEqual([metric["name"] for metric in result["metrics"]], ["memory"])

    def test_empty_metric_name_filter_returns_no_series(self) -> None:
        entries = [{"id": "service-1", "metrics": [{
            "name": "cpu",
            "unit": "seconds",
            "metrics": [{"timestamp": 10, "value": 0.5}],
            "summary": [],
        }]}]
        with self.mock_api_response(entries):
            result = DHCoreMetricsManager().collect_metrics(
                "service-1",
                window_from=10,
                window_to=10,
                metric_names=[],
            )
        self.assertEqual(result["metrics"], [])

    def test_metric_name_filter_requires_strings(self) -> None:
        for metric_names in ("cpu", ["cpu", 1]):
            with self.subTest(metric_names=metric_names):
                with self.assertRaises(TypeError):
                    DHCoreMetricsManager().collect_metrics(
                        "service-1",
                        metric_names=metric_names,
                    )

    def test_empty_series_do_not_invent_window(self) -> None:
        entries = [{"id": "service-1", "metrics": [{
            "name": "network", "unit": "bytes", "metrics": [], "summary": []
        }]}]
        with self.mock_api_response(entries):
            with self.assertRaisesRegex(ValueError, "without timestamps"):
                DHCoreMetricsManager().collect_metrics("service-1")

    def test_invalid_values_are_rejected(self) -> None:
        for timestamp, value in [(True, 1), (10.5, 1), (10, True), (10, float("nan"))]:
            with self.subTest(timestamp=timestamp, value=value):
                entries = [{"id": "service-1", "metrics": [{
                    "name": "cpu", "unit": "seconds", "summary": [],
                    "metrics": [{"timestamp": timestamp, "value": value}],
                }]}]
                with self.mock_api_response(entries):
                    with self.assertRaises(ValueError):
                        DHCoreMetricsManager().collect_metrics("service-1")

    def test_filters_inclusive_window_and_recomputes_summary(self) -> None:
        entries = [{"id": "service-1", "metrics": [{
            "name": "cpu", "unit": "seconds",
            "metrics": [{"timestamp": timestamp, "value": value}
                        for timestamp, value in [(10, 1), (20, 2), (30, 4), (40, 100)]],
            "summary": [{"name": "sum", "value": 107}],
        }]}]
        with self.mock_api_response(entries):
            manager = DHCoreMetricsManager()
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

    def test_builds_encoded_run_url_from_custom_prefix(self) -> None:
        entries = [{"id": "run/42", "metrics": [{
            "name": "cpu",
            "unit": "seconds",
            "metrics": [{"timestamp": 10, "value": 0.5}],
            "summary": [],
        }]}]
        manager = DHCoreMetricsManager("https://example.test/api/runs/")
        with self.mock_api_response(entries) as urlopen_mock:
            manager.collect_metrics("run/42")
        request = urlopen_mock.call_args.args[0]
        self.assertEqual(request.full_url, "https://example.test/api/runs/run%2F42")

    def test_requires_token_before_making_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"DHCORE_ACCESS_TOKEN": ""}):
                with patch(
                    "task_benchmark.impl.metrics_manager.dh_core.Path.home",
                    return_value=Path(directory),
                ):
                    with patch(
                        "task_benchmark.impl.metrics_manager.dh_core.urlopen"
                    ) as urlopen_mock:
                        with self.assertRaisesRegex(RuntimeError, "DHCORE_ACCESS_TOKEN"):
                            DHCoreMetricsManager().collect_metrics("run-1")
        urlopen_mock.assert_not_called()

    def test_reads_token_from_active_dhcore_ini_when_env_is_missing(self) -> None:
        entries = [{"id": "run-1", "metrics": [{
            "name": "cpu",
            "unit": "seconds",
            "metrics": [{"timestamp": 10, "value": 0.5}],
            "summary": [],
        }]}]
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / ".dhcore.ini"
            config_path.write_text(
                "[DEFAULT]\n"
                "current_environment = dev\n"
                "\n"
                "[dev]\n"
                "dhcore_access_token = ini-token\n"
                "\n"
                "[other]\n"
                "dhcore_access_token = other-token\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"DHCORE_ACCESS_TOKEN": ""}):
                with patch(
                    "task_benchmark.impl.metrics_manager.dh_core.Path.home",
                    return_value=Path(directory),
                ):
                    with patch(
                        "task_benchmark.impl.metrics_manager.dh_core.urlopen",
                        return_value=BytesIO(json.dumps(entries).encode("utf-8")),
                    ) as urlopen_mock:
                        DHCoreMetricsManager().collect_metrics("run-1")

        request = urlopen_mock.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer ini-token")

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