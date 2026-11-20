from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, call, patch


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "tvm"
    / "benchmark_dh_tvm.py"
)
spec = importlib.util.spec_from_file_location("benchmark_dh_tvm", EXAMPLE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load the TVM benchmark example")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


class TVMBenchmarkExampleTests(unittest.TestCase):
    def test_merge_dhcore_metrics_returns_benchmark_metrics_on_collection_error(self) -> None:
        benchmark_metrics = {
            "service_id": "benchmark-service",
            "window_from": 100,
            "window_to": 200,
            "metrics": [{"name": "quality_score", "values": []}],
        }
        with (
            patch.object(
                example.DHCoreMetricsManager,
                "collect_metrics",
                side_effect=ValueError("No DHCore metrics found"),
            ) as collect_metrics,
            self.assertLogs("benchmark_dh_tvm", level="WARNING") as logs,
        ):
            result = example.merge_dhcore_metrics(
                benchmark_metrics,
                "run-123",
                ["cpu", "memory"],
            )

        self.assertEqual(result, benchmark_metrics)
        self.assertIsNot(result, benchmark_metrics)
        collect_metrics.assert_called_once_with(
            "run-123",
            window_from=100,
            window_to=200,
            metric_names=["cpu", "memory"],
        )
        self.assertIn("returning benchmark metrics only", logs.output[0])

    def test_merge_dhcore_metrics_returns_benchmark_metrics_on_invalid_response(self) -> None:
        benchmark_metrics = {
            "service_id": "benchmark-service",
            "window_from": 100,
            "window_to": 200,
            "metrics": [{"name": "quality_score", "values": []}],
        }
        invalid_responses = [None, {"metrics": None}]
        for invalid_response in invalid_responses:
            with self.subTest(invalid_response=invalid_response):
                with (
                    patch.object(
                        example.DHCoreMetricsManager,
                        "collect_metrics",
                        return_value=invalid_response,
                    ),
                    self.assertLogs("benchmark_dh_tvm", level="WARNING"),
                ):
                    result = example.merge_dhcore_metrics(
                        benchmark_metrics,
                        "run-123",
                        ["cpu", "memory"],
                    )

                self.assertEqual(result, benchmark_metrics)

    def test_benchmarks_each_deployment_using_its_node_config(self) -> None:
        deployments = [
            ("cpu", "amd64", "https://service-a.example"),
            ("jetson-orin", "arm64", "https://service-b.example"),
        ]
        config = {
            "project": "test-deploy",
            "served_name": "model-default",
            "nodes": [
                {"profile": "cpu", "target_architecture": "amd64"},
                {
                    "profile": "jetson-orin",
                    "target_architecture": "arm64",
                    "served_name": "model-arm",
                },
            ],
        }
        deploy_manager = Mock()
        with patch.object(
            example,
            "benchmark_service",
            side_effect=[Path("first.json"), Path("second.json")],
        ) as benchmark_service:
            result = example.benchmark_deployments(
                deploy_manager,
                deployments,
                config,
                max_samples=5,
                local_port=8080,
            )

        self.assertEqual(result, [Path("first.json"), Path("second.json")])
        self.assertEqual(benchmark_service.call_args_list, [
            call(
                service_url="https://service-a.example",
                deploy_manager=deploy_manager,
                project="test-deploy",
                profile="cpu",
                target_architecture="amd64",
                served_name="model-default",
                max_samples=5,
                local_port=8080,
                use_port_forward=True,
            ),
            call(
                service_url="https://service-b.example",
                deploy_manager=deploy_manager,
                project="test-deploy",
                profile="jetson-orin",
                target_architecture="arm64",
                served_name="model-arm",
                max_samples=5,
                local_port=8081,
                use_port_forward=True,
            ),
        ])

    def test_benchmark_service_saves_json_with_extracted_run_id(self) -> None:
        run_id = "cbe0097faab5492895f218158d367747"
        service_url = f"s-tvmserve-{run_id}.dev-platform:8080"
        metadata = {
            "inputs": [{
                "name": "images",
                "datatype": "FP32",
                "shape": [1, 3, 224, 224],
            }],
            "outputs": [{
                "name": "output0",
                "datatype": "FP32",
                "shape": [1, 1000],
            }],
        }
        benchmark_metrics = {
            "service_id": "service",
            "window_from": 100,
            "window_to": 200,
            "metrics": [{"name": "quality_score"}],
        }
        resource_metrics = [{"name": "cpu"}]
        loader = Mock()
        classifier = Mock()
        benchmark_manager = Mock()
        benchmark_manager.run_benchmark.return_value = benchmark_metrics
        metrics_manager = Mock()
        metrics_manager.collect_metrics.return_value = {"metrics": resource_metrics}
        deploy_manager = Mock()
        base_url = "http://localhost:8080/v2/models/yolo26n-cls"
        deploy_manager.extract_run_id.return_value = run_id
        deploy_manager.open_inference_service.return_value = nullcontext(
            (base_url, metadata)
        )

        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as output_directory:
            os.chdir(output_directory)
            try:
                with (
                    patch.object(
                        example,
                        "ImageNet10ImageClassificationDataLoader",
                        return_value=loader,
                    ),
                    patch.object(example, "class_labels", return_value={0: "label"}),
                    patch.object(example, "OpenInferenceInvoker") as invoker,
                    patch.object(
                        example,
                        "OpenInferenceImageClassificationInvoker",
                        return_value=classifier,
                    ),
                    patch.object(example, "Top1ImageClassificationEvaluator"),
                    patch.object(
                        example,
                        "BenchmarkManager",
                        return_value=benchmark_manager,
                    ),
                    patch.object(
                        example,
                        "DHCoreMetricsManager",
                        return_value=metrics_manager,
                    ),
                ):
                    metrics_path = example.benchmark_service(
                        service_url=service_url,
                        deploy_manager=deploy_manager,
                        project="test-deploy",
                        profile="cpu",
                        target_architecture="amd64",
                        served_name="yolo26n-cls",
                        max_samples=5,
                        local_port=8080,
                    )

                self.assertTrue(metrics_path.is_file())
                saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            finally:
                os.chdir(original_directory)

        self.assertEqual(
            invoker.call_args.args[0],
            "http://localhost:8080/v2/models/yolo26n-cls/infer",
        )
        deploy_manager.open_inference_service.assert_called_once_with(
            service_url,
            project="test-deploy",
            served_name="yolo26n-cls",
            local_port=8080,
            use_port_forward=True,
        )
        deploy_manager.stop_run.assert_called_once_with("test-deploy", run_id)
        metrics_manager.collect_metrics.assert_called_once_with(
            run_id,
            window_from=100,
            window_to=200,
            metric_names=["cpu", "memory"],
        )
        self.assertEqual(saved_metrics["run_id"], run_id)
        self.assertEqual(saved_metrics["profile"], "cpu")
        self.assertEqual(saved_metrics["target_architecture"], "amd64")
        self.assertEqual(
            saved_metrics["metrics"],
            [{"name": "quality_score"}, *resource_metrics],
        )
        classifier.close.assert_called_once_with()

    def test_benchmark_service_can_use_direct_service_url(self) -> None:
        run_id = "cbe0097faab5492895f218158d367747"
        service_url = f"s-tvmserve-{run_id}.dev-platform:8080"
        deploy_manager = Mock()
        deploy_manager.extract_run_id.return_value = run_id
        deploy_manager.open_inference_service.return_value = nullcontext(
            ("http://s-tvmserve.example/v2/models/yolo26n-cls", {})
        )
        with patch.object(
            example,
            "ImageNet10ImageClassificationDataLoader",
        ) as data_loader:
            with self.assertRaisesRegex(ValueError, "Expected the YOLO FP32 input"):
                example.benchmark_service(
                    service_url=service_url,
                    deploy_manager=deploy_manager,
                    project="test-deploy",
                    profile="cpu",
                    target_architecture="amd64",
                    served_name="yolo26n-cls",
                    max_samples=5,
                    local_port=8080,
                    use_port_forward=False,
                )

        data_loader.assert_not_called()
        deploy_manager.open_inference_service.assert_called_once_with(
            service_url,
            project="test-deploy",
            served_name="yolo26n-cls",
            local_port=8080,
            use_port_forward=False,
        )
        deploy_manager.stop_run.assert_called_once_with("test-deploy", run_id)

    def test_environment_flag_parses_explicit_values(self) -> None:
        with patch.dict(os.environ, {"OPEN_INFERENCE_USE_PORT_FORWARD": "false"}):
            self.assertFalse(
                example._environment_flag("OPEN_INFERENCE_USE_PORT_FORWARD", default=True)
            )
        with patch.dict(os.environ, {"OPEN_INFERENCE_USE_PORT_FORWARD": "yes"}):
            self.assertTrue(
                example._environment_flag("OPEN_INFERENCE_USE_PORT_FORWARD", default=False)
            )

if __name__ == "__main__":
    unittest.main()
