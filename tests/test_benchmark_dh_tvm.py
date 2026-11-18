from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from contextlib import nullcontext
from io import StringIO
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
    def test_extracts_run_id_from_url_path_host_or_query(self) -> None:
        run_id = "cbe0097faab5492895f218158d367747"

        self.assertEqual(
            example.extract_run_id(f"https://host.example/runs/{run_id}/service"),
            run_id,
        )
        self.assertEqual(
            example.extract_run_id(f"https://{run_id}.svc.example/service"),
            run_id,
        )
        self.assertEqual(
            example.extract_run_id(f"https://host.example/service?run_id={run_id}"),
            run_id,
        )

    def test_extracts_alphanumeric_tvmserve_run_id_from_core_service_url(self) -> None:
        run_id = "54ab026c9b5649378ee2c001e579d819"
        service_url = f"s-tvmserve-{run_id}.dev-platform:8080"

        self.assertEqual(example.extract_run_id(service_url), run_id)
        self.assertEqual(
            example.open_inference_base_url(service_url, "yolo26n-cls"),
            f"http://s-tvmserve-{run_id}.dev-platform:8080/v2/models/yolo26n-cls",
        )
        non_hex_run_id = "g4ab026c9b5649378ee2c001e579d819"
        self.assertEqual(
            example.extract_run_id(
                f"s-tvmserve-{non_hex_run_id}.dev-platform:8080"
            ),
            non_hex_run_id,
        )

    def test_extract_run_id_fails_when_url_has_no_run_identifier(self) -> None:
        with self.assertRaisesRegex(ValueError, "Could not extract"):
            example.extract_run_id("https://service.example")

    def test_open_inference_base_url_normalizes_service_url(self) -> None:
        self.assertEqual(
            example.open_inference_base_url(
                "https://service.example/prefix",
                "yolo26n-cls",
            ),
            "https://service.example/prefix/v2/models/yolo26n-cls",
        )
        self.assertEqual(
            example.open_inference_base_url(
                "https://service.example/v2/models/yolo26n-cls/infer",
                "ignored-default",
            ),
            "https://service.example/v2/models/yolo26n-cls",
        )

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
        with patch.object(
            example,
            "benchmark_service",
            side_effect=[Path("first.json"), Path("second.json")],
        ) as benchmark_service:
            result = example.benchmark_deployments(
                deployments,
                config,
                max_samples=5,
                local_port=8080,
            )

        self.assertEqual(result, [Path("first.json"), Path("second.json")])
        self.assertEqual(benchmark_service.call_args_list, [
            call(
                service_url="https://service-a.example",
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
        process = Mock()
        process.poll.return_value = None

        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as output_directory:
            os.chdir(output_directory)
            try:
                with (
                    patch.object(
                        example,
                        "port_forward",
                        return_value=nullcontext((process, StringIO())),
                    ) as port_forward,
                    patch.object(
                        example.request,
                        "urlopen",
                        return_value=StringIO(json.dumps(metadata)),
                    ) as urlopen,
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
            urlopen.call_args.args[0],
            "http://localhost:8080/v2/models/yolo26n-cls",
        )
        self.assertEqual(
            invoker.call_args.args[0],
            "http://localhost:8080/v2/models/yolo26n-cls/infer",
        )
        port_forward.assert_called_once_with("test-deploy", run_id, 8080)
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
        with (
            patch.object(example, "port_forward") as port_forward,
            patch.object(
                example.request,
                "urlopen",
                return_value=StringIO("{}"),
            ) as urlopen,
        ):
            with self.assertRaisesRegex(ValueError, "Expected the YOLO FP32 input"):
                example.benchmark_service(
                    service_url=service_url,
                    project="test-deploy",
                    profile="cpu",
                    target_architecture="amd64",
                    served_name="yolo26n-cls",
                    max_samples=5,
                    local_port=8080,
                    use_port_forward=False,
                )

        urlopen.assert_called_once_with(
            "http://s-tvmserve-cbe0097faab5492895f218158d367747."
            "dev-platform:8080/v2/models/yolo26n-cls",
            timeout=60.0,
        )
        port_forward.assert_not_called()

    def test_environment_flag_parses_explicit_values(self) -> None:
        with patch.dict(os.environ, {"OPEN_INFERENCE_USE_PORT_FORWARD": "false"}):
            self.assertFalse(
                example._environment_flag("OPEN_INFERENCE_USE_PORT_FORWARD", default=True)
            )
        with patch.dict(os.environ, {"OPEN_INFERENCE_USE_PORT_FORWARD": "yes"}):
            self.assertTrue(
                example._environment_flag("OPEN_INFERENCE_USE_PORT_FORWARD", default=False)
            )

    def test_port_forward_starts_dhcli_with_project_run_id_and_local_port(self) -> None:
        run_id = "cbe0097faab5492895f218158d367747"
        process = Mock()
        process.poll.return_value = None
        executable = Path(r"C:\tools\dhcli.exe")

        with (
            patch.object(example, "resolve_dhcli_path", return_value=executable),
            patch.object(example.subprocess, "Popen", return_value=process) as popen,
        ):
            with example.port_forward("test-deploy", run_id, 8080):
                pass

        args, kwargs = popen.call_args
        self.assertEqual(args[0], [
            str(executable),
            "port-forward",
            "-p",
            "test-deploy",
            run_id,
            "-l",
            "8080",
        ])
        self.assertIs(kwargs["stderr"], example.subprocess.STDOUT)
        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=5)


if __name__ == "__main__":
    unittest.main()
