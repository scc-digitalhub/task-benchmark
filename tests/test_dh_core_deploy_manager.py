from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from task_benchmark.impl.deploy_manager import DHCoreDeployManager
from task_benchmark.impl.deploy_manager import dh_core
import yaml


class DHCoreDeployManagerTests(unittest.TestCase):
    def test_runs_source_function_once_per_profile_and_returns_urls(self) -> None:
        source_spec = {
            "task": "python+serve://my-project/task-id",
            "function": "python://my-project/model-server:version-123",
            "profile": "original-profile",
            "resources": {"cpu": "1", "mem": "2Gi"},
            "envs": [{"name": "MODEL", "value": "model-a"}],
        }
        source_run = SimpleNamespace(
            spec=SimpleNamespace(to_dict=lambda: source_spec),
            kind="python+serve",
            extensions=[{"name": "test-extension"}],
        )
        function = Mock()
        function.run.side_effect = [
            SimpleNamespace(status=SimpleNamespace(
                service={"url": "http://service-a.example"}
            )),
            SimpleNamespace(status=SimpleNamespace(
                service={"url": "http://service-b.example"}
            )),
        ]
        manager = DHCoreDeployManager()

        with (
            patch.object(dh_core.dh, "get_run", return_value=source_run) as get_run,
            patch.object(dh_core.dh, "get_function", return_value=function) as get_function,
        ):
            urls = manager.deploy("my-project", "source-run-id", ["small", "large"])

        get_run.assert_called_once_with(
            "source-run-id",
            project="my-project",
        )
        get_function.assert_called_once_with(
            "model-server",
            project="my-project",
            entity_id="version-123",
        )
        self.assertEqual(
            [call.args[0] for call in function.run.call_args_list],
            ["serve", "serve"],
        )
        self.assertEqual(
            [call.kwargs["profile"] for call in function.run.call_args_list],
            ["small", "large"],
        )
        for call in function.run.call_args_list:
            self.assertEqual(call.kwargs["resources"], source_spec["resources"])
            self.assertEqual(call.kwargs["envs"], source_spec["envs"])
            self.assertEqual(call.kwargs["extensions"], source_run.extensions)
            self.assertTrue(call.kwargs["wait"])
        self.assertEqual(
            urls,
            ["http://service-a.example", "http://service-b.example"],
        )

    def test_empty_profiles_do_not_load_sdk(self) -> None:
        manager = DHCoreDeployManager()
        with patch.object(dh_core.dh, "get_run") as get_run:
            self.assertEqual(manager.deploy("my-project", "source-run-id", []), [])
        get_run.assert_not_called()

    def test_rejects_source_run_without_function(self) -> None:
        source_run = SimpleNamespace(
            spec=SimpleNamespace(to_dict=lambda: {"task": "python+serve://task-id"}),
            kind="python+serve",
        )
        manager = DHCoreDeployManager()
        with patch.object(dh_core.dh, "get_run", return_value=source_run):
            with self.assertRaisesRegex(ValueError, "does not reference a function"):
                manager.deploy("my-project", "source-run-id", ["small"])

    def test_requires_service_url_for_each_created_run(self) -> None:
        source_run = SimpleNamespace(
            spec=SimpleNamespace(to_dict=lambda: {
                "function": "python://my-project/model-server:version-123",
            }),
            kind="python+serve",
        )
        function = Mock()
        function.run.return_value = SimpleNamespace(
            status=SimpleNamespace(service=None)
        )
        manager = DHCoreDeployManager()
        with (
            patch.object(dh_core.dh, "get_run", return_value=source_run),
            patch.object(dh_core.dh, "get_function", return_value=function),
        ):
            with self.assertRaisesRegex(RuntimeError, "did not expose a service URL"):
                manager.deploy("my-project", "source-run-id", ["small"])

    def test_validates_project_run_and_profiles(self) -> None:
        for args in (
            ("", "run-1", ["small"]),
            ("project", " ", ["small"]),
            ("project", "run-1", "small"),
            ("project", "run-1", [""]),
        ):
            with self.subTest(args=args):
                with self.assertRaises((TypeError, ValueError)):
                    DHCoreDeployManager().deploy(*args)

    def test_tvm_builds_compiles_and_serves_models_by_architecture(self) -> None:
        config = {
            "project": "test-tvm",
            "function": "yolo26n-cls",
            "model": "store://test-yolo/model/onnx/yolo26n-cls:model-id",
            "served_name": "yolo26n-cls",
            "build-resources": {"cpu": 1, "mem": "2Gi"},
            "targets": [
                {
                    "target_architecture": "arm64",
                    "complile-resources": {"cpu": 1, "mem": "2Gi"},
                },
                {
                    "target_architecture": "amd64",
                    "compile-resources": {"cpu": 2, "mem": "4Gi"},
                },
            ],
            "nodes": [
                {
                    "target_architecture": "arm64",
                    "profile": "jetson-orin",
                    "run-resources": {"cpu": 1, "mem": "512Mi"},
                },
                {
                    "target_architecture": "amd64",
                    "profile": "cpu",
                    "served_name": "yolo26n-cls-amd64",
                    "run-resources": {"cpu": 2, "mem": "1Gi"},
                },
            ],
        }
        build_run = SimpleNamespace(
            status=SimpleNamespace(state="COMPLETED"),
            refresh=Mock(),
        )
        compile_arm = SimpleNamespace(
            metadata={"so_model": "store://test-tvm/model/tvm-so/yolo-arm:arm-id"},
            status=SimpleNamespace(state="COMPLETED"),
            refresh=Mock(),
        )
        compile_amd = SimpleNamespace(
            metadata=SimpleNamespace(
                to_dict=lambda: {
                    "so_model": "store://test-tvm/model/tvm-so/yolo-amd64:amd-id"
                }
            ),
            status=SimpleNamespace(state="COMPLETED"),
            refresh=Mock(),
        )
        serve_arm = SimpleNamespace(
            status=SimpleNamespace(state="RUNNING", service={"url": "http://arm64.example"}),
            refresh=Mock(),
        )
        serve_amd = SimpleNamespace(
            status=SimpleNamespace(state="RUNNING", service={"url": "http://amd64.example"}),
            refresh=Mock(),
        )
        main_function = Mock()
        main_function.run.side_effect = [
            build_run,
            compile_arm,
            compile_amd,
            serve_arm,
            serve_amd,
        ]
        project = SimpleNamespace(name="test-tvm")
        manager = DHCoreDeployManager()

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "tvm.yml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with (
                patch.object(dh_core.dh, "get_project", return_value=project) as get_project,
                patch.object(
                    dh_core.dh,
                    "new_function",
                    return_value=main_function,
                ) as new_function,
            ):
                service_urls = manager.deployTVM(config_path)

        get_project.assert_called_once_with("test-tvm")
        new_function.assert_called_once_with(
            project="test-tvm",
            name="yolo26n-cls",
            kind="tvm",
            format="onnx",
            model=config["model"],
        )
        calls = main_function.run.call_args_list
        self.assertEqual([call.args[0] for call in calls], [
            "build",
            "compile",
            "compile",
            "serve",
            "serve",
        ])
        self.assertEqual(calls[0].kwargs, {
            "resources": {"cpu": 1, "mem": "2Gi"},
            "wait": True,
            "log_info": False,
        })
        self.assertEqual(calls[1].kwargs["target_architecture"], "arm64")
        self.assertEqual(calls[1].kwargs["tag"], "arm64")
        self.assertEqual(calls[1].kwargs["resources"], {"cpu": 1, "mem": "2Gi"})
        self.assertEqual(calls[2].kwargs["target_architecture"], "amd64")
        self.assertEqual(calls[2].kwargs["resources"], {"cpu": 2, "mem": "4Gi"})
        self.assertEqual(calls[3].kwargs["model_path"], compile_arm.metadata["so_model"])
        self.assertEqual(calls[3].kwargs["profile"], "jetson-orin")
        self.assertEqual(calls[3].kwargs["served_name"], "yolo26n-cls")
        self.assertEqual(calls[3].kwargs["resources"], {"cpu": 1, "mem": "512Mi"})
        self.assertEqual(
            calls[4].kwargs["model_path"],
            "store://test-tvm/model/tvm-so/yolo-amd64:amd-id",
        )
        self.assertEqual(calls[4].kwargs["served_name"], "yolo26n-cls-amd64")
        self.assertEqual(service_urls, [
            ("jetson-orin", "arm64", "http://arm64.example"),
            ("cpu", "amd64", "http://amd64.example"),
        ])
        for run in (build_run, compile_arm, compile_amd, serve_arm, serve_amd):
            run.refresh.assert_called_once_with()

    def test_tvm_refreshes_function_when_compile_run_has_no_model_metadata(self) -> None:
        config = {
            "project": "test-tvm",
            "function": "model",
            "model": "store://test-tvm/model/onnx/model:model-id",
            "served_name": "model",
            "build-resources": {"cpu": 1},
            "targets": [{
                "target_architecture": "arm64",
                "complile-resources": {"cpu": 1},
            }],
            "nodes": [{
                "target_architecture": "arm64",
                "profile": "jetson-orin",
                "run-resources": {"cpu": 1},
            }],
        }
        build_run = SimpleNamespace(
            status=SimpleNamespace(state="COMPLETED"),
            refresh=Mock(),
        )
        compile_run = SimpleNamespace(
            metadata={},
            spec=SimpleNamespace(to_dict=lambda: {}),
            status=SimpleNamespace(state="COMPLETED"),
            refresh=Mock(),
        )
        service_run = SimpleNamespace(
            status=SimpleNamespace(
                state="RUNNING",
                service={"url": "http://service.example"},
            ),
            refresh=Mock(),
        )
        main_function = Mock()
        main_function.run.side_effect = [
            build_run,
            compile_run,
            service_run,
        ]
        main_function.spec.to_dict.return_value = {
            "so_model": "store://test-tvm/model/tvm-so/model:compiled-id"
        }
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "tvm.yml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with (
                patch.object(
                    dh_core.dh,
                    "get_project",
                    return_value=SimpleNamespace(name="test-tvm"),
                ),
                patch.object(dh_core.dh, "new_function", return_value=main_function),
            ):
                service_urls = DHCoreDeployManager().deployTVM(config_path)

        main_function.refresh.assert_called_once_with()
        self.assertEqual(
            main_function.run.call_args_list[2].kwargs["model_path"],
            "store://test-tvm/model/tvm-so/model:compiled-id",
        )
        self.assertEqual(
            service_urls,
            [("jetson-orin", "arm64", "http://service.example")],
        )
        build_run.refresh.assert_called_once_with()
        compile_run.refresh.assert_called_once_with()
        service_run.refresh.assert_called_once_with()

    def test_tvm_raises_when_build_run_is_in_error(self) -> None:
        config = {
            "project": "test-tvm",
            "function": "model",
            "model": "store://test-tvm/model/onnx/model:model-id",
            "served_name": "model",
            "build-resources": {"cpu": 1},
            "targets": [],
            "nodes": [],
        }
        build_run = SimpleNamespace(
            id="build-run-id",
            status=SimpleNamespace(state="ERROR", message="Build failed"),
            refresh=Mock(),
        )
        main_function = Mock()
        main_function.run.return_value = build_run
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "tvm.yml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with (
                patch.object(
                    dh_core.dh,
                    "get_project",
                    return_value=SimpleNamespace(name="test-tvm"),
                ),
                patch.object(dh_core.dh, "new_function", return_value=main_function),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"build run build-run-id failed with state ERROR: Build failed",
                ):
                    DHCoreDeployManager().deployTVM(config_path)

        build_run.refresh.assert_called_once_with()
        main_function.run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
