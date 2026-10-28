# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
import types
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATIONS_DIR = PROJECT_ROOT / "task_benchmark" / "implementations"
INVOKER_PACKAGE_DIR = IMPLEMENTATIONS_DIR / "invoker"
INVOKER_PATH = (
    INVOKER_PACKAGE_DIR / "open_inference" / "invoker.py"
)


def load_invoker_module():
    package_paths = {
        "task_benchmark": PROJECT_ROOT / "task_benchmark",
        "task_benchmark.implementations": IMPLEMENTATIONS_DIR,
        "task_benchmark.implementations.invoker": INVOKER_PACKAGE_DIR,
    }
    package_modules = {}
    for package_name, package_path in package_paths.items():
        package_module = types.ModuleType(package_name)
        package_module.__path__ = [str(package_path)]
        package_modules[package_name] = package_module

    base_module_name = "task_benchmark.implementations.invoker.base_invoker"
    base_spec = importlib.util.spec_from_file_location(
        base_module_name,
        INVOKER_PACKAGE_DIR / "base_invoker.py",
    )
    module_spec = importlib.util.spec_from_file_location(
        "open_inference_invoker_under_test",
        INVOKER_PATH,
    )
    if (
        base_spec is None
        or base_spec.loader is None
        or module_spec is None
        or module_spec.loader is None
    ):
        raise RuntimeError("Could not load invoker modules")
    base_module = importlib.util.module_from_spec(base_spec)
    invoker_module = importlib.util.module_from_spec(module_spec)

    with patch.dict(sys.modules, package_modules):
        sys.modules[base_module_name] = base_module
        base_spec.loader.exec_module(base_module)
        module_spec.loader.exec_module(invoker_module)
    return invoker_module


INVOKER_MODULE = load_invoker_module()
OpenInferenceInvoker = INVOKER_MODULE.OpenInferenceInvoker


class OpenInferenceInvokerTests(unittest.TestCase):
    service_url = "http://localhost:8000/v2/models/test-model/infer"

    def test_inherits_from_generic_invoker(self) -> None:
        self.assertTrue(issubclass(OpenInferenceInvoker, INVOKER_MODULE.Invoker))

    @staticmethod
    def http_response(payload: dict) -> BytesIO:
        return BytesIO(json.dumps(payload).encode("utf-8"))

    def test_single_tensor_returns_first_output_tensor(self) -> None:
        metadata = {
            "inputs": [{"name": "input", "datatype": "FP32"}],
            "outputs": [{"name": "scores"}, {"name": "embedding"}],
        }
        response = {
            "outputs": [
                {
                    "name": "scores",
                    "datatype": "FP32",
                    "shape": [1, 2],
                    "data": [0.2, 0.8],
                }
            ]
        }

        with patch.object(
            INVOKER_MODULE.request,
            "urlopen",
            side_effect=[
                self.http_response(metadata),
                self.http_response(response),
            ],
        ) as urlopen:
            result = OpenInferenceInvoker(self.service_url).invoke(
                torch.tensor([[1.0]])
            )

        self.assertIsInstance(result, torch.Tensor)
        torch.testing.assert_close(result, torch.tensor([[0.2, 0.8]]))
        request_payload = json.loads(urlopen.call_args_list[1].args[0].data)
        self.assertEqual(request_payload["outputs"], [{"name": "scores"}])

    def test_mapping_invokes_multiple_inputs_and_returns_all_outputs(self) -> None:
        metadata = {
            "inputs": [
                {"name": "image", "datatype": "FP32"},
                {"name": "scale", "datatype": "INT64"},
            ],
            "outputs": [{"name": "scores"}, {"name": "embedding"}],
        }
        response = {
            "outputs": [
                {
                    "name": "scores",
                    "datatype": "FP32",
                    "shape": [1, 2],
                    "data": [0.2, 0.8],
                },
                {
                    "name": "embedding",
                    "datatype": "INT64",
                    "shape": [1],
                    "data": [7],
                },
            ]
        }

        with patch.object(
            INVOKER_MODULE.request,
            "urlopen",
            side_effect=[
                self.http_response(metadata),
                self.http_response(response),
            ],
        ) as urlopen:
            result = OpenInferenceInvoker(self.service_url).invoke(
                {
                    "image": torch.tensor([[1.0]]),
                    "scale": torch.tensor([2]),
                }
            )

        self.assertEqual(set(result), {"scores", "embedding"})
        torch.testing.assert_close(result["scores"], torch.tensor([[0.2, 0.8]]))
        self.assertEqual(result["embedding"].tolist(), [7])

        request_payload = json.loads(urlopen.call_args_list[1].args[0].data)
        self.assertEqual(
            [item["name"] for item in request_payload["inputs"]],
            ["image", "scale"],
        )
        self.assertEqual(
            [item["name"] for item in request_payload["outputs"]],
            ["scores", "embedding"],
        )

    def test_mapping_rejects_input_names_that_do_not_match_metadata(self) -> None:
        metadata = {
            "inputs": [{"name": "image", "datatype": "FP32"}],
            "outputs": [{"name": "scores"}],
        }

        with patch.object(
            INVOKER_MODULE.request,
            "urlopen",
            return_value=self.http_response(metadata),
        ):
            invoker = OpenInferenceInvoker(self.service_url)
            with self.assertRaisesRegex(ValueError, "Input names do not match"):
                invoker.invoke({"unexpected": torch.tensor([1.0])})

    def test_grpc_mapping_invokes_all_inputs_and_returns_all_outputs(self) -> None:
        metadata = {
            "inputs": [
                {"name": "image", "datatype": "FP32"},
                {"name": "scale", "datatype": "INT64"},
            ],
            "outputs": [{"name": "scores"}, {"name": "embedding"}],
        }
        output_values = {
            "scores": np.array([[0.2, 0.8]], dtype=np.float32),
            "embedding": np.array([7], dtype=np.int64),
        }

        class FakeResult:
            def as_numpy(self, name: str) -> np.ndarray:
                return output_values[name]

        class FakeGrpcClient:
            def get_model_metadata(self, model_name: str) -> dict:
                self.model_name = model_name
                return metadata

            def infer(self, model_name: str, inputs: list, outputs: list):
                self.model_name = model_name
                self.input_count = len(inputs)
                self.output_count = len(outputs)
                return FakeResult()

        fake_client = FakeGrpcClient()
        grpc_url = "grpc://localhost:8001/v2/models/test-model/infer"
        with patch(
            "tritonclient.grpc.InferenceServerClient",
            return_value=fake_client,
        ):
            invoker = OpenInferenceInvoker(grpc_url)
            result = invoker.invoke(
                {
                    "image": torch.tensor([[1.0]]),
                    "scale": torch.tensor([2]),
                }
            )

        self.assertEqual(fake_client.model_name, "test-model")
        self.assertEqual(fake_client.input_count, 2)
        self.assertEqual(fake_client.output_count, 2)
        self.assertEqual(set(result), {"scores", "embedding"})
        torch.testing.assert_close(result["scores"], torch.tensor([[0.2, 0.8]]))
        self.assertEqual(result["embedding"].tolist(), [7])


if __name__ == "__main__":
    unittest.main()