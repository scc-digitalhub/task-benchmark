# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATIONS_DIR = PROJECT_ROOT / "task_benchmark" / "implementations"
INVOKER_PACKAGE_DIR = IMPLEMENTATIONS_DIR / "invoker"
INVOKER_PATH = (
    INVOKER_PACKAGE_DIR / "image_classification" / "huggingface.py"
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
        "huggingface_image_invoker_under_test",
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
HuggingFaceImageClassificationInvoker = (
    INVOKER_MODULE.HuggingFaceImageClassificationInvoker
)


class HuggingFaceImageClassificationInvokerTests(unittest.TestCase):
    def test_inherits_from_generic_invoker(self) -> None:
        self.assertTrue(
            issubclass(
                HuggingFaceImageClassificationInvoker,
                INVOKER_MODULE.ImageClassificationInvoker,
            )
        )

    def test_classifier_requires_label_for(self) -> None:
        class IncompleteInvoker(INVOKER_MODULE.ImageClassificationInvoker):
            def invoke(self, test_data: torch.Tensor) -> torch.Tensor:
                return test_data

        with self.assertRaises(TypeError):
            IncompleteInvoker()

    def test_label_for_supports_integer_and_string_mapping_keys(self) -> None:
        processor = Mock()
        model = Mock()
        model.config = SimpleNamespace(id2label={0: "cat", "1": "dog"})
        model.to = Mock(return_value=model)
        model.eval = Mock(return_value=model)
        transformers = self.transformers_module(
            Mock(return_value=processor),
            Mock(return_value=model),
        )

        with patch.dict(sys.modules, {"transformers": transformers}):
            invoker = HuggingFaceImageClassificationInvoker("org/image-model")

        self.assertEqual(invoker.label_for(0), "cat")
        self.assertEqual(invoker.label_for(1), "dog")
        self.assertEqual(invoker.label_for(2), "2")
        with self.assertRaises(TypeError):
            invoker.label_for(True)

    def transformers_module(self, processor: Mock, model: Mock) -> types.ModuleType:
        transformers = types.ModuleType("transformers")
        transformers.AutoImageProcessor = SimpleNamespace(
            from_pretrained=processor
        )
        transformers.AutoModelForImageClassification = SimpleNamespace(
            from_pretrained=model
        )
        return transformers

    def test_invoke_preprocesses_image_and_returns_logits(self) -> None:
        expected_logits = torch.tensor([[0.2, 0.8]])
        processed_inputs = {"pixel_values": torch.zeros((1, 3, 224, 224))}
        processor = Mock(return_value=processed_inputs)
        processor_loader = Mock(return_value=processor)
        model = Mock(return_value=SimpleNamespace(logits=expected_logits))
        model.to = Mock(return_value=model)
        model.eval = Mock(return_value=model)
        model_loader = Mock(return_value=model)
        transformers = self.transformers_module(
            processor_loader,
            model_loader,
        )

        with patch.dict(sys.modules, {"transformers": transformers}):
            invoker = HuggingFaceImageClassificationInvoker(
                "org/image-model",
                device="cpu",
                model_kwargs={"revision": "main"},
                processor_kwargs={"use_fast": True},
            )
            image = torch.rand((3, 64, 64))
            result = invoker.invoke(image)

        processor_loader.assert_called_once_with("org/image-model", use_fast=True)
        model_loader.assert_called_once_with(
            "org/image-model",
            revision="main",
        )
        model.to.assert_called_once_with(torch.device("cpu"))
        model.eval.assert_called_once_with()
        processor.assert_called_once()
        processor_call = processor.call_args.kwargs
        self.assertEqual(processor_call["images"].device.type, "cpu")
        self.assertEqual(processor_call["return_tensors"], "pt")
        self.assertIs(processor_call["do_rescale"], False)
        model.assert_called_once_with(**processed_inputs)
        self.assertIsInstance(result, torch.Tensor)
        self.assertIs(result, expected_logits)
        self.assertFalse(result.requires_grad)

    def test_rejects_non_tensor_or_invalid_image_dimensions(self) -> None:
        processor = Mock()
        model = Mock()
        transformers = self.transformers_module(processor, model)

        with patch.dict(sys.modules, {"transformers": transformers}):
            invoker = HuggingFaceImageClassificationInvoker("org/image-model")
            with self.assertRaises(TypeError):
                invoker.invoke("image")
            with self.assertRaisesRegex(ValueError, "CHW or BCHW"):
                invoker.invoke(torch.zeros((3, 32)))


if __name__ == "__main__":
    unittest.main()