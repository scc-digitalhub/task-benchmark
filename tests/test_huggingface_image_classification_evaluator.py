# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from task_benchmark.implementations.data_loader.image_classification import (
    ImageClassificationDataLoader,
)
from task_benchmark.implementations.invoker.image_classification import (
    ImageClassificationInvoker,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATIONS_DIR = PROJECT_ROOT / "task_benchmark" / "implementations"
EVALUATOR_DIR = IMPLEMENTATIONS_DIR / "evaluator"
IMAGE_CLASSIFICATION_EVALUATOR_DIR = EVALUATOR_DIR / "image_classification"
BASE_EVALUATOR_NAME = "task_benchmark.implementations.evaluator.base_evaluator"
EVALUATOR_PATH = IMAGE_CLASSIFICATION_EVALUATOR_DIR / "top1.py"


def load_evaluator_class():
    package_paths = {
        "task_benchmark": PROJECT_ROOT / "task_benchmark",
        "task_benchmark.implementations": IMPLEMENTATIONS_DIR,
        "task_benchmark.implementations.evaluator": EVALUATOR_DIR,
        "task_benchmark.implementations.evaluator.image_classification": (
            IMAGE_CLASSIFICATION_EVALUATOR_DIR
        ),
    }
    package_modules = {}
    for package_name, package_path in package_paths.items():
        package_module = types.ModuleType(package_name)
        package_module.__path__ = [str(package_path)]
        package_modules[package_name] = package_module

    base_spec = importlib.util.spec_from_file_location(
        BASE_EVALUATOR_NAME,
        EVALUATOR_DIR / "base_evaluator.py",
    )
    evaluator_spec = importlib.util.spec_from_file_location(
        "top1_evaluator_under_test",
        EVALUATOR_PATH,
    )
    if (
        base_spec is None
        or base_spec.loader is None
        or evaluator_spec is None
        or evaluator_spec.loader is None
    ):
        raise RuntimeError("Could not load evaluator modules")
    base_module = importlib.util.module_from_spec(base_spec)
    evaluator_module = importlib.util.module_from_spec(evaluator_spec)

    with patch.dict(sys.modules, package_modules):
        sys.modules[BASE_EVALUATOR_NAME] = base_module
        base_spec.loader.exec_module(base_module)
        evaluator_spec.loader.exec_module(evaluator_module)

    return evaluator_module.Top1ImageClassificationEvaluator


Top1ImageClassificationEvaluator = load_evaluator_class()


class FakeDataLoader(ImageClassificationDataLoader):
    def next(self) -> tuple[str, torch.Tensor]:
        return "sample-1", torch.zeros(3, 1, 1)

    def labels_for(self, image_id: str) -> tuple[str, ...]:
        if image_id != "sample-1":
            raise KeyError(image_id)
        return ("cat", "pet")


class FakeInvoker(ImageClassificationInvoker):
    labels = {0: "dog", 1: "cat", 2: "bird"}

    def invoke(self, test_data: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[0.1, 0.9, 0.2]])

    def label_for(self, class_id: int) -> str:
        return self.labels.get(class_id, str(class_id))


class Top1ImageClassificationEvaluatorTests(unittest.TestCase):
    def test_example_evaluates_invoker_response(self) -> None:
        example_path = (
            PROJECT_ROOT / "examples" / "image_classification" / "huggingface"
            / "huggingface_image_classification.py"
        )
        spec = importlib.util.spec_from_file_location("huggingface_example", example_path)
        if spec is None or spec.loader is None:
            self.fail("Could not load Hugging Face example")
        example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(example)
        image = torch.zeros(3, 32, 32)
        logits = torch.tensor([[0.1, 0.9]])
        loader = Mock()
        loader.labels_for.return_value = ("cat",)
        invoker = Mock()
        invoker.invoke.return_value = logits
        invoker.label_for.return_value = "cat"
        samples = [("sample-1", image)]

        class ExampleLoader:
            def __iter__(self):
                return iter(samples)

            def labels_for(self, image_id: str) -> tuple[str, ...]:
                return loader.labels_for(image_id)

        example_loader = ExampleLoader()
        evaluator = Mock(wraps=Top1ImageClassificationEvaluator(example_loader, invoker))
        with tempfile.TemporaryDirectory() as cache_dir:
            with (
                patch.dict(os.environ, {
                    "HF_CACHE_DIR": cache_dir,
                    "DEVICE": "cpu",
                    "HF_MODEL_ID": "nateraw/vit-base-patch16-224-cifar10",
                }),
                patch.object(example, "HuggingFaceImageClassificationDataLoader", return_value=example_loader),
                patch.object(example, "HuggingFaceImageClassificationInvoker", return_value=invoker) as invoker_factory,
                patch.object(example, "Top1ImageClassificationEvaluator", return_value=evaluator) as factory,
                patch("builtins.print") as output,
            ):
                example.main()

        factory.assert_called_once_with(example_loader, invoker)
        model_kwargs = invoker_factory.call_args.kwargs["model_kwargs"]
        self.assertEqual(model_kwargs["id2label"][3], "cat")
        self.assertEqual(model_kwargs["id2label"][9], "truck")
        self.assertEqual(model_kwargs["label2id"]["cat"], 3)
        invoker.invoke.assert_called_once_with(image)
        evaluator.evaluate.assert_called_once_with("sample-1", logits)
        self.assertIn("top1_score=1.0", output.call_args.args[0])

    def setUp(self) -> None:
        self.evaluator = Top1ImageClassificationEvaluator(
            FakeDataLoader("test-dataset", "./data"),
            FakeInvoker(),
        )

    def test_returns_one_when_top_prediction_matches_annotation(self) -> None:
        score = self.evaluator.evaluate(
            "sample-1",
            torch.tensor([[0.1, 0.9, 0.2]]),
        )

        self.assertEqual(score, 1.0)

    def test_returns_zero_when_top_prediction_does_not_match(self) -> None:
        score = self.evaluator.evaluate(
            "sample-1",
            torch.tensor([[0.1, 0.2, 0.9]]),
        )

        self.assertEqual(score, 0.0)

    def test_rejects_responses_for_multiple_images(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one image"):
            self.evaluator.evaluate(
                "sample-1",
                torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
            )

    def test_propagates_missing_annotation_error(self) -> None:
        with self.assertRaisesRegex(KeyError, "sample-2"):
            self.evaluator.evaluate("sample-2", torch.tensor([0.1, 0.9]))


if __name__ == "__main__":
    unittest.main()