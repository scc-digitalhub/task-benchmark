from __future__ import annotations

import unittest
from unittest.mock import Mock

import torch

from task_benchmark.implementations.invoker.image_classification import (
    ImageClassificationInvoker,
    OpenInferenceImageClassificationInvoker,
)
from task_benchmark.implementations.invoker.open_inference import OpenInferenceInvoker
from task_benchmark.implementations.data_loader.image_classification import (
    ImageClassificationDataLoader,
)
from task_benchmark.implementations.evaluator.image_classification import (
    Top1ImageClassificationEvaluator,
)


class OpenInferenceImageClassificationInvokerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = Mock(spec=OpenInferenceInvoker)
        self.invoker = OpenInferenceImageClassificationInvoker(
            self.client, "pixels", "scores", {0: "cat", 1: "dog"}
        )

    def test_selects_output_and_normalizes_shape(self) -> None:
        scores = torch.tensor([0.2, 0.8])
        self.client.invoke.return_value = {
            "embedding": torch.zeros(4), "scores": scores
        }
        image = torch.zeros(1, 3, 32, 32)

        result = self.invoker.invoke(image)

        self.assertIsInstance(self.invoker, ImageClassificationInvoker)
        self.client.invoke.assert_called_once_with({"pixels": image})
        torch.testing.assert_close(result, scores.unsqueeze(0))
        self.assertEqual(self.invoker.label_for(1), "dog")

    def test_preserves_batched_logits(self) -> None:
        scores = torch.tensor([[-2.0, 3.0]])
        self.client.invoke.return_value = {"scores": scores}
        self.assertIs(self.invoker.invoke(torch.zeros(3, 2, 2)), scores)

    def test_rejects_invalid_outputs(self) -> None:
        invalid_outputs = [
            {},
            {"scores": torch.zeros(2, 2)},
            {"scores": torch.zeros(1, 3)},
            {"scores": torch.tensor([0, 1])},
            {"scores": torch.tensor([float("nan"), 1.0])},
            {"scores": torch.tensor([float("inf"), 1.0])},
        ]
        for outputs in invalid_outputs:
            with self.subTest(outputs=outputs):
                self.client.invoke.return_value = outputs
                with self.assertRaises(ValueError):
                    self.invoker.invoke(torch.zeros(3, 2, 2))

    def test_rejects_incomplete_label_mapping(self) -> None:
        with self.assertRaises(ValueError):
            OpenInferenceImageClassificationInvoker(
                self.client, "pixels", "scores", {1: "dog"}
            )

    def test_closes_wrapped_client(self) -> None:
        self.invoker.close()
        self.client.close.assert_called_once_with()

    def test_adapter_can_be_evaluated_with_top1(self) -> None:
        class DatasetLoader(ImageClassificationDataLoader):
            def next(self) -> tuple[str, torch.Tensor]:
                return "image-1", torch.zeros(1, 3, 32, 32)

            def labels_for(self, image_id: str) -> tuple[str, ...]:
                return ("dog",)

        loader = DatasetLoader("sample", "./data")
        evaluator = Top1ImageClassificationEvaluator(loader, self.invoker)
        self.client.invoke.return_value = {"scores": torch.tensor([0.1, 0.9])}
        image_id, image = loader.next()

        self.assertEqual(evaluator.evaluate(image_id, self.invoker.invoke(image)), 1.0)
        self.client.invoke.return_value = {"scores": torch.tensor([0.9, 0.1])}
        self.assertEqual(evaluator.evaluate(image_id, self.invoker.invoke(image)), 0.0)


if __name__ == "__main__":
    unittest.main()