"""Template for a self-registering image classification implementation."""

from __future__ import annotations

from task_benchmark.implementations import implementation_registry
from task_benchmark.tasks.image_classification import (
    ImageClassificationModel,
    Prediction,
)


class MyCustomImageClassifier(ImageClassificationModel):
    predicts_wnid = True

    def __init__(
        self,
        model_name: str = "",
        device: str = "cpu",
        class_descriptions: dict[str, str] | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.class_descriptions = class_descriptions or {}

    def predict_batch(
        self,
        inputs: list[bytes],
        top_k: int = 5,
    ) -> list[list[Prediction]]:
        # Replace with real inference.
        _ = inputs
        _ = top_k
        return []


implementation_registry.register(
    "image-classification",
    "my-custom",
    MyCustomImageClassifier,
)
