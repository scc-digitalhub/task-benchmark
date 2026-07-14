from __future__ import annotations

from task_benchmark.implementations import implementation_registry
from task_benchmark.tasks.image_classification import (
    ImageClassificationModel,
    Prediction,
)


class AlwaysFirstClassImageClassifier(ImageClassificationModel):
    """Always predicts the first available class."""

    predicts_wnid = True

    def __init__(
        self,
        model_name: str = "",
        device: str = "cpu",
        class_descriptions: dict[str, str] | None = None,
    ) -> None:
        _ = model_name
        _ = device

        classes = sorted((class_descriptions or {}).keys())
        if not classes:
            raise ValueError("class_descriptions cannot be empty")

        self.first_class = classes[0]

    def predict_batch(
        self,
        inputs: list[bytes],
        top_k: int = 5,
    ) -> list[list[Prediction]]:
        _ = top_k

        return [
            [Prediction(label=self.first_class, score=1.0)]
            for _ in inputs
        ]


implementation_registry.register(
    "image-classification",
    "always-first-class",
    AlwaysFirstClassImageClassifier,
)
