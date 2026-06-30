"""
Minimal example: always predict the first available class.

Useful for testing the benchmark pipeline without a real model.
"""

from __future__ import annotations

from ..models import (
    ImageClassificationModel,
    Prediction,
)


class AlwaysFirstClassModel(
    ImageClassificationModel
):
    """Always predicts the first class with score 1.0."""

    predicts_wnid = True

    def __init__(
        self,
        class_descriptions: dict[str, str],
        device: str = "cpu",
    ) -> None:
        _ = device

        self.first_class = next(
            iter(class_descriptions.keys())
        )

    def predict_batch(
        self,
        images: list[bytes],
        top_k: int = 5,
    ) -> list[list[Prediction]]:
        """Return first class for every image."""

        return [
            [
                Prediction(
                    label=self.first_class,
                    score=1.0,
                )
            ]
            for _ in images
        ]


def create_model(
    class_descriptions: dict[str, str],
    device: str,
) -> ImageClassificationModel:
    return AlwaysFirstClassModel(
        class_descriptions=class_descriptions,
        device=device,
    )
