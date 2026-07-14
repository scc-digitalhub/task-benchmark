from __future__ import annotations

import hashlib

from task_benchmark.implementations import implementation_registry
from task_benchmark.tasks.image_classification import (
    ImageClassificationModel,
    Prediction,
)


class HashBaselineImageClassifier(ImageClassificationModel):
    """Deterministic demo classifier based on image hash."""

    predicts_wnid = True

    def __init__(
        self,
        model_name: str = "",
        device: str = "cpu",
        class_descriptions: dict[str, str] | None = None,
    ) -> None:
        _ = model_name
        _ = device

        self.wnids = sorted((class_descriptions or {}).keys())
        if not self.wnids:
            raise ValueError("class_descriptions cannot be empty")

    def predict_batch(
        self,
        inputs: list[bytes],
        top_k: int = 5,
    ) -> list[list[Prediction]]:
        predictions: list[list[Prediction]] = []

        for image_bytes in inputs:
            digest = hashlib.sha256(image_bytes).digest()
            start_idx = int.from_bytes(digest[:4], byteorder="big") % len(self.wnids)

            per_image: list[Prediction] = []
            for rank in range(min(top_k, len(self.wnids))):
                wnid = self.wnids[(start_idx + rank) % len(self.wnids)]
                per_image.append(
                    Prediction(label=wnid, score=1.0 / float(rank + 1))
                )

            predictions.append(per_image)

        return predictions


implementation_registry.register(
    "image-classification",
    "hash-baseline",
    HashBaselineImageClassifier,
)
