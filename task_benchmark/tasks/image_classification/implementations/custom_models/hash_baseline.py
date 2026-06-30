from __future__ import annotations

import hashlib

from ..models import (
    ImageClassificationModel,
    Prediction,
)


class HashBaselineModel(
    ImageClassificationModel
):
    """
    Deterministic demo model.

    It always returns class labels from the provided dataset classes and maps
    each image to a stable top-k ranking based on image bytes hash.
    """

    predicts_wnid = True

    def __init__(
        self,
        class_descriptions: dict[str, str],
        device: str = "cpu",
    ) -> None:
        _ = device

        self.wnids = sorted(
            class_descriptions.keys()
        )

        if not self.wnids:
            raise ValueError(
                "class_descriptions cannot be empty"
            )

    def predict_batch(
        self,
        images: list[bytes],
        top_k: int = 5,
    ) -> list[list[Prediction]]:
        predictions: list[list[Prediction]] = []

        for image_bytes in images:
            digest = hashlib.sha256(
                image_bytes
            ).digest()

            start_idx = int.from_bytes(
                digest[:4],
                byteorder="big",
            ) % len(self.wnids)

            per_image: list[Prediction] = []

            for rank in range(
                min(top_k, len(self.wnids))
            ):
                wnid = self.wnids[
                    (start_idx + rank)
                    % len(self.wnids)
                ]

                per_image.append(
                    Prediction(
                        label=wnid,
                        score=1.0 / float(rank + 1),
                    )
                )

            predictions.append(per_image)

        return predictions


def create_model(
    class_descriptions: dict[str, str],
    device: str,
) -> ImageClassificationModel:
    return HashBaselineModel(
        class_descriptions=class_descriptions,
        device=device,
    )
