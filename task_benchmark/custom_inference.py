# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class CustomPrediction:
    """
    Lightweight prediction object used by custom backends.
    """

    label: str
    score: float


class CustomInferenceModel(ABC):
    """
    Base class for custom (non task-inference) models.

    You can implement anything here: classic ML, handcrafted features,
    or a small neural network pipeline.
    """

    predicts_wnid: bool = True

    @abstractmethod
    def predict_batch(
        self,
        images: list[bytes],
        top_k: int,
    ) -> list[list[CustomPrediction]]:
        """
        Return top-k predictions for each image in `images`.
        """


class HashBaselineModel(CustomInferenceModel):
    """
    Deterministic baseline that maps image bytes to class IDs.

    This is a placeholder model meant to be replaced by user code.
    """

    predicts_wnid = True

    def __init__(
        self,
        wnids: list[str],
        **_: Any,
    ) -> None:
        if not wnids:
            raise ValueError(
                "Cannot initialize custom model: no classes available."
            )

        self.wnids = sorted(set(wnids))

    def predict_batch(
        self,
        images: list[bytes],
        top_k: int,
    ) -> list[list[CustomPrediction]]:
        predictions: list[list[CustomPrediction]] = []

        for image_bytes in images:
            digest = hashlib.sha256(image_bytes).digest()
            start_idx = int.from_bytes(digest[:4], "big") % len(self.wnids)

            per_image: list[CustomPrediction] = []

            for rank in range(min(top_k, len(self.wnids))):
                idx = (start_idx + rank) % len(self.wnids)
                score = 1.0 / float(rank + 1)
                per_image.append(
                    CustomPrediction(label=self.wnids[idx], score=score)
                )

            predictions.append(per_image)

        return predictions


def create_custom_model(
    custom_model_name: str,
    class_descriptions: dict[str, str],
    device: str,
) -> CustomInferenceModel:
    """
    Factory for custom models.

    Add your own branches/classes here to register custom implementations.
    """

    _ = device

    if custom_model_name == "hash-baseline":
        return HashBaselineModel(wnids=list(class_descriptions.keys()))

    raise ValueError(
        "Unknown custom model "
        f"'{custom_model_name}'. "
        "Edit custom_inference.py and extend create_custom_model()."
    )
