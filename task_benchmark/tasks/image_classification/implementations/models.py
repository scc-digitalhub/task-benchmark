# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

from task_benchmark.abstract import BaseImplementation


@dataclass
class Prediction:
    label: str
    score: float


class ImageClassificationModel(BaseImplementation):
    predicts_wnid: bool = False

    @abstractmethod
    def predict_batch(
        self,
        inputs: list[bytes],
        top_k: int = 5,
    ) -> list[list[Prediction]]:
        pass
