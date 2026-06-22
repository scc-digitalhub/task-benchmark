from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Prediction:
    label: str
    score: float


class ImageClassificationModel(ABC):

    predicts_wnid: bool = False

    @abstractmethod
    def predict_batch(
        self,
        images: list[bytes],
        top_k: int = 5,
    ) -> list[list[Prediction]]:
        pass