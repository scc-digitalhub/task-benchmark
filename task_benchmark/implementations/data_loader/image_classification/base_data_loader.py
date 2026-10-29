from __future__ import annotations

from abc import abstractmethod

from torch import Tensor

from task_benchmark.implementations.data_loader.data_loader import DataLoader


class ImageClassificationDataLoader(DataLoader):
    """Abstract source of image tensors and their ground-truth labels."""

    @abstractmethod
    def next(self) -> tuple[str, Tensor]:
        """Return the next image identifier and tensor."""
        raise NotImplementedError

    @abstractmethod
    def labels_for(self, image_id: str) -> tuple[str, ...]:
        """Return the ground-truth labels associated with an image identifier."""
        raise NotImplementedError