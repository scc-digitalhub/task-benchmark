from __future__ import annotations

from abc import abstractmethod

from torch import Tensor

from task_benchmark.implementations.invoker.base_invoker import Invoker


class ImageClassificationInvoker(Invoker[Tensor, Tensor]):
    """Abstract image classifier with a model-class-to-label mapping."""

    @abstractmethod
    def label_for(self, class_id: int) -> str:
        """Return the label associated with a model output class index."""
        raise NotImplementedError