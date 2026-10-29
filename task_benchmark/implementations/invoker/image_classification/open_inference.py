from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from task_benchmark.implementations.invoker.open_inference import OpenInferenceInvoker

from .base_invoker import ImageClassificationInvoker


class OpenInferenceImageClassificationInvoker(ImageClassificationInvoker):
    """Adapt a named Open Inference output of class scores to [1, classes].

    Inputs must already satisfy the remote model's preprocessing and shape
    requirements. Class indices must be consecutive and match output order.
    """

    def __init__(
        self,
        invoker: OpenInferenceInvoker,
        input_name: str,
        output_name: str,
        labels: Mapping[int, str],
    ) -> None:
        if not input_name or not output_name:
            raise ValueError("input_name and output_name must not be empty")
        if (
            not labels
            or any(type(class_id) is not int for class_id in labels)
            or set(labels) != set(range(len(labels)))
        ):
            raise ValueError("labels must map consecutive indices starting at zero")
        if any(not isinstance(label, str) or not label.strip() for label in labels.values()):
            raise ValueError("labels must contain non-empty strings")
        self.invoker = invoker
        self.input_name = input_name
        self.output_name = output_name
        self.labels = dict(labels)

    def label_for(self, class_id: int) -> str:
        """Return the explicitly configured label for a class index."""
        if not isinstance(class_id, int) or isinstance(class_id, bool):
            raise TypeError("class_id must be an integer")
        return self.labels[class_id]

    def invoke(self, test_data: Tensor) -> Tensor:
        """Return one image's logits or probabilities without changing scores."""
        if not isinstance(test_data, Tensor):
            raise TypeError("test_data must be a torch.Tensor")
        outputs = self.invoker.invoke({self.input_name: test_data})
        if self.output_name not in outputs:
            raise ValueError(f"Response is missing output '{self.output_name}'")
        scores = outputs[self.output_name]
        if not isinstance(scores, Tensor):
            raise TypeError("The selected output must be a torch.Tensor")
        if scores.ndim == 1:
            scores = scores.unsqueeze(0)
        if scores.ndim != 2 or scores.shape[0] != 1:
            raise ValueError("Class scores must have shape [classes] or [1, classes]")
        if scores.shape[1] != len(self.labels):
            raise ValueError("The number of class scores must match the label mapping")
        if not scores.is_floating_point() or not torch.isfinite(scores).all().item():
            raise ValueError("Class scores must be finite floating-point values")
        return scores

    def close(self) -> None:
        """Close the wrapped Open Inference client."""
        self.invoker.close()