from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

from task_benchmark.implementations.evaluator.base_evaluator import Evaluator

if TYPE_CHECKING:
    from task_benchmark.implementations.data_loader.image_classification.base_data_loader import (
        ImageClassificationDataLoader,
    )
    from task_benchmark.implementations.invoker.image_classification.base_invoker import (
        ImageClassificationInvoker,
    )


class Top1ImageClassificationEvaluator(Evaluator):
    """Score one image's class scores against its dataset annotations.

    Return 1.0 when the highest-scoring class matches any annotated label,
    otherwise 0.0. This is top-1 correctness, not a multi-label F1 metric.
    """

    def __init__(
        self,
        data_loader: ImageClassificationDataLoader,
        invoker: ImageClassificationInvoker,
    ) -> None:
        self.data_loader = data_loader
        self.invoker = invoker

    def _evaluate(self, test_context: str, service_response: Tensor) -> float:
        if not isinstance(service_response, Tensor):
            raise TypeError("service_response must be a torch.Tensor")
        if service_response.ndim == 1:
            logits = service_response
        elif service_response.ndim == 2 and service_response.shape[0] == 1:
            logits = service_response[0]
        else:
            raise ValueError(
                "service_response must contain logits for exactly one image "
                "with shape [classes] or [1, classes]"
            )
        if logits.numel() == 0:
            raise ValueError("service_response must contain at least one class logit")

        annotated_labels = self.data_loader.labels_for(test_context)
        if not annotated_labels:
            raise ValueError(f"No ground-truth labels found for '{test_context}'")

        predicted_class_id = int(logits.detach().argmax().item())
        predicted_label = self.invoker.label_for(predicted_class_id)
        normalized_prediction = predicted_label.strip().casefold()
        return float(
            any(
                normalized_prediction == label.strip().casefold()
                for label in annotated_labels
            )
        )