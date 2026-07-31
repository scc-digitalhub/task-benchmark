from collections import Counter

from task_benchmark.implementations.registry import implementation_registry
from task_benchmark.tasks.image_classification.task import (
    ImageClassificationModel,
    Prediction,
)


class MostCommonClassModel(ImageClassificationModel):
    """Predict the most common label as top-1."""

    def __init__(
        self,
        model_name: str = "",
        device: str = "cpu",
        labels: list[str] | None = None,
    ) -> None:
        _ = model_name
        _ = device

        labels = labels or []

        if not labels:
            raise ValueError("labels cannot be empty")

        label_counts = Counter(labels)
        max_count = max(label_counts.values())
        majority_candidates = [
            label
            for label, count in label_counts.items()
            if count == max_count
        ]

        self.majority_class = sorted(set(majority_candidates))[0]
        self.other_classes = sorted(
            label for label in set(labels) if label != self.majority_class
        )

    def predict_batch(self, images: list[bytes], top_k: int = 5) -> list[list[Prediction]]:
        predictions: list[list[Prediction]] = []

        for _ in images:
            ranked_classes = [self.majority_class, *self.other_classes][: max(1, top_k)]
            per_image: list[Prediction] = []
            for rank, label in enumerate(ranked_classes):
                per_image.append(Prediction(label=label, score=1.0 / float(rank + 1)))
            predictions.append(per_image)

        return predictions


implementation_registry.register(
    task="image-classification",
    implementation="most-common-class",
    implementation_cls=MostCommonClassModel,
)
