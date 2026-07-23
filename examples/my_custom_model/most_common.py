from collections import Counter

from task_benchmark.implementations.registry import implementation_registry
from task_benchmark.tasks.image_classification.task import (
    ImageClassificationModel,
    Prediction,
)


class MostCommonClassModel(ImageClassificationModel):
    """Predict the class with the most common description as top-1."""

    def __init__(
        self,
        model_name: str = "",
        device: str = "cpu",
        class_descriptions: dict[str, str] | None = None,
    ) -> None:
        _ = model_name
        _ = device

        if class_descriptions is None:
            class_descriptions = {}

        if not class_descriptions:
            raise ValueError("class_descriptions cannot be empty")

        description_counts = Counter(class_descriptions.values())
        most_common_description, _ = description_counts.most_common(1)[0]

        majority_candidates = [
            label
            for label, description in class_descriptions.items()
            if description == most_common_description
        ]

        self.majority_class = sorted(majority_candidates)[0]
        self.other_classes = sorted(
            label for label in class_descriptions.keys() if label != self.majority_class
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
