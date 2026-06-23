"""
Template for implementing a custom image classification model.

Copy this file, rename it, and implement your model.

Example:
    cp template.py my_model.py
    # Edit my_model.py and implement MyModel class
    # In config: custom_model_name="my-model"
"""

from __future__ import annotations

from ..models import (
    ImageClassificationModel,
    Prediction,
)


class MyCustomModel(ImageClassificationModel):
    """
    Template custom image classification model.

    Replace this with your own implementation.
    You can use:
    - Classic ML (sklearn, XGBoost, etc.)
    - Handcrafted features
    - Small neural networks
    - Any other inference backend
    """

    predicts_wnid: bool = True
    """Set to True if predictions are WordNet IDs (wnid), False if other labels."""

    def __init__(
        self,
        class_descriptions: dict[str, str],
        device: str = "cpu",
    ) -> None:
        """
        Initialize your model.

        Args:
            class_descriptions: Dict mapping class IDs (e.g., wnids) to descriptions.
                               Use this to filter or rank your predictions.
            device: "cpu" or "cuda" (or "cuda:0", "cuda:1", etc.).
                   Adapt this to your inference framework.
        """

        self.class_descriptions = class_descriptions
        self.device = device

        # Example: load a pretrained model here
        # self.model = load_model("path/to/model.pth")

    def predict_batch(
        self,
        images: list[bytes],
        top_k: int = 5,
    ) -> list[list[Prediction]]:
        """
        Run inference on a batch of images.

        Args:
            images: List of image byte strings (raw image data).
            top_k: Return only top-k predictions per image.

        Returns:
            List of lists: one list per image, each with up to top_k Prediction objects.
            Prediction has .label (str) and .score (float, ideally 0–1).
        """

        predictions: list[list[Prediction]] = []

        for image_bytes in images:
            # TODO: implement your inference logic here
            # Example pseudo-code:
            #   image = Image.open(BytesIO(image_bytes))
            #   logits = self.model(image)
            #   top_indices = np.argsort(logits)[-top_k:][::-1]
            #   per_image = [
            #       Prediction(label=class_id, score=float(logits[idx]))
            #       for idx in top_indices
            #   ]

            per_image: list[Prediction] = []

            # Placeholder: return empty predictions
            # Replace with actual model predictions.

            predictions.append(per_image)

        return predictions


def create_model(
    class_descriptions: dict[str, str],
    device: str,
) -> ImageClassificationModel:
    """
    Factory function that must be present in every custom model module.

    Args:
        class_descriptions: Class mapping from benchmark config.
        device: Device string ("cpu" or "cuda...").

    Returns:
        An instance of ImageClassificationModel (or subclass).
    """

    return MyCustomModel(
        class_descriptions=class_descriptions,
        device=device,
    )
