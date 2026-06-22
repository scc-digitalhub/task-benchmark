from .models import (
    ImageClassificationModel,
    Prediction,
)


class HashBaselineModel(
    ImageClassificationModel
):

    predicts_wnid = True

    def __init__(
        self,
        class_descriptions,
        device="cpu",
    ):
        self.class_descriptions = class_descriptions

    def predict_batch(
        self,
        images,
        top_k=5,
    ):

        wnid = next(
            iter(self.class_descriptions.keys())
        )

        return [
            [
                Prediction(
                    label=wnid,
                    score=1.0,
                )
            ]
            for _ in images
        ]


def create_custom_model(
    custom_model_name,
    class_descriptions,
    device,
):

    if custom_model_name == "hash-baseline":

        return HashBaselineModel(
            class_descriptions,
            device,
        )

    raise ValueError(
        f"Unknown custom model: "
        f"{custom_model_name}"
    )