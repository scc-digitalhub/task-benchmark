from .task_inference import (
    TaskInferenceImageClassifier,
)

from .custom import (
    create_custom_model,
)


def create_image_classifier(
    inference_engine,
    model_name,
    custom_model_name,
    class_descriptions,
    device,
):

    if inference_engine == "task-inference":

        return TaskInferenceImageClassifier(
            model_name=model_name,
            device=device,
        )

    if inference_engine == "custom":

        return create_custom_model(
            custom_model_name=custom_model_name,
            class_descriptions=class_descriptions,
            device=device,
        )

    raise ValueError(
        f"Unsupported engine: "
        f"{inference_engine}"
    )