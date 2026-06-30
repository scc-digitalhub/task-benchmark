# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from .custom import create_custom_model
from .task_inference import TaskInferenceImageClassifier


def create_image_classifier(
    implementation_name: str,
    model_name: str,
    custom_model_name: str,
    class_descriptions: dict[str, str],
    device: str,
):
    if implementation_name == "task-inference":
        return TaskInferenceImageClassifier(
            model_name=model_name,
            device=device,
        )

    if implementation_name == "custom":
        return create_custom_model(
            custom_model_name=custom_model_name,
            class_descriptions=class_descriptions,
            device=device,
        )

    raise ValueError(
        f"Unsupported implementation: {implementation_name}"
    )
