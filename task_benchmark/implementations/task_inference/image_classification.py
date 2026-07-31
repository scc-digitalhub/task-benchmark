# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from task_inference import create_task
from task_inference.tasks.vision.image_classification import (
    ImageClassificationInput,
    ImageClassificationOutput,
)

from task_benchmark.tasks.image_classification.task import (
    ImageClassificationModel,
    Prediction,
)
from task_benchmark.implementations.registry import implementation_registry


class TaskInferenceImageClassifier(
    ImageClassificationModel
):

    def __init__(
        self,
        model_name: str,
        device: str,
        labels: list[str] | None = None,
    ):
        _ = labels

        self.tasks = []

        if not device.startswith("cuda"):
            self.tasks.append(
                create_task(
                    backend="transformers",
                    task_name="image-classification",
                    model_name=model_name,
                    model_params={
                        "device": device,
                    },
                )
            )
            return

        if device.startswith("cuda:"):
            self.tasks.append(
                create_task(
                    backend="transformers",
                    task_name="image-classification",
                    model_name=model_name,
                    model_params={
                        "device": device,
                    },
                )
            )
            return

        gpu_count = torch.cuda.device_count()

        if gpu_count <= 1:
            self.tasks.append(
                create_task(
                    backend="transformers",
                    task_name="image-classification",
                    model_name=model_name,
                    model_params={
                        "device": "cuda",
                    },
                )
            )
            return

        print(
            f"Using {gpu_count} GPUs with round-robin scheduling"
        )

        for gpu_id in range(gpu_count):
            self.tasks.append(
                create_task(
                    backend="transformers",
                    task_name="image-classification",
                    model_name=model_name,
                    model_params={
                        "device": f"cuda:{gpu_id}",
                    },
                )
            )

        self.batch_counter = 0

    def predict_batch(
        self,
        inputs: list[bytes],
        top_k: int = 5,
    ) -> list[list[Prediction]]:

        if len(self.tasks) == 1:
            task = self.tasks[0]
        else:
            task = self.tasks[
                self.batch_counter % len(self.tasks)
            ]
            self.batch_counter += 1

        inp = ImageClassificationInput(
            images=inputs,
            top_k=top_k,
        )

        response = task(
            inp.to_inference_request()
        )

        output = (
            ImageClassificationOutput
            .from_inference_response(response)
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        return [
            [
                Prediction(
                    label=p.label,
                    score=p.score,
                )
                for p in per_image
            ]
            for per_image in output.results
        ]


implementation_registry.register(
    task="image-classification",
    implementation="task-inference",
    implementation_cls=TaskInferenceImageClassifier,
)
