# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import torch
from task_inference import create_task
from task_inference.tasks.audio.audio_classification import (
    AudioClassificationInput,
    AudioClassificationOutput,
)

from task_benchmark.tasks.audio_classification.task import (
    AudioClassificationModel,
    Prediction,
)
from task_benchmark.implementations.registry import implementation_registry


class TaskInferenceAudioClassifier(
    AudioClassificationModel
):

    def __init__(
        self,
        model_name: str,
        device: str,
        labels: list[str] | None = None,
        model_params: dict[str, Any] | None = None,
    ):
        _ = labels

        self.tasks = []
        task_params = dict(model_params or {})

        if not device.startswith("cuda"):
            self.tasks.append(
                create_task(
                    backend="transformers",
                    task_name="audio-classification",
                    model_name=model_name,
                    model_params={**task_params, "device": device},
                )
            )
            return

        if device.startswith("cuda:"):
            self.tasks.append(
                create_task(
                    backend="transformers",
                    task_name="audio-classification",
                    model_name=model_name,
                    model_params={**task_params, "device": device},
                )
            )
            return

        gpu_count = torch.cuda.device_count()

        if gpu_count <= 1:
            self.tasks.append(
                create_task(
                    backend="transformers",
                    task_name="audio-classification",
                    model_name=model_name,
                    model_params={**task_params, "device": "cuda"},
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
                    task_name="audio-classification",
                    model_name=model_name,
                    model_params={
                        **task_params,
                        "device": f"cuda:{gpu_id}",
                    },
                )
            )

        self.batch_counter = 0

    def predict_batch(
        self,
        inputs: list[bytes],
        sample_rate: int,
        top_k: int = 5,
    ) -> list[list[Prediction]]:

        if len(self.tasks) == 1:
            task = self.tasks[0]
        else:
            task = self.tasks[
                self.batch_counter % len(self.tasks)
            ]
            self.batch_counter += 1

        inp = AudioClassificationInput(
            audio=inputs,
            sample_rate=sample_rate,
            top_k=top_k,
        )

        response = task(
            inp.to_inference_request()
        )

        output = (
            AudioClassificationOutput
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
                for p in per_audio
            ]
            for per_audio in output.results
        ]


implementation_registry.register(
    task="audio-classification",
    implementation="task-inference",
    implementation_cls=TaskInferenceAudioClassifier,
)
