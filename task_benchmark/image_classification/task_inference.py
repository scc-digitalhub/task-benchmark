from task_inference import create_task

from task_inference.tasks.vision.image_classification import (
    ImageClassificationInput,
    ImageClassificationOutput,
)

import torch

from .models import (
    Prediction,
    ImageClassificationModel,
)


class TaskInferenceImageClassifier(
    ImageClassificationModel
):

    predicts_wnid = False

    def __init__(
        self,
        model_name: str,
        device: str,
    ):
        self.tasks = []

        # CPU

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

        # Explicit GPU

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

        # device == "cuda"

        gpu_count = (
            torch.cuda.device_count()
        )

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
            f"Using {gpu_count} GPUs "
            "with round-robin scheduling"
        )

        for gpu_id in range(
            gpu_count
        ):
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

    @property
    def task_count(
        self,
    ) -> int:

        return len(
            self.tasks
        )

    def predict_batch(
        self,
        images: list[bytes],
        top_k: int = 5,
    ) -> list[list[Prediction]]:

        if len(self.tasks) == 1:

            task = self.tasks[0]

        else:

            task = self.tasks[
                self.batch_counter
                % len(self.tasks)
            ]

            self.batch_counter += 1

        inp = (
            ImageClassificationInput(
                images=images,
                top_k=top_k,
            )
        )

        response = task(
            inp.to_inference_request()
        )

        output = (
            ImageClassificationOutput
            .from_inference_response(
                response
            )
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
            for per_image
            in output.results
        ]

