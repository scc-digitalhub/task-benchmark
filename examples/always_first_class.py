from __future__ import annotations

import csv
from pathlib import Path
from tempfile import TemporaryDirectory

from task_benchmark.core import evaluate
from task_benchmark.implementations import implementation_registry
from task_benchmark.tasks.image_classification import (
    ImageClassificationModel,
    Prediction,
)


class AlwaysFirstClassImageClassifier(ImageClassificationModel):
    """Always predicts the first available class."""

    predicts_wnid = True

    def __init__(
        self,
        model_name: str = "",
        device: str = "cpu",
        class_descriptions: dict[str, str] | None = None,
    ) -> None:
        _ = model_name
        _ = device

        classes = sorted((class_descriptions or {}).keys())
        if not classes:
            raise ValueError("class_descriptions cannot be empty")

        self.first_class = classes[0]

    def predict_batch(
        self,
        inputs: list[bytes],
        top_k: int = 5,
    ) -> list[list[Prediction]]:
        _ = top_k

        return [
            [Prediction(label=self.first_class, score=1.0)]
            for _ in inputs
        ]


implementation_registry.register(
    "image-classification",
    "always-first-class",
    AlwaysFirstClassImageClassifier,
)


if __name__ == "__main__":
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        inputs_dir = tmp_path / "images"
        inputs_dir.mkdir(parents=True, exist_ok=True)

        # This demo implementation does not parse image bytes.
        (inputs_dir / "img1.bin").write_bytes(b"fake-image-1")
        (inputs_dir / "img2.bin").write_bytes(b"fake-image-2")

        dataset_path = tmp_path / "dataset.csv"
        with dataset_path.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "image_path",
                    "wnid",
                    "class_description",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "image_path": "img1.bin",
                    "wnid": "n01440764",
                    "class_description": "tench",
                }
            )
            writer.writerow(
                {
                    "image_path": "img2.bin",
                    "wnid": "n01443537",
                    "class_description": "goldfish",
                }
            )

        report = evaluate(
            dataset_path=dataset_path,
            task="image-classification",
            implementation="always-first-class",
            task_inputs_dir_path=inputs_dir,
            device="cpu",
            batch_size=2,
        )

        print("Top-1 accuracy:", report["top1_accuracy"])
        print("Top-5 accuracy:", report["top5_accuracy"])
