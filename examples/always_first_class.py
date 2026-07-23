import csv
from pathlib import Path
from tempfile import TemporaryDirectory

from task_benchmark.core import evaluate
from task_benchmark.implementations import implementation_registry
from task_benchmark.tasks.image_classification import (
    ImageClassificationModel,
    Prediction,
)


MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\x0bIDATx\x9cc`\x00\x02\x00\x00\x05\x00\x01\r\n-\xb4"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


class AlwaysFirstClassImageClassifier(ImageClassificationModel):
    """Always predicts the first available class."""

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
        (inputs_dir / "img1.bin").write_bytes(MINIMAL_PNG)
        (inputs_dir / "img2.bin").write_bytes(MINIMAL_PNG)

        dataset_csv = tmp_path / "dataset.csv"
        with dataset_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["image_path", "label"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "image_path": "img1.bin",
                    "label": "tench",
                }
            )
            writer.writerow(
                {
                    "image_path": "img2.bin",
                    "label": "goldfish",
                }
            )

        report = evaluate(
            dataset_path=tmp_path,
            task="image-classification",
            implementation="always-first-class",
            image_dir="images",
            device="cpu",
            batch_size=2,
        )

        print("Top-1 accuracy:", report["top1_accuracy"])
        print("Top-5 accuracy:", report["top5_accuracy"])
