import os
from pathlib import Path
from tempfile import TemporaryDirectory

from task_benchmark import evaluate
from task_benchmark.tasks.image_classification import ImageClassificationDataObject


MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\x0bIDATx\x9cc`\x00\x02\x00\x00\x05\x00\x01\r\n-\xb4"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


if __name__ == "__main__":
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        inputs_dir = tmp_path / "images"
        inputs_dir.mkdir(parents=True, exist_ok=True)

        (inputs_dir / "img1.png").write_bytes(MINIMAL_PNG)
        (inputs_dir / "img2.png").write_bytes(MINIMAL_PNG)

        data_object = ImageClassificationDataObject(
            images_path=[
                str(inputs_dir / "img1.png"),
                str(inputs_dir / "img2.png"),
            ],
            labels=["tench", "goldfish"],
        )

        report = evaluate(
            task="image-classification",
            implementation="open-inference",
            data_object=data_object,
            model_name=os.getenv(
                "OPEN_INFERENCE_MODEL",
                "google/vit-base-patch16-224",
            ),
            model_params={
                "base_url": os.getenv(
                    "OPEN_INFERENCE_BASE_URL",
                    "http://localhost:8080",
                ),
            },
            batch_size=2,
        )

        print("Top-1 accuracy:", report["top1_accuracy"])
        print("Top-5 accuracy:", report["top5_accuracy"])
