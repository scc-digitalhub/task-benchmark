import io
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from task_benchmark import evaluate
from task_benchmark.tasks.image_classification import ImageClassificationDataObject


def make_valid_png() -> bytes:
    image = Image.new("RGB", (224, 224), color="red")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


if __name__ == "__main__":
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        inputs_dir = tmp_path / "images"
        inputs_dir.mkdir(parents=True, exist_ok=True)

        valid_png = make_valid_png()
        (inputs_dir / "img1.png").write_bytes(valid_png)
        (inputs_dir / "img2.png").write_bytes(valid_png)

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
