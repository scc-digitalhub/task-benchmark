from __future__ import annotations

import os
from pathlib import Path

import kagglehub
import pandas as pd

from task_benchmark import evaluate
from task_benchmark.tasks.image_classification import ImageClassificationDataObject


DATASET_NAME = "akash2sharma/tiny-imagenet"
DATASET_SIZE = int(os.getenv("OPEN_INFERENCE_DATASET_SIZE", "1000"))
MODEL_NAME = os.getenv(
    "OPEN_INFERENCE_MODEL",
    "google/vit-base-patch16-224",
)
BASE_URL = os.getenv("OPEN_INFERENCE_BASE_URL", "http://localhost:8080")
BATCH_SIZE = int(os.getenv("OPEN_INFERENCE_BATCH_SIZE", "16"))
REPORT_PATH = Path(__file__).parent / "report_open_inference.json"


def build_dataframe(val_root: Path, max_samples: int) -> pd.DataFrame:
    words_file = val_root.parent / "words.txt"
    class_id_to_label = dict(
        line.strip().split("\t", 1)
        for line in words_file.open()
        if line.strip()
    )

    images_dir = val_root / "images"
    annotations = val_root / "val_annotations.txt"
    rows = []

    with annotations.open() as file_handle:
        for line in file_handle:
            filename, class_id, *_ = line.split("\t")
            rows.append(
                {
                    "image_path": str(images_dir / filename),
                    "label": class_id_to_label[class_id],
                }
            )

            if len(rows) >= max_samples:
                break

    return pd.DataFrame(rows)


def load_data_object(dataframe: pd.DataFrame) -> ImageClassificationDataObject:
    return ImageClassificationDataObject(
        images_path=dataframe["image_path"].tolist(),
        labels=dataframe["label"].tolist(),
    )


if __name__ == "__main__":
    dataset_path = kagglehub.dataset_download(DATASET_NAME)
    dataset_root = Path(dataset_path) / "tiny-imagenet-200" / "val"

    dataframe = build_dataframe(
        val_root=dataset_root,
        max_samples=DATASET_SIZE,
    )
    data_object = load_data_object(dataframe)

    report = evaluate(
        task="image-classification",
        implementation="open-inference",
        data_object=data_object,
        model_name=MODEL_NAME,
        model_params={"base_url": BASE_URL},
        batch_size=BATCH_SIZE,
        report_path=REPORT_PATH,
    )

    print("Top-1 accuracy:", report["top1_accuracy"])
    print("Top-5 accuracy:", report["top5_accuracy"])
    print("Report saved to:", REPORT_PATH)
