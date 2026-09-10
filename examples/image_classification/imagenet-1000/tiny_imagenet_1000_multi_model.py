from pathlib import Path

import kagglehub
import pandas as pd

from task_benchmark import evaluate
from task_benchmark.tasks.image_classification import ImageClassificationDataObject


DATASET_NAME = "akash2sharma/tiny-imagenet"
DATASET_SIZE = 1000
MODELS = [
    "microsoft/cvt-13-384",
    "facebook/convnext-base-384-22k-1k",
]
BATCH_SIZE = 128
DEVICE = "cpu"
REPORT_DIR = Path(__file__).parent / "reports"


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

    with annotations.open() as fh:
        for line in fh:
            filename, class_id, *_ = line.split("\t")
            rows.append({
                "image_path": str(images_dir / filename),
                "label": class_id_to_label[class_id],
            })

            if len(rows) >= max_samples:
                break

    return pd.DataFrame(rows)


def load_data_object(df: pd.DataFrame) -> ImageClassificationDataObject:
    return ImageClassificationDataObject(
        images_path=df["image_path"].tolist(),
        labels=df["label"].tolist(),
    )


def report_path_for_model(model_name: str) -> Path:
    model_slug = model_name.replace("/", "__")
    return REPORT_DIR / f"report_{model_slug}.json"


def build_comparison_row(model_name: str, report_path: Path, report: dict) -> dict:
    row = {
        "model_name": model_name,
        "report_path": str(report_path),
    }

    for key, value in report.items():
        if isinstance(value, str | int | float | bool):
            row[key] = value

    return row


def run_evaluation(model_name: str, data_object: ImageClassificationDataObject) -> dict:
    report_path = report_path_for_model(model_name)

    report = evaluate(
        task="image-classification",
        implementation="task-inference",
        data_object=data_object,
        model_name=model_name,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        report_path=report_path,
    )

    return build_comparison_row(
        model_name=model_name,
        report_path=report_path,
        report=report,
    )


if __name__ == "__main__":
    dataset_path = kagglehub.dataset_download(DATASET_NAME)
    dataset_root = Path(dataset_path) / "tiny-imagenet-200" / "val"

    df = build_dataframe(
        val_root=dataset_root,
        max_samples=DATASET_SIZE,
    )
    data_object = load_data_object(df)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    comparison_rows = [
        run_evaluation(
            model_name=model_name,
            data_object=data_object,
        )
        for model_name in MODELS
    ]

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = REPORT_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    print()
    print("Model comparison:")
    print(comparison_df.to_string(index=False))
    print("Comparison saved to:", comparison_path)