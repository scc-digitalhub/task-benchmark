from __future__ import annotations

from pathlib import Path
from typing import Any

import kagglehub
import pandas as pd
from digitalhub_runtime_python import handler

from task_benchmark import evaluate
from task_benchmark.tasks import create_task_handler
from task_benchmark.tasks.image_classification.task import ImageClassificationDataObject


MODEL_NAME = "microsoft/cvt-13-384"
BATCH_SIZE = 128
DEVICE = "cpu"
REPORT_PATH = Path(__file__).parent / "report_cvt13.json"


def build_dataframe(val_root: Path) -> pd.DataFrame:
    """Build a DataFrame from Tiny-ImageNet validation annotations."""
    words_file = val_root.parent / "words.txt"
    synset_to_label = dict(
        line.strip().split("\t", 1)
        for line in words_file.open()
    )

    images_dir = val_root / "images"
    annotations = val_root / "val_annotations.txt"
    rows = []
    with annotations.open() as fh:
        for line in fh:
            filename, synset, *_ = line.split("\t")
            rows.append(
                {
                    "image_path": str(images_dir / filename),
                    "label": synset_to_label[synset],
                }
            )
    return pd.DataFrame(rows)


def load_data_object(df: pd.DataFrame) -> ImageClassificationDataObject:
    """Convert DataFrame columns into task-benchmark data object."""
    return ImageClassificationDataObject(
        images_path=df["image_path"].tolist(),
        labels=df["label"].tolist(),
    )


def _download_if_artifact(value: Any) -> Any:
    downloader = getattr(value, "download", None)
    if callable(downloader):
        return downloader()
    return value


def _materialize_artifacts(value: Any) -> Any:
    # Resolve DigitalHub artifact handles to local runtime paths.
    if isinstance(value, dict):
        return {
            key: _materialize_artifacts(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_materialize_artifacts(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_materialize_artifacts(item) for item in value)

    return _download_if_artifact(value)


@handler(outputs=["evaluation_report"])
def evaluate_model(
    project,
    data_object: dict[str, Any],
    profile: str = "default",
    task: str = "image-classification",
    implementation: str = "task-inference",
    device: str = "cpu",
    **kwargs,
):
    """DigitalHub runtime entrypoint."""

    resolved_data_object = _materialize_artifacts(data_object)
    resolved_kwargs = _materialize_artifacts(kwargs)

    task_handler = create_task_handler(task=task)
    task_data_object = task_handler.build_data_object(payload=resolved_data_object)

    output_path = Path("evaluation_report.json")

    evaluate(
        task=task,
        implementation=implementation,
        data_object=task_data_object,
        report_path=output_path,
        profile=profile,
        device=device,
        **resolved_kwargs,
    )

    return project.log_artifact(
        name="evaluation_report",
        kind="table",
        source=str(output_path),
    )


if __name__ == "__main__":
    path = kagglehub.dataset_download("akash2sharma/tiny-imagenet")
    dataset_root = Path(path) / "tiny-imagenet-200" / "val"

    df = build_dataframe(dataset_root).head(1000)
    local_data_object = load_data_object(df)

    report = evaluate(
        task="image-classification",
        implementation="task-inference",
        data_object=local_data_object,
        model_name=MODEL_NAME,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        report_path=REPORT_PATH,
    )

    print("Top-1 accuracy:", report["top1_accuracy"])
    print("Top-5 accuracy:", report["top5_accuracy"])
    print("Report saved to:", REPORT_PATH)