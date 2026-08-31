from __future__ import annotations

from pathlib import Path
from typing import Any

from digitalhub_runtime_python import handler

from task_benchmark import evaluate
from task_benchmark.tasks import create_task_handler


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


def _resolve_tiny_imagenet_val_root(dataset_artifact: Any) -> Path:
    """Resolve val root from a downloaded Tiny-ImageNet artifact."""
    resolved = _materialize_artifacts(dataset_artifact)
    dataset_path = Path(str(resolved))

    if dataset_path.is_file():
        raise ValueError(
            "dataset artifact must resolve to a directory, not a file. "
            f"Got: {dataset_path}"
        )

    direct_val = dataset_path
    nested_val = dataset_path / "tiny-imagenet-200" / "val"

    for candidate in (direct_val, nested_val):
        if (candidate / "val_annotations.txt").is_file() and (candidate / "images").is_dir():
            return candidate

    raise ValueError(
        "Could not resolve Tiny-ImageNet val directory from artifact. "
        "Expected either <artifact>/val_annotations.txt + <artifact>/images, "
        "or <artifact>/tiny-imagenet-200/val/...."
    )


def _build_payload_from_tiny_imagenet_val(
    val_root: Path,
    max_samples: int | None,
) -> dict[str, list[str]]:
    words_file = val_root.parent / "words.txt"
    if not words_file.is_file():
        raise ValueError(f"Missing words.txt at expected path: {words_file}")

    class_id_to_label = {
        class_id: description
        for class_id, description in (
            line.strip().split("\t", 1)
            for line in words_file.read_text().splitlines()
            if line.strip()
        )
    }

    annotations = val_root / "val_annotations.txt"
    if not annotations.is_file():
        raise ValueError(f"Missing val_annotations.txt at expected path: {annotations}")

    images_dir = val_root / "images"
    images_path: list[str] = []
    labels: list[str] = []

    for line in annotations.read_text().splitlines():
        if not line.strip():
            continue

        filename, class_id, *_ = line.split("\t")
        label = class_id_to_label.get(class_id)
        if label is None:
            raise ValueError(f"Class id '{class_id}' not found in words.txt")

        images_path.append(str(images_dir / filename))
        labels.append(label)

        if max_samples is not None and len(images_path) >= max_samples:
            break

    if not images_path:
        raise ValueError("No samples found while building payload from Tiny-ImageNet artifact.")

    return {
        "images_path": images_path,
        "labels": labels,
    }


@handler(outputs=["evaluation_report"])
def evaluate_model(
    project,
    dataset_artifact,
    profile: str = "default",
    task: str = "image-classification",
    implementation: str = "task-inference",
    device: str = "cpu",
    max_samples: int = 1000,
    **kwargs,
):
    """DigitalHub runtime entrypoint."""

    val_root = _resolve_tiny_imagenet_val_root(dataset_artifact)
    payload = _build_payload_from_tiny_imagenet_val(
        val_root=val_root,
        max_samples=max_samples,
    )
    resolved_kwargs = _materialize_artifacts(kwargs)

    task_handler = create_task_handler(task=task)
    task_data_object = task_handler.build_data_object(payload=payload)

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
        source=str(output_path),
    )