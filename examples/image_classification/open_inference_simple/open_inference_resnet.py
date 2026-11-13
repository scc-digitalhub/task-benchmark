from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import request

from torch import Tensor

from task_benchmark import BenchmarkManager
from task_benchmark.impl.data_loader.image_classification import (
    HuggingFaceImageClassificationDataLoader,
)
from task_benchmark.impl.evaluator.image_classification import (
    Top1ImageClassificationEvaluator,
)
from task_benchmark.impl.invoker.base_invoker import Invoker
from task_benchmark.impl.invoker.image_classification import (
    OpenInferenceImageClassificationInvoker,
)
from task_benchmark.impl.invoker.open_inference import OpenInferenceInvoker


CIFAR10_DATASET_ID = "uoft-cs/cifar10"
CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def save_metrics(metrics: dict[str, Any]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path.cwd() / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = 0
    while True:
        collision_suffix = "" if suffix == 0 else f"_{suffix}"
        metrics_path = output_dir / f"metrics_resnet_{timestamp}{collision_suffix}.json"
        try:
            with metrics_path.open("x", encoding="utf-8") as metrics_file:
                json.dump(metrics, metrics_file, indent=2)
                metrics_file.write("\n")
            return metrics_path
        except FileExistsError:
            suffix += 1


class BatchedImageInvoker(Invoker[Tensor, Tensor]):
    def __init__(self, invoker: OpenInferenceImageClassificationInvoker) -> None:
        self.invoker = invoker

    def invoke(self, test_data: Tensor) -> Tensor:
        return self.invoker.invoke(test_data.unsqueeze(0))


def class_labels(
    loader: HuggingFaceImageClassificationDataLoader,
    dataset_id: str,
) -> dict[int, str]:
    mapping_path = os.getenv("OPEN_INFERENCE_LABELS_PATH")
    if mapping_path:
        mapping = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
        if not isinstance(mapping, dict):
            raise ValueError("OPEN_INFERENCE_LABELS_PATH must contain a JSON object")
        labels: dict[int, str] = {}
        for class_id, label in mapping.items():
            try:
                index = int(class_id)
            except (TypeError, ValueError) as exc:
                raise ValueError("Label mapping keys must be integer class indices") from exc
            if index in labels:
                raise ValueError(f"Duplicate class index {index} in label mapping")
            labels[index] = label
    else:
        if dataset_id != CIFAR10_DATASET_ID:
            raise ValueError(
                "Set OPEN_INFERENCE_LABELS_PATH for datasets other than CIFAR-10"
            )
        labels = dict(enumerate(CIFAR10_CLASSES))

    if set(labels) != set(range(len(CIFAR10_CLASSES))):
        raise ValueError("Labels must map all 10 ResNet output indices from zero")
    if any(not isinstance(label, str) or not label.strip() for label in labels.values()):
        raise ValueError("Labels must contain non-empty strings")

    dataset_labels = {label.strip().casefold() for label in loader.classes}
    mapped_labels = {label.strip().casefold() for label in labels.values()}
    if not dataset_labels or dataset_labels != mapped_labels:
        raise ValueError(
            "ResNet and dataset labels differ; configure OPEN_INFERENCE_LABELS_PATH"
        )
    return labels


def main() -> None:
    base_url = os.getenv(
        "OPEN_INFERENCE_BASE_URL",
        "http://localhost:9091/v2/models/resnet18_cifar10",
    ).rstrip("/").removesuffix("/infer")
    dataset_id = os.getenv("HF_DATASET_ID", CIFAR10_DATASET_ID)
    default_split = "test" if dataset_id == CIFAR10_DATASET_ID else "train"
    default_image_column = "img" if dataset_id == CIFAR10_DATASET_ID else "image"
    max_samples = int(os.getenv("HF_MAX_SAMPLES", "150"))
    if max_samples < 1:
        raise ValueError("HF_MAX_SAMPLES must be greater than zero")

    with request.urlopen(base_url, timeout=60.0) as response:
        metadata: dict[str, Any] = json.load(response)
    expected_input = {
        "name": "pixel_values",
        "datatype": "FP32",
        "shape": [1, 3, 32, 32],
    }
    expected_output = {
        "name": "scores",
        "datatype": "FP32",
        "shape": [1, len(CIFAR10_CLASSES)],
    }
    if metadata.get("inputs") != [expected_input]:
        raise ValueError("Expected the ResNet FP32 input with shape [1, 3, 32, 32]")
    if metadata.get("outputs") != [expected_output]:
        raise ValueError("Expected 10 FP32 class scores from the ResNet server")

    loader = HuggingFaceImageClassificationDataLoader(
        dataset_id=dataset_id,
        local_path=os.getenv("HF_CACHE_DIR", ".cache/huggingface"),
        params={
            "split": os.getenv("HF_DATASET_SPLIT", default_split),
            "image_column": os.getenv("HF_IMAGE_COLUMN", default_image_column),
            "label_column": os.getenv("HF_LABEL_COLUMN", "label"),
            "streaming": True,
        },
        max_samples=max_samples,
    )
    labels = class_labels(loader, dataset_id)
    client = OpenInferenceInvoker(f"{base_url}/infer")
    classifier = OpenInferenceImageClassificationInvoker(
        client,
        input_name="pixel_values",
        output_name="scores",
        labels=labels,
    )
    evaluator = Top1ImageClassificationEvaluator(loader, classifier)
    try:
        metrics = BenchmarkManager(
            data_loader=loader,
            invoker=BatchedImageInvoker(classifier),
            evaluator=evaluator,
            service_id=base_url,
        ).run_benchmark()
        print(json.dumps(metrics, indent=2))
        metrics_path = save_metrics(metrics)
        print(f"Metrics saved to: {metrics_path}")
    finally:
        classifier.close()


if __name__ == "__main__":
    main()
