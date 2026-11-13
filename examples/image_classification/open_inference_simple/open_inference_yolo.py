from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import request

import torch
from torch import Tensor
from torch.nn import functional as F

from task_benchmark import BenchmarkManager
from task_benchmark.impl.data_loader.image_classification import (
    ImageNet10ImageClassificationDataLoader,
)
from task_benchmark.impl.evaluator.image_classification import (
    Top1ImageClassificationEvaluator,
)
from task_benchmark.impl.invoker.base_invoker import Invoker
from task_benchmark.impl.invoker.image_classification import (
    OpenInferenceImageClassificationInvoker,
)
from task_benchmark.impl.invoker.open_inference import OpenInferenceInvoker


MODEL_CLASSES = 1000
IMAGE_SIZE = 224


def save_metrics(metrics: dict[str, Any]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path.cwd() / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = 0
    while True:
        collision_suffix = "" if suffix == 0 else f"_{suffix}"
        metrics_path = output_dir / f"metrics_yolo_{timestamp}{collision_suffix}.json"
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
        if test_data.ndim != 3 or test_data.shape[0] != 3:
            raise ValueError("test_data must have RGB CHW shape")
        if not torch.isfinite(test_data).all().item():
            raise ValueError("test_data must contain finite values")
        if test_data.min() < 0 or test_data.max() > 1:
            raise ValueError("test_data values must be in [0, 1]")

        height, width = test_data.shape[-2:]
        scale = IMAGE_SIZE / min(height, width)
        resized_height = max(IMAGE_SIZE, round(height * scale))
        resized_width = max(IMAGE_SIZE, round(width * scale))
        resized = F.interpolate(
            test_data.unsqueeze(0),
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        top = (resized_height - IMAGE_SIZE) // 2
        left = (resized_width - IMAGE_SIZE) // 2
        pixels = resized[
            :,
            :,
            top : top + IMAGE_SIZE,
            left : left + IMAGE_SIZE,
        ]
        return self.invoker.invoke(pixels)


def class_labels(
    loader: ImageNet10ImageClassificationDataLoader,
) -> dict[int, str]:
    if len(loader.classes) != 10:
        raise ValueError("ImageNet10 must provide exactly 10 classes")

    mapping_path = os.getenv("OPEN_INFERENCE_LABELS_PATH")
    if mapping_path:
        raw_mapping = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
        if not isinstance(raw_mapping, dict):
            raise ValueError("OPEN_INFERENCE_LABELS_PATH must contain a JSON object")
        labels: dict[int, str] = {}
        for class_id, label in raw_mapping.items():
            try:
                index = int(class_id)
            except (TypeError, ValueError) as exc:
                raise ValueError("Label mapping keys must be integer class indices") from exc
            if index in labels:
                raise ValueError(f"Duplicate class index {index} in label mapping")
            labels[index] = label
        if set(labels) != set(range(MODEL_CLASSES)):
            raise ValueError("Label mapping must contain all 1000 output class indices")
    else:
        labels = {
            index: f"__imagenet_class_{index}__"
            for index in range(MODEL_CLASSES)
        }
        labels.update(enumerate(loader.classes))

    if any(not isinstance(label, str) or not label.strip() for label in labels.values()):
        raise ValueError("Labels must contain non-empty strings")
    for index, expected_label in enumerate(loader.classes):
        if labels[index].strip().casefold() != expected_label.strip().casefold():
            raise ValueError(
                "The first 10 label indices must match the ImageNet10 class order"
            )
    return labels


def main() -> None:
    base_url = os.getenv(
        "OPEN_INFERENCE_BASE_URL",
        "http://localhost:59462/v2/models/yolo26n-cls",
    ).rstrip("/").removesuffix("/infer")
    max_samples = int(os.getenv("IMAGENET10_MAX_SAMPLES", "12"))
    if max_samples < 1:
        raise ValueError("IMAGENET10_MAX_SAMPLES must be greater than zero")

    with request.urlopen(base_url, timeout=60.0) as response:
        metadata: dict[str, Any] = json.load(response)
    expected_input = {
        "name": "images",
        "datatype": "FP32",
        "shape": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
    }
    expected_output = {
        "name": "output0",
        "datatype": "FP32",
        "shape": [1, MODEL_CLASSES],
    }
    if metadata.get("inputs") != [expected_input]:
        raise ValueError("Expected the YOLO FP32 input [1, 3, 224, 224]")
    if metadata.get("outputs") != [expected_output]:
        raise ValueError("Expected the YOLO output0 with 1000 FP32 class scores")

    loader = ImageNet10ImageClassificationDataLoader(
        local_path=os.getenv("IMAGENET10_CACHE_DIR", ".cache/imagenet10"),
        split=os.getenv("IMAGENET10_SPLIT", "val"),
        max_samples=max_samples,
    )
    labels = class_labels(loader)
    client = OpenInferenceInvoker(f"{base_url}/infer")
    classifier = OpenInferenceImageClassificationInvoker(
        client,
        input_name="images",
        output_name="output0",
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
