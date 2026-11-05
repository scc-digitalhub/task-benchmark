from __future__ import annotations

import json
import os
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
from task_benchmark.impl.invoker.image_classification import (
    OpenInferenceImageClassificationInvoker,
)
from task_benchmark.impl.invoker.open_inference import OpenInferenceInvoker
from task_benchmark.impl.invoker.base_invoker import Invoker


class BatchedImageInvoker(Invoker[Tensor, Tensor]):
    def __init__(self, invoker: OpenInferenceImageClassificationInvoker) -> None:
        self.invoker = invoker

    def invoke(self, test_data: Tensor) -> Tensor:
        return self.invoker.invoke(test_data.unsqueeze(0))


def class_labels(metadata, loader, dataset_id: str) -> dict[int, str]:
    mapping_path = os.getenv("OPEN_INFERENCE_LABELS_PATH")
    if mapping_path:
        mapping = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
        return {int(class_id): label for class_id, label in mapping.items()}

    parameters = metadata.get("parameters", {})
    mapping = parameters.get("id2label")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Server metadata must expose parameters.id2label, or set OPEN_INFERENCE_LABELS_PATH")
    labels = {int(class_id): label for class_id, label in mapping.items()}
    if (
        dataset_id == "uoft-cs/cifar10"
        and parameters.get("model_id") == "nateraw/vit-base-patch16-224-cifar10"
    ):
        class_names = (
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        )
        if set(labels) != set(range(10)) or set(loader.classes) != set(class_names):
            raise ValueError("Expected the documented CIFAR-10 model and dataset classes")
        return dict(enumerate(class_names))
    if (
        dataset_id == "mrm8488/ImageNet1K-val"
        and parameters.get("model_id") == "google/vit-base-patch16-224"
    ):
        classes = loader.classes
        if (
            len(classes) != 1000 or len(labels) != 1000
            or tuple(sorted(classes)) != tuple(classes)
            or any(len(label) != 9 or not label.startswith("n")
                   or not label[1:].isdigit() for label in classes)
        ):
            raise ValueError("Expected the documented 1000-class ImageNet WordNet ordering")
        return dict(enumerate(classes))

    dataset_labels = {label.strip().casefold() for label in loader.classes}
    if not dataset_labels or not dataset_labels.issubset(
        {label.strip().casefold() for label in labels.values()}
    ):
        raise ValueError("Model and dataset labels differ; configure OPEN_INFERENCE_LABELS_PATH")
    return labels


def main() -> None:
    base_url = os.getenv(
        "OPEN_INFERENCE_BASE_URL",
        "http://localhost:9090/v2/models/vit-base-patch16-224",
    ).rstrip("/").removesuffix("/infer")
    dataset_id = os.getenv("HF_DATASET_ID", "uoft-cs/cifar10")
    default_split = "test" if dataset_id == "uoft-cs/cifar10" else "train"
    default_image_column = "img" if dataset_id == "uoft-cs/cifar10" else "image"
    max_samples = int(os.getenv("HF_MAX_SAMPLES", "150"))
    if max_samples < 1:
        raise ValueError("HF_MAX_SAMPLES must be greater than zero")
    with request.urlopen(base_url, timeout=60.0) as response:
        metadata = json.load(response)
    if [item["name"] for item in metadata.get("inputs", [])] != ["pixel_values"]:
        raise ValueError("Rebuild the example server: expected the numeric pixel_values input")
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
    labels = class_labels(metadata, loader, dataset_id)
    client = OpenInferenceInvoker(f"{base_url}/infer")
    classifier = OpenInferenceImageClassificationInvoker(
        client, input_name="pixel_values", output_name="scores", labels=labels,
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
    finally:
        classifier.close()


if __name__ == "__main__":
    main()