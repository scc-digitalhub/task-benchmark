from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from task_benchmark import BenchmarkManager
from task_benchmark.impl.data_loader.image_classification import (
    HuggingFaceImageClassificationDataLoader,
)
from task_benchmark.impl.invoker.image_classification import (
    HuggingFaceImageClassificationInvoker,
)
from task_benchmark.impl.evaluator.image_classification import (
    Top1ImageClassificationEvaluator,
)


def main() -> None:
    dataset_id = os.getenv("HF_DATASET_ID", "uoft-cs/cifar10")
    model_name = os.getenv(
        "HF_MODEL_ID",
        "nateraw/vit-base-patch16-224-cifar10",
    )
    cache_dir = Path(os.getenv("HF_CACHE_DIR", ".cache/huggingface")).expanduser()
    max_samples = int(os.getenv("HF_MAX_SAMPLES", "150"))
    device = os.getenv(
        "DEVICE",
        "cuda" if torch.cuda.is_available() else "cpu",
    )

    if max_samples < 1:
        raise ValueError("HF_MAX_SAMPLES must be greater than zero")

    cache_dir.mkdir(parents=True, exist_ok=True)
    loader = HuggingFaceImageClassificationDataLoader(
        dataset_id=dataset_id,
        local_path=str(cache_dir),
        params={
            "split": os.getenv("HF_DATASET_SPLIT", "test"),
            "image_column": "img",
            "label_column": "label",
        },
        max_samples=max_samples,
    )
    model_kwargs = {"cache_dir": str(cache_dir)}
    if model_name == "nateraw/vit-base-patch16-224-cifar10":
        class_names = (
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        )
        id_to_label = dict(enumerate(class_names))
        model_kwargs.update(
            id2label=id_to_label,
            label2id={label: class_id for class_id, label in id_to_label.items()},
        )
    invoker = HuggingFaceImageClassificationInvoker(
        model_name=model_name,
        device=device,
        model_kwargs=model_kwargs,
        processor_kwargs={"cache_dir": str(cache_dir)},
    )
    evaluator = Top1ImageClassificationEvaluator(loader, invoker)
    benchmark_manager = BenchmarkManager(
        data_loader=loader,
        invoker=invoker,
        evaluator=evaluator,
        service_id=model_name,
    )
    metrics = benchmark_manager.run_benchmark()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()