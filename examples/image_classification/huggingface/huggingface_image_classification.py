from __future__ import annotations

import os
from pathlib import Path

import torch

from task_benchmark.implementations.data_loader.image_classification import (
    HuggingFaceImageClassificationDataLoader,
)
from task_benchmark.implementations.invoker.image_classification import (
    HuggingFaceImageClassificationInvoker,
)


def main() -> None:
    dataset_id = os.getenv("HF_DATASET_ID", "uoft-cs/cifar10")
    model_name = os.getenv(
        "HF_MODEL_ID",
        "nateraw/vit-base-patch16-224-cifar10",
    )
    cache_dir = Path(os.getenv("HF_CACHE_DIR", ".cache/huggingface")).expanduser()
    max_samples = int(os.getenv("HF_MAX_SAMPLES", "50"))
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
    )
    invoker = HuggingFaceImageClassificationInvoker(
        model_name=model_name,
        device=device,
        model_kwargs={"cache_dir": str(cache_dir)},
        processor_kwargs={"cache_dir": str(cache_dir)},
    )

    for sample_index, (image_id, image_tensor) in enumerate(loader):
        logits = invoker.invoke(image_tensor)
        probabilities = logits.softmax(dim=-1)[0]
        predicted_index = int(probabilities.argmax().item())
        predicted_label = invoker.label_for(predicted_index)
        ground_truth = ", ".join(loader.labels_for(image_id))
        confidence = float(probabilities[predicted_index].item())

        print(
            f"{image_id}: prediction={predicted_label} "
            f"(confidence={confidence:.3f}), ground_truth={ground_truth}"
        )

        if sample_index + 1 >= max_samples:
            break


if __name__ == "__main__":
    main()