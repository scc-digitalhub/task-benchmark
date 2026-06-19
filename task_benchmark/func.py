# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

import csv
import json
import time
import psutil
import torch

from pathlib import Path
from typing import Any
from __future__ import annotations


from digitalhub_runtime_python import handler
from transformers import (
    AutoConfig,
)
from task_inference import create_task
from task_inference.tasks.vision.image_classification import (
    ImageClassificationInput,
    ImageClassificationOutput,
)

try:
    from .custom_inference import (
        CustomInferenceModel,
        create_custom_model,
    )
except ImportError:
    from custom_inference import (  # type: ignore[no-redef]
        CustomInferenceModel,
        create_custom_model,
    )



# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZE = 8
DEFAULT_INFERENCE_ENGINE = "task-inference"


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def load_dataset_csv(
    dataset_csv: Path,
) -> list[dict[str, str]]:
    """
    Load dataset rows from CSV.
    """

    if not dataset_csv.is_file():

        raise FileNotFoundError(
            f"Could not find dataset CSV: "
            f"{dataset_csv}"
        )

    with dataset_csv.open(
        newline=""
    ) as fh:

        reader = csv.DictReader(fh)

        rows = list(reader)

    required_columns = {
        "image_path",
        "wnid",
        "class_description",
    }

    if not rows:

        raise RuntimeError(
            f"Dataset CSV is empty: "
            f"{dataset_csv}"
        )

    if not required_columns.issubset(
        set(rows[0].keys())
    ):

        raise ValueError(
            f"Dataset CSV must contain columns: "
            f"{sorted(required_columns)}"
        )

    return rows


def resolve_image_path(
    image_path_value: str,
    images_dir: Path,
) -> Path:
    """
    Resolve image path from CSV.
    """

    raw_path = Path(image_path_value)

    # Absolute path already valid
    if (
        raw_path.is_absolute()
        and raw_path.exists()
    ):
        return raw_path

    # Relative to images directory
    candidate = (
        Path(images_dir) / raw_path
    )

    if candidate.exists():
        return candidate

    # Try filename only
    candidate = (
        Path(images_dir)
        / raw_path.name
    )

    if candidate.exists():
        return candidate

    # Strip parent folders
    parts = raw_path.parts

    for i in range(len(parts)):

        candidate = (
            Path(images_dir)
            .joinpath(*parts[i:])
        )

        if candidate.exists():
            return candidate

    return raw_path


def build_label_to_wnid(
    class_descriptions: dict[str, str],
    model_id2label: dict[int, str],
) -> dict[str, str]:
    """
    Build mapping:
    {model_label: wnid}
    """

    lower_model_labels = {
        v.lower(): v
        for v in model_id2label.values()
    }

    label_to_wnid = {}

    unmapped = []

    for (
        wnid,
        description,
    ) in class_descriptions.items():

        desc_lower = description.lower()

        # Exact match
        if desc_lower in lower_model_labels:

            label_to_wnid[
                lower_model_labels[
                    desc_lower
                ]
            ] = wnid

            continue

        # Fuzzy match
        matched = None

        for (
            model_lower,
            model_orig,
        ) in lower_model_labels.items():

            if (
                desc_lower
                in model_lower
                or model_lower
                in desc_lower
            ):
                matched = model_orig
                break

        if matched:

            label_to_wnid[
                matched
            ] = wnid

        else:

            unmapped.append(
                f"{wnid} ({description})"
            )

    if unmapped:

        print(
            f"[WARNING] Could not map "
            f"{len(unmapped)} classes.\n"
            + "\n".join(
                unmapped[:10]
            )
        )

    return label_to_wnid


def get_peak_rss_mb() -> float:
    """
    Return RSS memory in MB.
    """

    process = psutil.Process()

    return (
        process.memory_info().rss
        / (1024 * 1024)
    )


def is_cuda_device(
    device: str,
) -> bool:
    return (
        device == "cuda"
        or device.startswith("cuda:")
    )


def build_task_inference_tasks(
    model_name: str,
    device: str,
    cuda_enabled: bool,
) -> tuple[list[Any], str]:
    """
    Create task-inference task(s) with explicit
    device-based conditioning.
    """

    if not cuda_enabled:

        task = create_task(
            backend="transformers",
            task_name="image-classification",
            model_name=model_name,
            model_params={"device": device},
        )

        return [task], "single-device"

    if device.startswith("cuda:"):

        task = create_task(
            backend="transformers",
            task_name="image-classification",
            model_name=model_name,
            model_params={"device": device},
        )

        return [task], "single-device"

    gpu_count = torch.cuda.device_count()

    if gpu_count > 1:

        device_ids = list(range(gpu_count))
        tasks = []

        for i in device_ids:

            task = create_task(
                backend="transformers",
                task_name="image-classification",
                model_name=model_name,
                model_params={
                    "device": f"cuda:{i}"
                },
            )

            tasks.append(task)

        print(
            "Using per-device task list "
            f"across {len(tasks)} GPUs"
        )

        return tasks, "round-robin-multi-device"

    task = create_task(
        backend="transformers",
        task_name="image-classification",
        model_name=model_name,
        model_params={"device": "cuda"},
    )

    return [task], "single-device"


def run_task_inference_batch(
    tasks: list[Any],
    batches_evaluated: int,
    batch_bytes: list[bytes],
    cuda_enabled: bool,
) -> list[list[str]]:
    """
    Run one batch using task-inference and return predicted labels.
    """

    task = tasks[
        batches_evaluated % len(tasks)
    ]

    inp = ImageClassificationInput(
        images=batch_bytes,
        top_k=5,
    )

    resp = task(
        inp.to_inference_request()
    )

    result = (
        ImageClassificationOutput
        .from_inference_response(resp)
    )

    if cuda_enabled:
        torch.cuda.synchronize()

    return [
        [p.label for p in per_image]
        for per_image in result.results
    ]


def run_custom_inference_batch(
    custom_model: CustomInferenceModel,
    batch_bytes: list[bytes],
) -> list[list[str]]:
    """
    Run one batch using a custom backend and return predicted labels.
    """

    custom_predictions = (
        custom_model.predict_batch(
            images=batch_bytes,
            top_k=5,
        )
    )

    return [
        [p.label for p in per_image]
        for per_image in custom_predictions
    ]


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------


@handler(
    outputs=["evaluation_report"]
)
def evaluate_model(
    project,
    dataset,
    images_dir,
    model_name: str = (
        "microsoft/cvt-13-384"
    ),
    batch_size: int = (
        DEFAULT_BATCH_SIZE
    ),
    device: str = "cpu",
    inference_engine: str = (
        DEFAULT_INFERENCE_ENGINE
    ),
    custom_model_name: str = (
        "hash-baseline"
    ),
):
    """
    Evaluate an image classifier with either:
    - task-inference backend (default)
    - custom backend defined in custom_inference.py
    """

    print(
        "Starting evaluation job..."
    )

    # -----------------------------------------------------------------------
    # Resolve paths
    # -----------------------------------------------------------------------

    dataset_path = (
        dataset.download()
    )

    images_dir_path = (
        images_dir.download()
    )

    print(
        f"Dataset CSV path: "
        f"{dataset_path}"
    )

    print(
        f"Images directory path: "
        f"{images_dir_path}"
    )

    if not Path(
        images_dir_path
    ).exists():

        raise RuntimeError(
            f"Images directory "
            f"does not exist: "
            f"{images_dir_path}"
        )

    # -----------------------------------------------------------------------
    # CUDA checks
    # -----------------------------------------------------------------------

    cuda_available = (
        torch.cuda.is_available()
    )

    if (
        is_cuda_device(device)
        and not cuda_available
    ):

        raise RuntimeError(
            "CUDA requested but "
            "not available."
        )

    cuda_enabled = (
        is_cuda_device(device)
        and cuda_available
    )

    if cuda_enabled:

        print("CUDA enabled")

        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()

    print(
        "CUDA available:",
        torch.cuda.is_available(),
    )

    print(
        "CUDA device count:",
        torch.cuda.device_count(),
    )

    if torch.cuda.is_available():

        for i in range(
            torch.cuda.device_count()
        ):

            print(
                f"CUDA device {i}:",
                torch.cuda.get_device_name(i),
            )

    print(
        "Torch version:",
        torch.__version__,
    )

    print(
        "Torch CUDA version:",
        torch.version.cuda,
    )

    # -----------------------------------------------------------------------
    # Benchmark init
    # -----------------------------------------------------------------------

    process = psutil.Process()

    wall_start = (
        time.perf_counter()
    )

    cpu_start = (
        process.cpu_times()
    )

    peak_rss_mb = (
        get_peak_rss_mb()
    )

    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------

    rows = load_dataset_csv(
        Path(dataset_path)
    )

    print(
        f"Loaded {len(rows)} "
        f"dataset rows."
    )

    # -----------------------------------------------------------------------
    # Label mapping
    # -----------------------------------------------------------------------

    class_descriptions = {}

    for row in rows:

        wnid = row["wnid"]

        if (
            wnid
            not in class_descriptions
        ):

            class_descriptions[
                wnid
            ] = row.get(
                "class_description",
                "",
            )

    label_to_wnid: dict[str, str] = {}

    tasks: list[Any] = []
    custom_model: CustomInferenceModel | None = None
    custom_predicts_wnid = False

    if inference_engine == "task-inference":

        # -------------------------------------------------------------------
        # Load config
        # -------------------------------------------------------------------

        print(
            f"Loading model config: "
            f"{model_name}"
        )

        config = (
            AutoConfig
            .from_pretrained(
                model_name
            )
        )

        id2label = {
            int(k): v
            for k, v in (
                config.id2label.items()
            )
        }

        print(
            "Building label mapping..."
        )

        label_to_wnid = (
            build_label_to_wnid(
                class_descriptions,
                id2label,
            )
        )

        print(
            f"Mapped "
            f"{len(label_to_wnid)} "
            f"labels."
        )

    # -----------------------------------------------------------------------
    # Create inference task(s)
    # -----------------------------------------------------------------------

    if inference_engine == "task-inference":

        print(
            f"Initializing inference task "
            f"on device={device}"
        )

        tasks, task_mode = (
            build_task_inference_tasks(
                model_name=model_name,
                device=device,
                cuda_enabled=cuda_enabled,
            )
        )

    elif inference_engine == "custom":

        print(
            "Initializing custom model "
            f"'{custom_model_name}' "
            f"on device={device}"
        )

        custom_model = create_custom_model(
            custom_model_name=custom_model_name,
            class_descriptions=class_descriptions,
            device=device,
        )

        custom_predicts_wnid = (
            custom_model.predicts_wnid
        )

        task_mode = "custom"

    else:

        raise ValueError(
            "Invalid inference_engine "
            f"'{inference_engine}'. "
            "Use 'task-inference' or 'custom'."
        )

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------

    top1_correct = 0
    top5_correct = 0
    total = 0
    skipped = 0

    batches_evaluated = 0

    batch_cpu_seconds_values = []
    batch_cpu_usage_values = []
    batch_wall_seconds_values = []
    batch_memory_values = []

    batch_gpu_allocated_values = []
    batch_gpu_reserved_values = []

    # -----------------------------------------------------------------------
    # Evaluation loop
    # -----------------------------------------------------------------------

    for batch_start in range(
        0,
        len(rows),
        batch_size,
    ):

        batch_wall_start = (
            time.perf_counter()
        )

        batch_cpu_start = (
            process.cpu_times()
        )

        batch_rows = rows[
            batch_start:
            batch_start + batch_size
        ]

        batch_bytes = []

        batch_wnids = []

        for row in batch_rows:
            image_path = (
                resolve_image_path(
                    row["image_path"],
                    Path(
                        images_dir_path
                    ),
                )
            )

            gt_wnid = row["wnid"]

            if not image_path.is_file():

                print(
                    f"[WARNING] "
                    f"Missing image: "
                    f"{image_path}"
                )

                skipped += 1

                continue

            try:
                batch_bytes.append(
                    image_path.read_bytes()
                )

                batch_wnids.append(
                    gt_wnid
                )

            except Exception as e:
                print(
                    f"Failed reading image: "
                    f"{image_path}"
                )

                print(e)

                skipped += 1

        if not batch_bytes:
            continue

        # -------------------------------------------------------------------
        # Inference
        # -------------------------------------------------------------------

        try:
            if inference_engine == "task-inference":

                prediction_labels = (
                    run_task_inference_batch(
                        tasks=tasks,
                        batches_evaluated=batches_evaluated,
                        batch_bytes=batch_bytes,
                        cuda_enabled=cuda_enabled,
                    )
                )

            else:

                if custom_model is None:

                    raise RuntimeError(
                        "Custom model was not initialized."
                    )

                prediction_labels = (
                    run_custom_inference_batch(
                        custom_model=custom_model,
                        batch_bytes=batch_bytes,
                    )
                )

        except Exception as e:
            print(
                f"Inference failed "
                f"for batch "
                f"{batch_start}"
            )

            print(e)

            skipped += len(batch_bytes)

            continue

        # -------------------------------------------------------------------
        # Memory stats
        # -------------------------------------------------------------------

        batch_rss_mb = (
            get_peak_rss_mb()
        )

        peak_rss_mb = max(
            peak_rss_mb,
            batch_rss_mb,
        )

        # -------------------------------------------------------------------
        # Accuracy
        # -------------------------------------------------------------------

        for (
            pred_labels,
            gt_wnid,
        ) in zip(
            prediction_labels,
            batch_wnids,
        ):

            if inference_engine == "task-inference":

                pred_wnids = [
                    label_to_wnid.get(
                        pred_label
                    )
                    for pred_label in pred_labels
                ]

            elif custom_predicts_wnid:

                pred_wnids = pred_labels

            else:

                pred_wnids = [
                    None
                    for _ in pred_labels
                ]

            if (
                pred_wnids
                and pred_wnids[0]
                == gt_wnid
            ):

                top1_correct += 1

            if gt_wnid in pred_wnids[:5]:

                top5_correct += 1

            total += 1

        # -------------------------------------------------------------------
        # Timing stats
        # -------------------------------------------------------------------

        batch_cpu_end = (
            process.cpu_times()
        )

        batch_cpu_seconds = (
            (
                batch_cpu_end.user
                - batch_cpu_start.user
            )
            + (
                batch_cpu_end.system
                - batch_cpu_start.system
            )
        )

        batch_wall_seconds = (
            time.perf_counter()
            - batch_wall_start
        )

        batch_cpu_usage_percent = (
            batch_cpu_seconds
            / batch_wall_seconds
            * 100.0
            if batch_wall_seconds > 0
            else 0.0
        )

        batch_cpu_seconds_values.append(
            batch_cpu_seconds
        )

        batch_cpu_usage_values.append(
            batch_cpu_usage_percent
        )

        batch_wall_seconds_values.append(
            batch_wall_seconds
        )

        batch_memory_values.append(
            batch_rss_mb
        )

        # -------------------------------------------------------------------
        # GPU stats
        # -------------------------------------------------------------------

        if cuda_enabled:

            batch_gpu_allocated_values.append(
                torch.cuda.memory_allocated()
                / (1024 * 1024)
            )

            batch_gpu_reserved_values.append(
                torch.cuda.memory_reserved()
                / (1024 * 1024)
            )

        batches_evaluated += 1

    # -----------------------------------------------------------------------
    # Final checks
    # -----------------------------------------------------------------------

    if total == 0:

        raise RuntimeError(
            "No images evaluated."
        )

    # -----------------------------------------------------------------------
    # Global stats
    # -----------------------------------------------------------------------

    cpu_end = (
        process.cpu_times()
    )

    cpu_seconds = (
        (cpu_end.user - cpu_start.user)
        + (
            cpu_end.system
            - cpu_start.system
        )
    )

    wall_seconds = (
        time.perf_counter()
        - wall_start
    )

    cpu_usage_percent = (
        cpu_seconds
        / wall_seconds
        * 100.0
        if wall_seconds > 0
        else 0.0
    )

    top1 = (
        top1_correct / total
    )

    top5_acc = (
        top5_correct / total
    )

    report = {
        "model_name": model_name,
        "top1_accuracy":
            top1,
        "top5_accuracy":
            top5_acc,
        "cpu_time_seconds":
            cpu_seconds,
        "cpu_usage_percent":
            cpu_usage_percent,
        "wall_time_seconds":
            wall_seconds,
        "batch_size":
            batch_size,
        "batches_evaluated":
            batches_evaluated,
        "device":
            device,
        "task_mode":
            task_mode,
        "task_count":
            len(tasks)
            if inference_engine
            == "task-inference"
            else 1,
        "inference_engine":
            inference_engine,
        "custom_model_name":
            custom_model_name
            if inference_engine
            == "custom"
            else None,
        "cuda_available":
            cuda_available,
        "cuda_enabled":
            cuda_enabled,
        "gpu_count":
            torch.cuda.device_count()
            if cuda_enabled
            else 0,
        "peak_memory_mb":
            peak_rss_mb,
        "dataset":
            Path(dataset_path).name,
        "dataset_images_evaluated":
            total,
        "dataset_images_skipped":
            skipped,                
        "avg_batch_cpu_time_seconds":
            sum(
                batch_cpu_seconds_values
            )
            / len(
                batch_cpu_seconds_values
            ),
        "avg_batch_cpu_usage_percent":
            sum(
                batch_cpu_usage_values
            )
            / len(
                batch_cpu_usage_values
            ),
        "avg_batch_wall_time_seconds":
            sum(
                batch_wall_seconds_values
            )
            / len(
                batch_wall_seconds_values
            ),
        "avg_batch_memory_mb":
            sum(
                batch_memory_values
            )
            / len(
                batch_memory_values
            ),
        "min_batch_cpu_time_seconds":
            min(
                batch_cpu_seconds_values
            ),
        "max_batch_cpu_time_seconds":
            max(
                batch_cpu_seconds_values
            ),
        "min_batch_cpu_usage_percent":
            min(
                batch_cpu_usage_values
            ),
        "max_batch_cpu_usage_percent":
            max(
                batch_cpu_usage_values
            ),
        "min_batch_wall_time_seconds":
            min(
                batch_wall_seconds_values
            ),
        "max_batch_wall_time_seconds":
            max(
                batch_wall_seconds_values
            ),
        "min_batch_memory_mb":
            min(
                batch_memory_values
            ),
        "max_batch_memory_mb":
            max(
                batch_memory_values
            ),
    }

    # -----------------------------------------------------------------------
    # GPU metrics
    # -----------------------------------------------------------------------

    if cuda_enabled:

        report[
            "cuda_device_name"
        ] = (
            torch.cuda.get_device_name(
                0
            )
        )

        report[
            "peak_gpu_allocated_mb"
        ] = (
            torch.cuda
            .max_memory_allocated()
            / (1024 * 1024)
        )

        report[
            "peak_gpu_reserved_mb"
        ] = (
            torch.cuda
            .max_memory_reserved()
            / (1024 * 1024)
        )

    # -----------------------------------------------------------------------
    # Save report
    # -----------------------------------------------------------------------

    output_path = Path(
        "evaluation_report.json"
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )

    artifact = project.log_artifact(
        name="evaluation_report",
        kind="table",
        source=str(output_path),
    )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    print()

    print("=" * 60)

    print(
        f"Model: {model_name}"
    )

    print(
        f"Images evaluated: "
        f"{total}"
    )

    print(
        f"Images skipped: "
        f"{skipped}"
    )

    print(
        f"Top-1 accuracy: "
        f"{top1:.4f}"
    )

    print(
        f"Top-5 accuracy: "
        f"{top5_acc:.4f}"
    )

    print(
        f"Wall time (s): "
        f"{wall_seconds:.2f}"
    )

    print(
        f"CPU time (s): "
        f"{cpu_seconds:.2f}"
    )

    print(
        f"Peak memory (MB): "
        f"{peak_rss_mb:.2f}"
    )

    if cuda_enabled:

        print(
            f"Peak GPU allocated "
            f"(MB): "
            f"{report['peak_gpu_allocated_mb']:.2f}"
        )

    print("=" * 60)

    return artifact