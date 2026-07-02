# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import torch
from digitalhub_runtime_python import handler

from task_benchmark.abstract import RuntimeMetricsCollector
from task_benchmark.tasks import create_task_handler


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TASK_NAME = "image-classification"
DEFAULT_BATCH_SIZE = 8
DEFAULT_IMPLEMENTATION_NAME = "task-inference"


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
    model_name: str = "",
    batch_size: int = (
        DEFAULT_BATCH_SIZE
    ),
    device: str = "cpu",
    task_name: str = (
        DEFAULT_TASK_NAME
    ),
    implementation_name: str = (
        DEFAULT_IMPLEMENTATION_NAME
    ),
    custom_model_name: str = (
        "hash-baseline"
    ),
    custom_model_import_path: str = "",
):
    """
    Evaluate a model by flowing through 3 levels:
    1) abstract level (task-independent runtime handler)
    2) task level (task-specific adapter)
    3) implementation level (task-inference/custom backend)
    """

    print(
        "Starting evaluation job..."
    )

    selected_implementation = implementation_name

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
        RuntimeMetricsCollector.is_cuda_device(device)
        and not cuda_available
    ):

        raise RuntimeError(
            "CUDA requested but "
            "not available."
        )

    cuda_enabled = (
        RuntimeMetricsCollector.is_cuda_device(device)
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
    # Benchmark init (abstract-level resource metrics)
    # -----------------------------------------------------------------------

    runtime_metrics = RuntimeMetricsCollector()
    runtime_metrics.start()

    # -----------------------------------------------------------------------
    # First level: abstract task selection
    # -----------------------------------------------------------------------

    task_handler = create_task_handler(
        task_name=task_name
    )

    # -----------------------------------------------------------------------
    # Second level: task-specific data/mapping setup
    # -----------------------------------------------------------------------

    rows = task_handler.load_dataset_rows(
        Path(dataset_path)
    )

    print(
        f"Loaded {len(rows)} "
        f"dataset rows."
    )

    class_descriptions = task_handler.extract_class_descriptions(
        rows=rows
    )

    label_mapping = task_handler.build_label_mapping(
        implementation_name=selected_implementation,
        model_name=model_name,
        class_descriptions=class_descriptions,
    )

    # -----------------------------------------------------------------------
    # Third level: concrete implementation setup
    # -----------------------------------------------------------------------

    classifier = task_handler.create_implementation(
        implementation_name=selected_implementation,
        model_name=model_name,
        custom_model_name=custom_model_name,
        custom_model_import_path=custom_model_import_path,
        class_descriptions=class_descriptions,
        device=device,
    )

    if selected_implementation == "custom":

        print(
            "Custom inference selected"
        )

        print(
            f"Custom model key/module: "
            f"{custom_model_name}"
        )

        if custom_model_import_path:
            print(
                "Custom model import path:",
                custom_model_import_path,
            )

        print(
            "Custom model loaded from:",
            getattr(
                classifier,
                "custom_model_source",
                "unknown source",
            ),
        )

    predicts_task_labels = (
        classifier.predicts_wnid
    )

    task_mode = getattr(
        classifier,
        "task_mode",
        selected_implementation,
    )

    # -----------------------------------------------------------------------
    # Task metrics
    # -----------------------------------------------------------------------

    task_metrics = task_handler.create_task_metrics()

    skipped = 0

    # -----------------------------------------------------------------------
    # Evaluation loop
    # -----------------------------------------------------------------------

    for batch_start in range(
        0,
        len(rows),
        batch_size,
    ):

        batch_wall_start, batch_cpu_start = (
            runtime_metrics.start_batch()
        )

        batch_rows = rows[
            batch_start:
            batch_start + batch_size
        ]

        batch_inputs: list[bytes] = []

        batch_gt_labels: list[str] = []

        for row in batch_rows:
            input_path = task_handler.resolve_input_path(
                row[task_handler.input_column],
                Path(images_dir_path),
            )

            gt_label = task_handler.get_ground_truth(
                row
            )

            if not input_path.is_file():

                print(
                    f"[WARNING] "
                    f"Missing input file: "
                    f"{input_path}"
                )

                skipped += 1

                continue

            try:
                batch_inputs.append(
                    input_path.read_bytes()
                )

                batch_gt_labels.append(
                    gt_label
                )

            except Exception as exc:
                print(
                    f"Failed reading input file: "
                    f"{input_path}"
                )

                print(exc)

                skipped += 1

        if not batch_inputs:
            continue

        # -------------------------------------------------------------------
        # Inference
        # -------------------------------------------------------------------

        try:
            predictions = (
                classifier.predict_batch(
                    batch_inputs,
                    top_k=5,
                )
            )

            prediction_labels = [
                [
                    prediction.label
                    for prediction in per_image
                ]
                for per_image in predictions
            ]
        except Exception as exc:
            print(
                f"Inference failed "
                f"for batch "
                f"{batch_start}"
            )

            print(exc)

            skipped += len(batch_inputs)

            continue

        # -------------------------------------------------------------------
        # Task-level metrics (e.g. top-k for image classification)
        # -------------------------------------------------------------------

        for (
            pred_labels,
            gt_label,
        ) in zip(
            prediction_labels,
            batch_gt_labels,
        ):

            task_handler.update_task_metrics(
                task_metrics=task_metrics,
                pred_labels=pred_labels,
                gt_label=gt_label,
                predicts_task_labels=predicts_task_labels,
                label_mapping=label_mapping,
            )

        # -------------------------------------------------------------------
        # Abstract-level resource metrics
        # -------------------------------------------------------------------
        runtime_metrics.finish_batch(
            batch_wall_start=batch_wall_start,
            batch_cpu_start=batch_cpu_start,
        )

    # -----------------------------------------------------------------------
    # Global stats
    # -----------------------------------------------------------------------

    resource_metrics = runtime_metrics.finalize()
    task_report_metrics = task_handler.finalize_task_metrics(
        task_metrics=task_metrics
    )

    report = {
        "task_name": task_name,
        "model_name": model_name,
        **task_report_metrics,
        **resource_metrics,
        "batch_size":
            batch_size,
        "device":
            device,
        "task_mode":
            task_mode,
        "task_count": getattr(
            classifier,
            "task_count",
            1,
            ),
        "implementation_name":
            selected_implementation,
        "custom_model_name":
            custom_model_name
            if selected_implementation
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
        "dataset":
            Path(dataset_path).name,
        "dataset_images_skipped":
            skipped,
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
        f"Task: {task_name}"
    )

    print(
        f"Implementation: {selected_implementation}"
    )

    print(
        f"Model: {model_name}"
    )

    for summary_line in task_handler.build_task_summary_lines(
        report=report
    ):
        print(summary_line)

    print(
        f"Images skipped: "
        f"{skipped}"
    )

    print(
        f"Wall time (s): "
        f"{float(report['wall_time_seconds']):.2f}"
    )

    print(
        f"CPU time (s): "
        f"{float(report['cpu_time_seconds']):.2f}"
    )

    print(
        f"Peak memory (MB): "
        f"{float(report['peak_memory_mb']):.2f}"
    )

    if cuda_enabled:

        print(
            f"Peak GPU allocated "
            f"(MB): "
            f"{report['peak_gpu_allocated_mb']:.2f}"
        )

    print("=" * 60)

    return artifact
