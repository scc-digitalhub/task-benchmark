# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from task_benchmark.abstract import RuntimeMetricsCollector
from task_benchmark.tasks import create_task_handler


DEFAULT_TASK_NAME = "image-classification"
DEFAULT_BATCH_SIZE = 8
DEFAULT_IMPLEMENTATION_NAME = "task-inference"


def _print_summary(
    report: dict[str, Any],
    task_handler,
    implementation_name: str,
) -> None:
    print()
    print("=" * 60)
    print(f"Task: {report['task_name']}")
    print(f"Implementation: {implementation_name}")
    print(f"Model: {report['model_name']}")

    for summary_line in task_handler.build_task_summary_lines(
        report=report
    ):
        print(summary_line)

    for summary_line in RuntimeMetricsCollector.build_summary_lines(
        report=report
    ):
        print(summary_line)

    print("=" * 60)


def evaluate_model_paths(
    dataset_path: str | Path,
    task_inputs_dir_path: str | Path,
    model_name: str = "",
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = "cpu",
    task_name: str = DEFAULT_TASK_NAME,
    implementation_name: str = DEFAULT_IMPLEMENTATION_NAME,
    custom_model_name: str = "hash-baseline",
    custom_model_import_path: str = "",
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Evaluate a task using local filesystem paths.

    This is the plain-Python entrypoint and does not depend on DigitalHub
    dataset/artifact wrappers.
    """

    print(
        "Starting evaluation job..."
    )

    selected_implementation = implementation_name
    dataset_path = Path(dataset_path)
    task_inputs_dir_path = Path(task_inputs_dir_path)

    print(
        f"Dataset CSV path: "
        f"{dataset_path}"
    )

    print(
        f"Task inputs directory path: "
        f"{task_inputs_dir_path}"
    )

    if not task_inputs_dir_path.exists():
        raise RuntimeError(
            f"Task inputs directory "
            f"does not exist: "
            f"{task_inputs_dir_path}"
        )

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

    runtime_metrics = RuntimeMetricsCollector()
    runtime_metrics.start()

    task_handler = create_task_handler(
        task_name=task_name
    )

    rows = task_handler.load_dataset_rows(
        dataset_path
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

    task_mode = getattr(
        classifier,
        "task_mode",
        selected_implementation,
    )

    task_metrics = task_handler.create_task_metrics()

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
                task_inputs_dir_path,
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
                task_handler.record_skipped_items(
                    task_metrics=task_metrics,
                )
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
                task_handler.record_skipped_items(
                    task_metrics=task_metrics,
                )

        if not batch_inputs:
            continue

        try:
            batch_output = task_handler.execute_batch(
                implementation=classifier,
                batch_inputs=batch_inputs,
            )
        except Exception as exc:
            print(
                f"Task execution failed "
                f"for batch "
                f"{batch_start}"
            )
            print(exc)
            task_handler.record_skipped_items(
                task_metrics=task_metrics,
                count=len(batch_inputs),
            )
            continue

        task_handler.update_task_metrics_from_batch(
            task_metrics=task_metrics,
            batch_output=batch_output,
            batch_gt_labels=batch_gt_labels,
            label_mapping=label_mapping,
            implementation=classifier,
        )

        runtime_metrics.finish_batch(
            batch_wall_start=batch_wall_start,
            batch_cpu_start=batch_cpu_start,
        )

    resource_metrics = runtime_metrics.finalize()
    task_report_metrics = task_handler.finalize_task_metrics(
        task_metrics=task_metrics
    )

    report = {
        "task_name": task_name,
        "model_name": model_name,
        **task_report_metrics,
        **resource_metrics,
        "batch_size": batch_size,
        "device": device,
        "task_mode": task_mode,
        "task_count": getattr(
            classifier,
            "task_count",
            1,
        ),
        "implementation_name": selected_implementation,
        "custom_model_name": (
            custom_model_name
            if selected_implementation == "custom"
            else None
        ),
        "cuda_available": cuda_available,
        "cuda_enabled": cuda_enabled,
        "gpu_count": (
            torch.cuda.device_count()
            if cuda_enabled
            else 0
        ),
        "dataset": dataset_path.name,
    }

    if cuda_enabled:
        report["cuda_device_name"] = (
            torch.cuda.get_device_name(0)
        )
        report["peak_gpu_allocated_mb"] = (
            torch.cuda.max_memory_allocated()
            / (1024 * 1024)
        )
        report["peak_gpu_reserved_mb"] = (
            torch.cuda.max_memory_reserved()
            / (1024 * 1024)
        )

    if report_path is not None:
        report_path = Path(report_path)
        report_path.write_text(
            json.dumps(
                report,
                indent=2,
            )
        )

    _print_summary(
        report=report,
        task_handler=task_handler,
        implementation_name=selected_implementation,
    )

    return report
