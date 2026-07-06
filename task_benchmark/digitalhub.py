# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from digitalhub_runtime_python import handler

from .core import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_IMPLEMENTATION_NAME,
    DEFAULT_TASK_NAME,
    evaluate_model_paths,
)


@handler(
    outputs=["evaluation_report"]
)
def evaluate_model(
    project,
    dataset,
    task_inputs_dir=None,
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
    DigitalHub adapter for the benchmark evaluator.
    """

    dataset_path = dataset.download()

    if task_inputs_dir is None:
        raise ValueError(
            "Missing required input directory. "
            "Provide task_inputs_dir."
        )

    task_inputs_dir_path = task_inputs_dir.download()
    output_path = Path("evaluation_report.json")

    evaluate_model_paths(
        dataset_path=dataset_path,
        task_inputs_dir_path=task_inputs_dir_path,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        task_name=task_name,
        implementation_name=implementation_name,
        custom_model_name=custom_model_name,
        custom_model_import_path=custom_model_import_path,
        report_path=output_path,
    )

    return project.log_artifact(
        name="evaluation_report",
        kind="table",
        source=str(output_path),
    )
