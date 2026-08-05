# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

from digitalhub_runtime_python import handler

from task_benchmark.core import evaluate
from task_benchmark.tasks import create_task_handler


@handler(
    outputs=["evaluation_report"]
)
def evaluate_model(
    project,
    data_object: dict[str, Any],
    profile: str = "default",
    task: str = "image-classification",
    implementation: str = "task-inference",
    device: str = "cpu",
    **kwargs,
):
    """
    DigitalHub adapter for the benchmark evaluator.
    """

    task_handler = create_task_handler(task=task)

    task_data_object = task_handler.build_data_object(
        payload=data_object,
    )

    output_path = Path("evaluation_report.json")

    evaluate(
        task=task,
        implementation=implementation,
        data_object=task_data_object,
        device=device,
        report_path=output_path,
        profile=profile,
        **kwargs,
    )

    return project.log_artifact(
        name="evaluation_report",
        kind="table",
        source=str(output_path),
    )
