# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from digitalhub_runtime_python import handler

from task_benchmark.core import evaluate


@handler(
    outputs=["evaluation_report"]
)
def evaluate_model(
    project,
    dataset,
    profile: str = "default",
    task: str = "image-classification",
    implementation: str = "task-inference",
    device: str = "cpu",
    **kwargs,
):
    """
    DigitalHub adapter for the benchmark evaluator.
    """

    dataset_path = dataset.download()

    output_path = Path("evaluation_report.json")

    evaluate(
        dataset_path=dataset_path,
        task=task,
        implementation=implementation,
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
