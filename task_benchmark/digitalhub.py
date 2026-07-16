# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from digitalhub_runtime_python import handler

from .core import evaluate


@handler(
    outputs=["evaluation_report"]
)
def evaluate_model(
    project,
    dataset,
    # model_name: str = "",
    # batch_size: int = 8,
    profile: str = "default",
    task: str = "image-classification",
    implementation: str = "task-inference",
    device: str = "cpu",
    # report_path: str | Path | None = None,
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
        # model_name=model_name,
        # batch_size=batch_size,
        profile=profile,
        # implementation_import_path=implementation_import_path,
        **kwargs,
    )

    return project.log_artifact(
        name="evaluation_report",
        kind="table",
        source=str(output_path),
    )
