# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from task_benchmark.abstract import RuntimeMetricsCollector
from task_benchmark.tasks import create_task_handler


def _print_summary(
    report: dict[str, Any],
    task_handler,
) -> None:
    print()
    print("=" * 60)
    print(f"Task: {report['task']}")
    print(f"Implementation: {report['implementation']}")

    for summary_line in task_handler.build_task_summary_lines(
        report=report
    ):
        print(summary_line)

    for summary_line in RuntimeMetricsCollector.build_summary_lines(
        report=report
    ):
        print(summary_line)

    print("=" * 60)


def evaluate(
    dataset_path: str | Path,
    task: str,
    implementation: str,
    device: str = "cpu",
    report_path: str | Path | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Abstract task evaluation orchestrator.

    Core responsibility: coordinate task execution and resource tracking.

    Args:
        dataset_path: Path to dataset CSV
        task: Name of task to execute
        implementation: Name of implementation to use
        device: Compute device (cpu, cuda, cuda:0, etc.)
        report_path: Optional path to save JSON report
        **kwargs: Task and implementation-specific parameters

    Returns:
        Evaluation report dict
    """

    print("Starting evaluation job...")

    dataset_path = Path(dataset_path)

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    runtime_metrics = RuntimeMetricsCollector()
    runtime_metrics.start()

    task_handler = create_task_handler(
        task=task
    )

    task_report = task_handler.execute_evaluation(
        dataset_path=dataset_path,
        implementation=implementation,
        device=device,
        runtime_metrics=runtime_metrics,
        **kwargs,
    )

    resource_metrics = runtime_metrics.finalize()

    report = {
        "task": task,
        "implementation": implementation,
        "device": device,
        **task_report,
        **resource_metrics,
    }

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
    )

    return report

