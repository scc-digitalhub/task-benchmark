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
    print(f"Task: {report['task_name']}")
    print(f"Implementation: {report['implementation_name']}")

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
    task_name: str,
    implementation_name: str,
    device: str = "cpu",
    report_path: str | Path | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Abstract task evaluation orchestrator.

    Core responsibility: coordinate task execution and resource tracking.

    Args:
        dataset_path: Path to dataset CSV
        task_name: Name of task to execute
        implementation_name: Name of implementation to use
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
        task_name=task_name
    )

    task_metrics, task_report = task_handler.execute_evaluation(
        dataset_path=dataset_path,
        implementation_name=implementation_name,
        device=device,
        runtime_metrics=runtime_metrics,
        **kwargs,
    )

    resource_metrics = runtime_metrics.finalize()

    report = {
        "task_name": task_name,
        "implementation_name": implementation_name,
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

