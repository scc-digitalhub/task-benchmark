# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from task_benchmark.abstract import BaseDataObject, RuntimeMetricsCollector
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
    task: str,
    implementation: str,
    data_object: BaseDataObject,
    profile: str = "default",
    report_path: str | Path | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Abstract task evaluation orchestrator.

    Core responsibility: coordinate task execution and resource tracking.

    Args:
        task: Name of task to execute
        implementation: Name of implementation to use
        data_object: Task-specific dataset object
        profile: Resource profile to use (default, high-performance, etc.)
        report_path: Optional path to save JSON report
        **kwargs: Task and implementation-specific parameters

    Returns:
        Evaluation report dict
    """

    print("Starting evaluation job...")

    runtime_metrics = RuntimeMetricsCollector()
    runtime_metrics.start()

    task_handler = create_task_handler(
        task=task
    )

    task_report = task_handler.execute_evaluation(
        implementation=implementation,
        data_object=data_object,
        runtime_metrics=runtime_metrics,
        **kwargs,
    )

    resource_metrics = runtime_metrics.finalize()

    report = {
        "task": task,
        "implementation": implementation,
        "profile": profile,
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

