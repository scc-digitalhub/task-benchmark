# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .runtime_metrics import RuntimeMetricsCollector


class BaseTask(ABC): # Task
    """
    Generic task abstraction that is independent from concrete implementations.
    """

    task: str
    input_column: str
    label_column: str

    @property
    def required_columns(self) -> set[str]:
        return {
            self.input_column,
            self.label_column,
        }

    def load_dataset_rows(
        self,
        dataset_csv: Path,
    ) -> list[dict[str, str]]:
        if not dataset_csv.is_file():
            raise FileNotFoundError(
                f"Could not find dataset CSV: {dataset_csv}"
            )

        with dataset_csv.open(newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        if not rows:
            raise RuntimeError(
                f"Dataset CSV is empty: {dataset_csv}"
            )

        if not self.required_columns.issubset(set(rows[0].keys())):
            raise ValueError(
                "Dataset CSV must contain columns: "
                f"{sorted(self.required_columns)}"
            )

        return rows

    def resolve_input_path(
        self,
        input_value: str,
        inputs_dir: Path,
    ) -> Path:
        raw_path = Path(input_value)

        if raw_path.is_absolute() and raw_path.exists():
            return raw_path

        candidate = Path(inputs_dir) / raw_path
        if candidate.exists():
            return candidate

        candidate = Path(inputs_dir) / raw_path.name
        if candidate.exists():
            return candidate

        parts = raw_path.parts

        for idx in range(len(parts)):
            candidate = Path(inputs_dir).joinpath(*parts[idx:])
            if candidate.exists():
                return candidate

        return raw_path

    def get_ground_truth(
        self,
        row: dict[str, str],
    ) -> str:
        return row[self.label_column]

    @abstractmethod
    def build_task_summary_lines(
        self,
        report: dict[str, Any],
    ) -> list[str]:
        """
        Build task-specific summary lines for console output.
        """

    @abstractmethod
    def execute_evaluation(
        self,
        dataset_path: Path,
        implementation: str,
        device: str,
        runtime_metrics: RuntimeMetricsCollector,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Execute full task evaluation.

        This method handles all task-specific orchestration:
        - Load dataset rows
        - Prepare implementation
        - Iterate batches
        - Execute inference
        - Track metrics

        Returns:
            Finalized report fields dict.
        """
