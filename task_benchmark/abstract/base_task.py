# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .base_implementation import BaseImplementation


class BaseTask(ABC):
    """
    Generic task abstraction that is independent from concrete implementations.
    """

    task_name: str
    input_column: str
    label_column: str
    description_column: str

    @property
    def required_columns(self) -> set[str]:
        return {
            self.input_column,
            self.label_column,
            self.description_column,
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

    def extract_class_descriptions(
        self,
        rows: list[dict[str, str]],
    ) -> dict[str, str]:
        class_descriptions: dict[str, str] = {}

        for row in rows:
            class_id = row[self.label_column]
            if class_id not in class_descriptions:
                class_descriptions[class_id] = row.get(
                    self.description_column,
                    "",
                )

        return class_descriptions

    def get_ground_truth(
        self,
        row: dict[str, str],
    ) -> str:
        return row[self.label_column]

    def create_task_metrics(
        self,
    ) -> dict[str, Any]:
        """
        Initialize task-specific metrics state.
        """

        return {}

    def execute_batch(
        self,
        implementation: BaseImplementation,
        batch_inputs: list[bytes],
    ) -> Any:
        """
        Execute one batch for this task and return task-specific batch output.
        """

        return None

    def record_skipped_items(
        self,
        task_metrics: dict[str, Any],
        count: int = 1,
    ) -> None:
        """
        Record skipped task items for task-specific reporting.
        """

        _ = task_metrics
        _ = count

    def update_task_metrics_from_batch(
        self,
        task_metrics: dict[str, Any],
        batch_output: Any,
        batch_gt_labels: list[str],
        label_mapping: dict[str, str],
        implementation: BaseImplementation,
    ) -> None:
        """
        Update task-specific metrics from one task-specific batch output.
        """

    def finalize_task_metrics(
        self,
        task_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Convert task-specific metrics state into report fields.
        """

        return {}

    def build_task_summary_lines(
        self,
        report: dict[str, Any],
    ) -> list[str]:
        """
        Build task-specific summary lines for console output.
        """

        return []

    @abstractmethod
    def build_label_mapping(
        self,
        implementation_name: str,
        model_name: str,
        class_descriptions: dict[str, str],
    ) -> dict[str, str]:
        """
        Build mapping from predicted labels to class ids.
        """

    @abstractmethod
    def create_implementation(
        self,
        implementation_name: str,
        model_name: str,
        custom_model_name: str,
        custom_model_import_path: str,
        class_descriptions: dict[str, str],
        device: str,
    ) -> BaseImplementation:
        """
        Create the concrete implementation for this task.
        """
