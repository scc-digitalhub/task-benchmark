# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from .data_object import BaseDataObject
from .runtime_metrics import RuntimeMetricsCollector


DataObjectT = TypeVar(
    "DataObjectT",
    bound=BaseDataObject,
)


class BaseTask(ABC, Generic[DataObjectT]): # Task
    """
    Generic task abstraction that is independent from concrete implementations.
    """

    task: str

    @abstractmethod
    def build_data_object(
        self,
        payload: dict[str, Any],
    ) -> BaseDataObject:
        """
        Build and validate a concrete task data object from a plain payload.
        """

    @abstractmethod
    def get_ground_truth(
        self,
        data_object: DataObjectT,
        index: int,
    ) -> str:
        """
        Return the expected label for one dataset sample by index.
        """

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
        implementation: str,
        data_object: BaseDataObject,
        runtime_metrics: RuntimeMetricsCollector,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Execute full task evaluation.

        This method handles all task-specific orchestration:
        - Validate and consume dataset data object
        - Prepare implementation
        - Iterate batches
        - Execute inference
        - Track metrics

        Args:
            implementation: Name of implementation to use
            data_object: Task-specific dataset object
            runtime_metrics: Runtime metrics collector
            **kwargs: Task and implementation-specific parameters

        Returns:
            Finalized report fields dict.
        """
