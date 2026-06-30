# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from typing import Any

import psutil


class RuntimeMetricsCollector:
    """
    Task-agnostic runtime metrics collector.

    This tracks CPU/wall time, process memory, and per-batch runtime stats.
    """

    def __init__(
        self,
    ) -> None:
        self.process = psutil.Process()

        self.wall_start = 0.0
        self.cpu_start = None

        self.peak_rss_mb = 0.0

        self.batch_cpu_seconds_values: list[float] = []
        self.batch_cpu_usage_values: list[float] = []
        self.batch_wall_seconds_values: list[float] = []
        self.batch_memory_values: list[float] = []

        self.batches_evaluated = 0

    @staticmethod
    def get_peak_rss_mb() -> float:
        process = psutil.Process()

        return (
            process.memory_info().rss
            / (1024 * 1024)
        )

    @staticmethod
    def is_cuda_device(
        device: str,
    ) -> bool:
        return (
            device == "cuda"
            or device.startswith("cuda:")
        )

    def start(
        self,
    ) -> None:
        self.wall_start = time.perf_counter()
        self.cpu_start = self.process.cpu_times()
        self.peak_rss_mb = self.get_peak_rss_mb()

    def start_batch(
        self,
    ) -> tuple[float, Any]:
        return (
            time.perf_counter(),
            self.process.cpu_times(),
        )

    def finish_batch(
        self,
        batch_wall_start: float,
        batch_cpu_start: Any,
    ) -> None:
        batch_rss_mb = self.get_peak_rss_mb()

        self.peak_rss_mb = max(
            self.peak_rss_mb,
            batch_rss_mb,
        )

        batch_cpu_end = self.process.cpu_times()

        batch_cpu_seconds = (
            (
                batch_cpu_end.user
                - batch_cpu_start.user
            )
            + (
                batch_cpu_end.system
                - batch_cpu_start.system
            )
        )

        batch_wall_seconds = (
            time.perf_counter()
            - batch_wall_start
        )

        batch_cpu_usage_percent = (
            batch_cpu_seconds
            / batch_wall_seconds
            * 100.0
            if batch_wall_seconds > 0
            else 0.0
        )

        self.batch_cpu_seconds_values.append(
            batch_cpu_seconds
        )

        self.batch_cpu_usage_values.append(
            batch_cpu_usage_percent
        )

        self.batch_wall_seconds_values.append(
            batch_wall_seconds
        )

        self.batch_memory_values.append(
            batch_rss_mb
        )

        self.batches_evaluated += 1

    def _safe_avg(
        self,
        values: list[float],
    ) -> float | None:
        if not values:
            return None

        return sum(values) / len(values)

    def _safe_min(
        self,
        values: list[float],
    ) -> float | None:
        if not values:
            return None

        return min(values)

    def _safe_max(
        self,
        values: list[float],
    ) -> float | None:
        if not values:
            return None

        return max(values)

    def finalize(
        self,
    ) -> dict[str, float | int | None]:
        cpu_end = self.process.cpu_times()

        cpu_seconds = (
            (cpu_end.user - self.cpu_start.user)
            + (
                cpu_end.system
                - self.cpu_start.system
            )
        )

        wall_seconds = (
            time.perf_counter()
            - self.wall_start
        )

        cpu_usage_percent = (
            cpu_seconds
            / wall_seconds
            * 100.0
            if wall_seconds > 0
            else 0.0
        )

        return {
            "cpu_time_seconds": cpu_seconds,
            "cpu_usage_percent": cpu_usage_percent,
            "wall_time_seconds": wall_seconds,
            "peak_memory_mb": self.peak_rss_mb,
            "batches_evaluated": self.batches_evaluated,
            "avg_batch_cpu_time_seconds": self._safe_avg(
                self.batch_cpu_seconds_values
            ),
            "avg_batch_cpu_usage_percent": self._safe_avg(
                self.batch_cpu_usage_values
            ),
            "avg_batch_wall_time_seconds": self._safe_avg(
                self.batch_wall_seconds_values
            ),
            "avg_batch_memory_mb": self._safe_avg(
                self.batch_memory_values
            ),
            "min_batch_cpu_time_seconds": self._safe_min(
                self.batch_cpu_seconds_values
            ),
            "max_batch_cpu_time_seconds": self._safe_max(
                self.batch_cpu_seconds_values
            ),
            "min_batch_cpu_usage_percent": self._safe_min(
                self.batch_cpu_usage_values
            ),
            "max_batch_cpu_usage_percent": self._safe_max(
                self.batch_cpu_usage_values
            ),
            "min_batch_wall_time_seconds": self._safe_min(
                self.batch_wall_seconds_values
            ),
            "max_batch_wall_time_seconds": self._safe_max(
                self.batch_wall_seconds_values
            ),
            "min_batch_memory_mb": self._safe_min(
                self.batch_memory_values
            ),
            "max_batch_memory_mb": self._safe_max(
                self.batch_memory_values
            ),
        }
