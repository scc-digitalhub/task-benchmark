from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .base_metrics_manager import JSONValue, MetricsManager


class MockupMetricsManager(MetricsManager):
    """Return a fixed JSON response loaded from a local file.

    The fixture is loaded once. Every service ID receives the same response,
    including its original metadata and timestamps, without filtering.
    """

    def __init__(self, metrics_path: str | Path) -> None:
        with Path(metrics_path).expanduser().open(encoding="utf-8") as source:
            self._metrics: JSONValue = json.load(source)

    def collect_metrics(
        self,
        service_id: str,
        params: dict[str, Any] | None = None,
    ) -> JSONValue:
        """Return an independent copy of the fixture, ignoring request params."""
        if not isinstance(service_id, str):
            raise TypeError("service_id must be a string")
        if not service_id.strip():
            raise ValueError("service_id must not be empty")
        return deepcopy(self._metrics)