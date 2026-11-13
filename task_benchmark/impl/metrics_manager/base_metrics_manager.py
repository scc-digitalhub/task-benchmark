from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeAlias


JSONValue: TypeAlias = (
    str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
)


class MetricsManager(ABC):
    """Abstract source of JSON-compatible pod or service measurements."""

    @abstractmethod
    def collect_metrics(
        self,
        service_id: str,
        params: dict[str, Any] | None = None,
        *,
        window_from: int | None = None,
        window_to: int | None = None,
        metric_names: list[str] | None = None,
    ) -> JSONValue:
        """Return measurements within optional time bounds and metric-name filter."""
        raise NotImplementedError

    @staticmethod
    def _validate_window(window_from: int | None, window_to: int | None) -> None:
        for name, value in (("window_from", window_from), ("window_to", window_to)):
            if value is not None and type(value) is not int:
                raise TypeError(f"{name} must be an integer Unix timestamp or None")
        if window_from is not None and window_to is not None and window_from > window_to:
            raise ValueError("window_from must not be greater than window_to")