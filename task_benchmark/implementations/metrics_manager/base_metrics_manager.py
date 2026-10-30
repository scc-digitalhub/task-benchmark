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
    ) -> JSONValue:
        """Return measurements using optional implementation-specific request parameters."""
        raise NotImplementedError