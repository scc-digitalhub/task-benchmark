# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DataLoader(ABC):
    """Abstract source of named dataset items."""

    def __init__(
        self,
        dataset_id: str,
        local_path: str,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.dataset_id = dataset_id
        self.local_path = local_path
        self.params = params if params is not None else {}

    @abstractmethod
    def next(self) -> tuple[str, object]:
        """Return the next item's name and value."""
        raise NotImplementedError

    def __iter__(self) -> DataLoader:
        return self

    def __next__(self) -> tuple[str, object]:
        return self.next()