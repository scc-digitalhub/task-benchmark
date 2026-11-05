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
        *,
        max_samples: int | None = None,
    ) -> None:
        if max_samples is not None and type(max_samples) is not int:
            raise TypeError("max_samples must be an integer or None")
        if max_samples is not None and max_samples < 1:
            raise ValueError("max_samples must be greater than zero")

        self.dataset_id = dataset_id
        self.local_path = local_path
        self.params = params if params is not None else {}
        self.max_samples = max_samples
        self._yielded_samples = 0

    @abstractmethod
    def next(self) -> tuple[str, object]:
        """Return the next item's name and value."""
        raise NotImplementedError

    def __iter__(self) -> DataLoader:
        return self

    def __next__(self) -> tuple[str, object]:
        if (
            self.max_samples is not None
            and self._yielded_samples >= self.max_samples
        ):
            raise StopIteration
        item = self.next()
        self._yielded_samples += 1
        return item