# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Invoker(ABC, Generic[InputT, OutputT]):
    """Abstract model invocation interface."""

    @abstractmethod
    def invoke(self, test_data: InputT) -> OutputT:
        """Run inference for the supplied input and return the model output."""
        raise NotImplementedError