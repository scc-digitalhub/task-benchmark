# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from math import isfinite

from torch import Tensor


class Evaluator(ABC):
    """Base evaluator that validates scores returned by implementations."""

    def evaluate(self, test_context: str, service_response: Tensor) -> float:
        """Return a finite quality score in the inclusive range [0, 1]."""
        score = float(self._evaluate(test_context, service_response))
        if not isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"Evaluator score must be between 0 and 1, got {score}")
        return score

    @abstractmethod
    def _evaluate(self, test_context: str, service_response: Tensor) -> float:
        """Compute an implementation-specific inference quality score."""
        raise NotImplementedError