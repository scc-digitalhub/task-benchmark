# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""task-benchmark package."""

from .core import evaluate

__all__ = ["evaluate"]

try:
	from .digitalhub import evaluate_model
	__all__ = ["evaluate", "evaluate_model"]
except ModuleNotFoundError:
	pass
