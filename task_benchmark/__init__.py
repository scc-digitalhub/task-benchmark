# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""task-benchmark package."""

from .core import evaluate_model_paths
from .digitalhub import evaluate_model

__all__ = [
	"evaluate_model",
	"evaluate_model_paths",
]
