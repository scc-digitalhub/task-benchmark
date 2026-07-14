# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

import importlib

from .registry import implementation_registry

# Import built-in implementations for side-effect registration.
importlib.import_module("task_benchmark.implementations.task_inference")

__all__ = ["implementation_registry"]
