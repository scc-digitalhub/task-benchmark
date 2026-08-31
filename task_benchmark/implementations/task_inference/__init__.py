# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

import importlib

# Import modules for side-effect self-registration.
importlib.import_module(
	"task_benchmark.implementations.task_inference.image_classification"
)
importlib.import_module(
	"task_benchmark.implementations.task_inference.audio_classification"
)
