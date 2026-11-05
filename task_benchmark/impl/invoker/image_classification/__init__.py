# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from .base_invoker import ImageClassificationInvoker
from .huggingface import HuggingFaceImageClassificationInvoker
from .open_inference import OpenInferenceImageClassificationInvoker

__all__ = [
	"ImageClassificationInvoker",
	"HuggingFaceImageClassificationInvoker",
	"OpenInferenceImageClassificationInvoker",
]