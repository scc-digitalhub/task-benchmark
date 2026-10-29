# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from .base_data_loader import ImageClassificationDataLoader
from .local_store import LocalStoreImageClassificationDataLoader
from .huggingface import HuggingFaceImageClassificationDataLoader

__all__ = [
	"ImageClassificationDataLoader",
	"HuggingFaceImageClassificationDataLoader",
	"LocalStoreImageClassificationDataLoader",
]