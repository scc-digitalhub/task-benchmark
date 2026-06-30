# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from .factory import create_image_classifier
from .models import (
    ImageClassificationModel,
    Prediction,
)

__all__ = [
    "create_image_classifier",
    "ImageClassificationModel",
    "Prediction",
]
