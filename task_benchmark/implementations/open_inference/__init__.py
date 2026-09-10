# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from .audio_classification import OpenInferenceAudioClassifier
from .image_classification import OpenInferenceImageClassifier

__all__ = ["OpenInferenceAudioClassifier", "OpenInferenceImageClassifier"]