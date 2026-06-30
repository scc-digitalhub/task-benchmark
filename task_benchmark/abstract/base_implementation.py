# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC


class BaseImplementation(ABC):
    """
    Generic implementation abstraction used across tasks.

    """

    predicts_wnid: bool = False
