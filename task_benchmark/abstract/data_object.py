# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaseDataObject:
    """
    Base data object shared by all tasks.

    Concrete tasks should subclass this and define required fields.
    """
