# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ..abstract import BaseTask

from .image_classification import ImageClassificationTask


_TASKS: dict[str, type[BaseTask]] = {
    "image-classification": ImageClassificationTask,
}


def create_task_handler(
    task: str,
) -> BaseTask:
    task_cls = _TASKS.get(task)

    if task_cls is None:
        available = ", ".join(sorted(_TASKS))
        raise ValueError(
            f"Unsupported task '{task}'. Available tasks: {available}."
        )

    return task_cls()
