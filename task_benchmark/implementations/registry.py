# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any


class ImplementationRegistry:
    """
    Global registry for task implementations.

    Keys are (task, implementation) pairs and values are
    implementation classes.
    """

    def __init__(self) -> None:
        self._registry: dict[tuple[str, str], type[Any]] = {}

    def register(
        self,
        task: str,
        implementation: str,
        implementation_cls: type[Any],
    ) -> None:
        key = (task, implementation)

        existing = self._registry.get(key)

        if existing is implementation_cls:
            return

        if existing is not None:
            raise ValueError(
                "Implementation already registered for "
                f"task='{task}', implementation='{implementation}'."
            )

        self._registry[key] = implementation_cls

    def get(
        self,
        task: str,
        implementation: str,
    ) -> type[Any]:
        key = (task, implementation)

        implementation_cls = self._registry.get(key)

        if implementation_cls is None:
            available = [
                impl_name
                for t, impl_name in self._registry.keys()
                if t == task
            ]
            available_text = ", ".join(sorted(available)) or "none"
            raise ValueError(
                "Unsupported implementation "
                f"'{implementation}' for task '{task}'. "
                f"Available implementations: {available_text}."
            )

        return implementation_cls

    def create(
        self,
        task: str,
        implementation: str,
        **kwargs,
    ) -> Any:
        implementation_cls = self.get(
            task=task,
            implementation=implementation,
        )

        return implementation_cls(**kwargs)


implementation_registry = ImplementationRegistry()
