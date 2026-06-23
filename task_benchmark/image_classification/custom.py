from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType

from .models import ImageClassificationModel


LOCAL_MODELS_PACKAGE = (
    f"{__package__}.custom_models"
)


def _normalize_module_name(
    custom_model_name: str,
) -> str:
    return custom_model_name.replace(
        "-",
        "_",
    )


def _available_model_names() -> list[str]:
    package = importlib.import_module(
        LOCAL_MODELS_PACKAGE
    )

    names: list[str] = []

    for module_info in pkgutil.iter_modules(
        package.__path__
    ):
        if module_info.name.startswith("_"):
            continue

        names.append(
            module_info.name.replace(
                "_",
                "-",
            )
        )

    return sorted(names)


def list_custom_models() -> list[str]:
    """
    Return local custom model names available in custom_models package.
    """

    return _available_model_names()


def _load_custom_model_module(
    custom_model_name: str,
) -> tuple[ModuleType, str]:
    """
    Load custom model module.

    Supported values for custom_model_name:
    - Local key: "hash-baseline" -> image_classification/custom_models/hash_baseline.py
    - External module path: "my_package.my_model"
    """

    if "." in custom_model_name:
        module = importlib.import_module(
            custom_model_name
        )

        return (
            module,
            f"external module '{custom_model_name}'",
        )

    module_name = _normalize_module_name(
        custom_model_name
    )

    fq_module = (
        f"{LOCAL_MODELS_PACKAGE}.{module_name}"
    )

    try:
        module = importlib.import_module(
            fq_module
        )

        return (
            module,
            f"local module '{fq_module}'",
        )

    except ModuleNotFoundError as exc:

        if exc.name != fq_module:
            raise

        available = _available_model_names()

        available_text = (
            ", ".join(available)
            if available
            else "none"
        )

        raise ValueError(
            "Unknown custom model "
            f"'{custom_model_name}'. "
            "Use a local model key (from custom_models) or an installed "
            "external Python module path (e.g. 'my_package.my_model'). "
            f"Available local models: {available_text}."
        ) from exc


def create_custom_model(
    custom_model_name: str,
    class_descriptions: dict[str, str],
    device: str,
) -> ImageClassificationModel:
    """
    Create a user-defined custom model.

    Custom modules must expose a callable named
    `create_model(class_descriptions, device)`.

    `custom_model_name` can be:
    - local key (e.g. "hash-baseline") from image_classification/custom_models
    - external installed module path (e.g. "my_package.my_model")
    """

    module, source_text = _load_custom_model_module(
        custom_model_name
    )

    create_model = getattr(
        module,
        "create_model",
        None,
    )

    if not callable(create_model):
        raise ValueError(
            "Custom model module "
            f"'{custom_model_name}' "
            "must define callable create_model(class_descriptions, device)."
        )

    model = create_model(
        class_descriptions=class_descriptions,
        device=device,
    )

    if not isinstance(
        model,
        ImageClassificationModel,
    ):
        raise ValueError(
            "create_model(...) must return ImageClassificationModel, got "
            f"{type(model).__name__}."
        )

    setattr(
        model,
        "custom_model_source",
        source_text,
    )

    setattr(
        model,
        "custom_model_name",
        custom_model_name,
    )

    return model