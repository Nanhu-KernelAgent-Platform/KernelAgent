"""Bind problem inputs and model state to a generated kernel signature."""
from __future__ import annotations
import inspect
from collections.abc import Callable, Sequence
from typing import Any

_ALIASES = {"w": "weight", "gamma": "weight", "beta": "bias"}
_MODULE_ATTRIBUTES = (
    "stride",
    "padding",
    "dilation",
    "output_padding",
    "groups",
    "eps",
    "num_groups",
    "normalized_shape",
    "kernel_size",
)


class KernelCallBindingError(ValueError):
    """Raised when a required kernel parameter has no unambiguous source."""


def _set_leaf_alias(values: dict[str, Any], qualified_name: str, value: Any) -> None:
    values[qualified_name] = value
    values.setdefault(qualified_name.rsplit(".", 1)[-1], value)


def collect_model_values(model: Any | None) -> tuple[dict[str, Any], list[Any]]:
    """Collect named parameters, buffers, and common module configuration."""
    if model is None:
        return {}, []
    values: dict[str, Any] = {}
    ordered_parameters: list[Any] = []
    for name, value in model.named_parameters():
        ordered_parameters.append(value)
        _set_leaf_alias(values, name, value)
    for name, value in model.named_buffers():
        _set_leaf_alias(values, name, value)
    for _, module in model.named_modules():
        for name in _MODULE_ATTRIBUTES:
            if hasattr(module, name):
                values.setdefault(name, getattr(module, name))
    for alias, canonical in _ALIASES.items():
        if canonical in values:
            values.setdefault(alias, values[canonical])
    return values, ordered_parameters


def bind_kernel_call(
    kernel_function: Callable[..., Any], model: Any | None, inputs: Sequence[Any]
) -> Callable[[], Any]:
    """Return a zero-argument call with inputs and model values bound by name.

    ``get_init_inputs()`` is intentionally absent: those values construct the
    model and are never guessed to be kernel arguments.
    """
    signature = inspect.signature(kernel_function)
    parameters = list(signature.parameters.values())
    model_values, ordered_parameters = collect_model_values(model)
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in parameters):
        positional = [*inputs, *ordered_parameters]
        return lambda: kernel_function(*positional)
    positional: list[Any] = []
    keyword: dict[str, Any] = {}
    input_index = 0
    missing: list[str] = []
    for parameter in parameters:
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        source_name = _ALIASES.get(parameter.name, parameter.name)
        if source_name in model_values:
            value = model_values[source_name]
        elif input_index < len(inputs):
            value = inputs[input_index]
            input_index += 1
        elif parameter.default is not inspect.Parameter.empty:
            continue
        else:
            missing.append(parameter.name)
            continue
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            positional.append(value)
        else:
            keyword[parameter.name] = value
    if missing:
        available = ", ".join(sorted(model_values)) or "none"
        raise KernelCallBindingError(
            f"Cannot bind required kernel parameter(s): {', '.join(missing)}; "
            f"available model values: {available}"
        )
    return lambda: kernel_function(*positional, **keyword)
