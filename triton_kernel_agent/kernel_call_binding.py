"""Bind problem inputs and model state to a generated kernel signature."""
from __future__ import annotations
import ast
import inspect
import re
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


def _resolve_state_alias(name: str, values: dict[str, Any]) -> Any | None:
    """Match generated ``module_weight`` names to qualified model state."""
    stem, separator, leaf = name.rpartition("_")
    if not separator or leaf not in {"weight", "bias"}:
        return None
    qualified = [
        (key, value) for key, value in values.items()
        if key.endswith(f".{leaf}")
    ]
    exact = [
        value for key, value in qualified
        if key.rsplit(".", 1)[0].replace(".", "_") == stem
    ]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise KernelCallBindingError(f"Ambiguous model state for {name}")
    numbered = [
        value for key, value in qualified
        if re.fullmatch(re.escape(stem) + r"\d+", key.rsplit(".", 1)[0])
    ]
    if len(numbered) > 1:
        raise KernelCallBindingError(f"Ambiguous model state for {name}")
    return numbered[0] if numbered else None


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
        elif (state_value := _resolve_state_alias(source_name, model_values)) is not None:
            value = state_value
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


# ---------------------------------------------------------------------------
# Static bindability check (seed-extraction time, before any verification round)
# ---------------------------------------------------------------------------

def _parse_text(source: str) -> ast.Module | None:
    try:
        return ast.parse(source or "")
    except SyntaxError:
        return None


def _top_level(tree: ast.Module, name: str) -> ast.AST | None:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and node.name == name:
            return node
    return None


def _import_aliases(tree: ast.Module) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name.split(".")[0]] = (
                    item.name if item.asname else item.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom):
            for item in node.names:
                aliases[item.asname or item.name] = (node.module or "") + "." + item.name
    return aliases


def _statement_returns(statements: list[ast.stmt]) -> list[ast.Return]:
    """Return statements of a function body, excluding nested definitions."""
    found: list[ast.Return] = []
    for stmt in statements:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(stmt, ast.Return):
            found.append(stmt)
            continue
        for field in ("body", "orelse", "finalbody"):
            inner = getattr(stmt, field, None)
            if isinstance(inner, list):
                found.extend(_statement_returns(inner))
        for handler in getattr(stmt, "handlers", []):
            found.extend(_statement_returns(handler.body))
    return found


def _literal_arity(node: ast.expr | None) -> int | None:
    """Element count of a ``[a, b]``/``(a, b)`` literal, else None."""
    if isinstance(node, (ast.List, ast.Tuple)):
        if any(isinstance(element, ast.Starred) for element in node.elts):
            return None
        return len(node.elts)
    return None


def _get_inputs_arity(problem_tree: ast.Module) -> int | None:
    """How many values ``bind_kernel_call`` can pass positionally."""
    factory = _top_level(problem_tree, "get_inputs")
    if not isinstance(factory, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    returns = _statement_returns(factory.body)
    # Conditional/early returns make the arity path-dependent; stay silent.
    if len(returns) != 1 or factory.body[-1] is not returns[0]:
        return None
    value = returns[0].value
    arity = _literal_arity(value)
    if arity is not None:
        return arity
    if isinstance(value, ast.Name):
        # ``inputs = [x, y]; return inputs`` -- resolve one level of indirection.
        bound: ast.expr | None = None
        for stmt in factory.body:
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                if any(isinstance(t, ast.Name) and t.id == value.id for t in targets):
                    bound = stmt.value
        return _literal_arity(bound)
    return None


def _bindable_names(problem_tree: ast.Module) -> set[str] | None:
    """Superset of the names ``collect_model_values`` can expose for this Model.

    Over-approximating is deliberate: a name that is claimed bindable here but
    is not at runtime can only hide a genuine mismatch (a missed catch), while
    under-approximating would reject a kernel the runtime accepts.
    """
    model = _top_level(problem_tree, "Model")
    if not isinstance(model, ast.ClassDef):
        return None
    init = next(
        (n for n in model.body
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "__init__"),
        None,
    )
    if init is None or init.args.vararg or init.args.kwarg:
        return None
    aliases = _import_aliases(problem_tree)
    names: set[str] = {*_ALIASES, *_ALIASES.values()}
    has_module = False
    # ``layer = nn.Conv2d(...)`` then ``self.conv = layer`` keeps the module in a
    # local first; remembering which locals hold modules keeps that form
    # recognized without treating plain configuration (``self.dim = dim``) as
    # bindable state.
    local_modules: set[str] = set()
    for stmt in init.body:
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            value = stmt.value
            if not isinstance(value, ast.Call):
                continue
            path = _dotted(value.func, aliases)
            leaf = path.rsplit(".", 1)[-1]
            if path.startswith("torch.nn.") and leaf not in {"Parameter", "Sequential",
                                                            "ModuleList", "ModuleDict"}:
                has_module = True
                local_modules.update(
                    t.id for t in targets if isinstance(t, ast.Name))
    for stmt in init.body:
        # Dynamic state construction defeats any static bound.
        for node in ast.walk(stmt):
            if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp,
                                 ast.GeneratorExp, ast.Import, ast.ImportFrom)):
                return None
            if isinstance(node, ast.Call):
                path = _dotted(node.func, aliases)
                if path.rsplit(".", 1)[-1] in {"setattr", "add_module",
                                               "ModuleList", "ModuleDict"}:
                    return None
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            if len(targets) != 1:
                return None
            target = targets[0]
            if not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                    and target.value.id == "self"):
                continue
            value = stmt.value
            if not isinstance(value, ast.Call):
                # ``self.dim = dim`` is configuration, not model state: the
                # runtime cannot bind it, so it must be baked into the kernel.
                # ``self.conv = layer`` (a module built above) is state.
                if not (isinstance(value, ast.Name) and value.id in local_modules):
                    continue
                names.add(target.attr)
                names.add(f"{target.attr}_weight")
                names.add(f"{target.attr}_bias")
                continue
            path = _dotted(value.func, aliases)
            leaf = path.rsplit(".", 1)[-1]
            if leaf in {"Sequential", "ModuleList", "ModuleDict"}:
                return None
            names.add(target.attr)
            if path.startswith("torch.nn.") and leaf != "Parameter":
                has_module = True
                names.add(f"{target.attr}_weight")
                names.add(f"{target.attr}_bias")
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            leaf = _dotted(call.func, aliases).rsplit(".", 1)[-1]
            if leaf in {"register_buffer", "register_parameter"}:
                if not (call.args and isinstance(call.args[0], ast.Constant)
                        and isinstance(call.args[0].value, str)):
                    return None
                names.add(call.args[0].value)
    if has_module:
        names |= set(_MODULE_ATTRIBUTES)
    return names


def _dotted(node: ast.AST, aliases: dict[str, str]) -> str:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        return _dotted(node.value, aliases) + "." + node.attr
    return ""


def kernel_signature_bindability(kernel_source: str, problem_source: str) -> str | None:
    """Return why ``bind_kernel_call`` cannot fill ``kernel_function``, or None.

    ``bind_kernel_call`` fills every kernel parameter either by name from the
    reference model's state (named parameters/buffers, ``w``/``gamma``/``beta``
    aliases, module configuration) or positionally from ``get_inputs()``; a
    parameter left over without a default raises
    ``Cannot bind required kernel parameter(s)`` at *runtime* -- one whole
    verification round (25-30 min) after the seed was already accepted (P57
    round 6 declared ``stride_h``/``padding_w``/``batch_size``/... on top of the
    single tensor input). Reject such a seed statically instead.

    Deliberately conservative: reproduces the binding decision only when the
    kernel signature and the problem's ``get_inputs()``/``Model`` are readable
    unambiguously, and treats the name sources as a superset of the runtime's
    (so it can miss a genuine mismatch but cannot reject a bindable signature).
    """
    kernel_tree = _parse_text(kernel_source)
    problem_tree = _parse_text(problem_source)
    if kernel_tree is None or problem_tree is None:
        return None
    signature = _top_level(kernel_tree, "kernel_function")
    if not isinstance(signature, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    arguments = signature.args
    if arguments.vararg is not None:
        # bind_kernel_call feeds every source positionally; it cannot run out.
        return None
    positional = [*arguments.posonlyargs, *arguments.args]
    defaulted = set()
    if arguments.defaults:
        defaulted = set(range(len(positional) - len(arguments.defaults), len(positional)))
    bindable = _bindable_names(problem_tree)
    arity = _get_inputs_arity(problem_tree)
    if bindable is None or arity is None:
        return None

    def resolves(name: str) -> bool:
        return _ALIASES.get(name, name) in bindable

    slots = arity
    missing: list[str] = []
    for index, parameter in enumerate(positional):
        if resolves(parameter.arg):
            continue
        if slots:
            slots -= 1
            continue
        if index in defaulted:
            continue
        missing.append(parameter.arg)
    for parameter, default in zip(arguments.kwonlyargs, arguments.kw_defaults):
        if resolves(parameter.arg) or default is not None:
            continue
        missing.append(parameter.arg)
    if not missing:
        return None
    return (
        "kernel_function declares parameter(s) that bind_kernel_call cannot "
        f"supply: {', '.join(missing)}. Every consumer binds parameters by name "
        "from model state (named parameters/buffers, w/gamma/beta aliases, and "
        f"module configuration such as {', '.join(_MODULE_ATTRIBUTES)}) and "
        f"positionally from the {arity} value(s) returned by get_inputs(); "
        "anything else must have a default. Bake get_init_inputs() values, "
        "per-dimension sizes, and derived output shapes into kernel.py as module "
        "constants or closures instead of declaring them as parameters."
    )
