"""Versioned, best-effort PyTorch Export graph identity for verified problems.

The probe runs in a separate process with a timeout because reference modules
are user-authored Python. Failure never establishes graph equivalence.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from functools import lru_cache
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

GRAPH_VERSION = "torch_export_aten_v1"
BACKWARD_GRAPH_VERSION = "torch_make_fx_forward_backward_v1"
_RESULT_PREFIX = "KERNELAGENT_GRAPH_IDENTITY="


def graph_digest(nodes) -> str:
    encoded = json.dumps(nodes, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()[:24]


@lru_cache(maxsize=256)
def _arguments(operator):
    import torch
    name, _, overload = operator.partition('.')
    for schema in torch._C._jit_get_schemas_for_operator(name):
        if schema.overload_name == overload:
            kinds = {arg.name: arg.type.kind() for arg in schema.arguments}
            kinds.update(enumerate(arg.type.kind() for arg in schema.arguments if not arg.kwarg_only))
            return kinds
    return {}


def canonical_nodes(nodes, tensor_specs):
    """Canonicalize exact integral floating Scalars, not arbitrary literals.

    Only ATen NumberType arguments on a known homogeneous floating-tensor
    operation qualify. The value must be exactly representable in that dtype
    and as a safe integer; signed zero and nonfinite values retain their types.
    """
    import torch
    def dtype_at(index):
        spec = tensor_specs.get(str(index))
        return spec.get('dtype') if isinstance(spec, dict) else None
    def refs(value):
        if isinstance(value, dict):
            if set(value) == {'node'}:
                yield dtype_at(value['node'])
            else:
                for item in value.values():
                    yield from refs(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from refs(item)
    result = []
    for index, node in enumerate(nodes):
        arguments = _arguments(node['op']) if node['op'].startswith('aten::') else {}
        dtypes = set(refs([node['args'], node['kwargs']]))
        dtype = dtype_at(index)
        eligible = (dtype in {'torch.float16', 'torch.bfloat16', 'torch.float32', 'torch.float64'}
                    and dtypes == {dtype})
        def scalar(value, kind):
            if (eligible and kind == 'NumberType' and type(value) is float
                    and math.isfinite(value) and value.is_integer() and abs(value) <= 2**53
                    and not (value == 0 and math.copysign(1, value) < 0)):
                converted = torch.tensor(value, dtype=getattr(torch, dtype[6:]), device='cpu').item()
                if converted == value:
                    return int(value)
            return value
        result.append({**node,
            'args': [scalar(value, arguments.get(i, ''))
                     for i, value in enumerate(node['args'])],
            'kwargs': {name: scalar(value, arguments.get(name, ''))
                       for name, value in node['kwargs'].items()}})
    return result


def same_graph(left, right) -> bool:
    """Compare derived identities without rewriting immutable recorded graphs."""
    if left.get('graph_digest') == right.get('graph_digest'):
        return True
    for graph in (left, right):
        nodes = graph.get('graph_nodes')
        if not nodes or graph_digest(nodes) != graph.get('graph_digest'):
            return False
    return len({graph_digest(canonical_nodes(graph['graph_nodes'], graph.get('graph_tensor_specs', {})))
                for graph in (left, right)}) == 1


def _tensor_spec(value: Any) -> Any:
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {
            "shape": [int(dim) for dim in value.shape],
            "dtype": str(value.dtype),
            "stride": [int(dim) for dim in value.stride()],
        }
    if isinstance(value, (list, tuple)):
        return [_tensor_spec(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return {"scalar_type": type(value).__name__, "value": value}
    return {"scalar_type": type(value).__name__}


def _run_probe(path: Path) -> dict[str, Any]:
    import torch
    from torch.fx import Node

    torch.manual_seed(0)
    spec = importlib.util.spec_from_file_location("kernelagent_reference_graph", path)
    if spec is None or spec.loader is None:
        raise ValueError("reference module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    init_args = module.get_init_inputs() if hasattr(module, "get_init_inputs") else []
    if not isinstance(init_args, (list, tuple)):
        init_args = [init_args]
    inputs = module.get_inputs()
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]
    # Preserve constructor-selected training mode: silently forcing eval can
    # identify a different computation from the reference actually verified.
    model = module.Model(*init_args)
    direction = getattr(module, "KERNELAGENT_OPTIMIZATION_TARGET", "forward")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from triton_kernel_agent.case_suite import load_cases, prepare, gradient_call, reference_contracts
    if direction in {"backward", "joint"}:
        from torch.fx.experimental.proxy_tensor import make_fx
        case = load_cases(module)[0]
        model, inputs = prepare(module, case, "cpu", direction)
        parameters = dict(model.named_parameters())
        def backward_graph(arguments, state):
            output = torch.func.functional_call(model, state, tuple(arguments))
            # The requested gradient names and supplied upstream gradient values
            # are part of the interface; do not label a forward graph backward.
            lookup_model = copy_model_with_state(state)
            invoke, _ = gradient_call(output, arguments, lookup_model, case)
            return invoke()
        def copy_model_with_state(state):
            class StateView:
                def named_parameters(self):
                    return state.items()
            return StateView()
        graph_module = make_fx(backward_graph)(inputs, parameters)
        version = BACKWARD_GRAPH_VERSION
    else:
        exported = torch.export.export(model, tuple(inputs), strict=False)
        graph_module = exported.graph_module
        version = GRAPH_VERSION
    nodes = list(graph_module.graph.nodes)
    positions = {node: index for index, node in enumerate(nodes)}

    def normalize(value: Any) -> Any:
        if isinstance(value, Node):
            return {"node": positions[value]}
        if isinstance(value, (tuple, list)):
            return [normalize(item) for item in value]
        if isinstance(value, dict):
            return {str(key): normalize(item) for key, item in sorted(value.items())}
        if isinstance(value, torch.dtype):
            return str(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)

    graph_nodes = []
    for node in nodes:
        if node.op == "placeholder":
            target = "input"
        elif node.op == "call_function":
            schema = getattr(node.target, "_schema", None)
            target = str(schema.name) + "." + str(schema.overload_name) if schema else str(node.target)
        elif node.op == "output":
            target = "output"
        elif node.op == "get_attr" and direction in {"backward", "joint"}:
            # make_fx may lift explicitly supplied upstream tensors/constants.
            target = "constant"
        else:
            # Export should normally lower Python module calls to ATen. A
            # remaining opaque node is not safe for structural matching.
            raise ValueError(f"unsupported graph node: {node.op}")
        graph_nodes.append({
            "op": target,
            "args": normalize(node.args),
            "kwargs": normalize(node.kwargs),
        })
    def output_spec(value: Any) -> Any:
        if isinstance(value, Node):
            return _tensor_spec(value.meta.get("val"))
        if isinstance(value, (tuple, list)):
            return [output_spec(item) for item in value]
        return _tensor_spec(value)

    interface = {
        "inputs": [_tensor_spec(item) for item in inputs],
        "parameters": [_tensor_spec(item) for item in model.parameters()],
        "outputs": output_spec(nodes[-1].args[0]),
    }
    if direction in {"backward", "joint"} or hasattr(module, "get_test_cases"):
        interface["direction"] = direction
        interface["cases"] = reference_contracts(module, direction)
        interface["explicit_case_suite"] = hasattr(module, "get_test_cases")
    tensor_specs = {str(positions[node]): _tensor_spec(node.meta['val'])
                    for node in nodes if 'val' in node.meta}
    graph_nodes = canonical_nodes(graph_nodes, tensor_specs)
    return {
        "version": version,
        "digest": graph_digest(graph_nodes),
        "nodes": graph_nodes,
        "interface": interface,
        # Kept outside the identity hash so historical full-graph identities
        # remain stable. Missing metadata never proves local compatibility.
        "tensor_specs": tensor_specs,
    }


def trace_reference_graph(source: str, *, timeout_s: int = 20) -> dict[str, Any] | None:
    """Return an identity only when a complete reference Model can be traced."""
    with tempfile.TemporaryDirectory(prefix="kernelagent_graph_") as directory:
        path = Path(directory) / "reference.py"
        path.write_text(source, encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, str(Path(__file__)), str(path)],
                capture_output=True, text=True, timeout=timeout_s,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
    if result.returncode != 0:
        return None
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(_RESULT_PREFIX):
            try:
                payload = json.loads(line[len(_RESULT_PREFIX):])
                return payload if payload.get("version") in {GRAPH_VERSION, BACKWARD_GRAPH_VERSION} else None
            except (ValueError, TypeError):
                return None
    return None


if __name__ == "__main__":
    try:
        print(_RESULT_PREFIX + json.dumps(_run_probe(Path(sys.argv[1])), sort_keys=True))
    except Exception as exc:
        print(f"graph probe unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
