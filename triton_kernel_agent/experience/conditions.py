"""Bounded condition checks against facts, not operator names or model prose.

All input tensors must satisfy a condition. Lists denote allowed values; shape
dimensions allow None or inclusive min/max bounds. Unsupported conditions are
unknown, not unconditional. Unspecified shapes are deliberately not exact gates.
"""
from .models import OperatorSignature

VERSION = "observed_applicability_v1"


def _shape(value):
    return isinstance(value, (list, tuple)) and all(
        isinstance(size, int) and not isinstance(size, bool) and size >= 0 for size in value)


def context(signature: OperatorSignature, *, hardware="", layout="", direction="unknown", architecture=""):
    result = {"hardware": hardware, "direction": direction,
              "architecture": architecture,
              "dtype": list(signature.dtypes), "shape": [list(s) for s in signature.shapes]}
    def visit(value):
        if isinstance(value, dict):
            if "shape" in value and "dtype" in value:
                yield value
            else:
                for item in value.values():
                    yield from visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from visit(item)
    interface = signature.interface
    tensors = list(visit(interface.get("inputs", [])))
    for case in interface.get("cases", []):
        tensors.extend(visit(case.get("contract", {}).get("inputs", [])))
    if tensors:
        result["shape"] = [t["shape"] for t in tensors] if all(_shape(t["shape"]) for t in tensors) else []
        result["dtype"] = [t["dtype"] for t in tensors] if all(isinstance(t["dtype"], str) and t["dtype"] for t in tensors) else []
    layouts = []
    for tensor in tensors:
        shape, stride = tensor["shape"], tensor.get("stride")
        if stride is None and tensor.get("layout") in ("contiguous", "strided"):
            layouts.append(tensor["layout"])
            continue
        if (not _shape(shape) or not isinstance(stride, (list, tuple)) or len(shape) != len(stride)
                or any(not isinstance(step, int) or isinstance(step, bool) for step in stride)):
            layouts = []
            break
        expected, contiguous = 1, True
        for size, step in zip(reversed(shape), reversed(stride)):
            if size != 1 and step != expected:
                contiguous = False
            expected *= max(size, 1)
        layouts.append("contiguous" if contiguous else "strided")
    result["layout"] = layouts if tensors else ([layout] if layout else [])
    result["rank"] = [len(s) for s in result["shape"]]
    return result


def observed(contract, *, hardware="", environment=None, inputs=None):
    # A cited case overrides aggregate metadata; never borrow another case's
    # layout or shape to establish applicability.
    environment = environment or {}
    contract = contract.get("signature", contract)
    interface = inputs if isinstance(inputs, dict) else contract.get("interface", {})
    if isinstance(inputs, dict) and "shape" in inputs and "dtype" in inputs:
        interface = {"inputs": [inputs]}
    signature = OperatorSignature(
        "unknown", dtypes=tuple(contract.get("dtypes", [])) or
            ((contract["dtype"],) if contract.get("dtype") else ()),
        shapes=tuple(contract.get("shapes", [])),
        interface=interface,
    )
    result = context(signature, hardware=hardware,
                     layout=contract.get("layout", environment.get("layout", "")),
                     direction=contract.get("direction", "unknown"))
    result["architecture"] = environment.get("architecture") or environment.get("gpu_architecture", "")
    return result


def _canonical(key, value):
    if key == "dtype" and isinstance(value, str):
        return value.removeprefix("torch.")
    if key == "layout" and value == "noncontiguous":
        return "strided"
    return value


def check(conditions, facts):
    """Return compatible, conflict, or unknown; conflict takes precedence."""
    unknown = False
    for raw_key, expected in conditions.items():
        key = "dtype" if raw_key == "dtypes" else raw_key
        if key not in {"hardware", "architecture", "dtype", "layout", "direction", "rank", "shape"}:
            unknown = True
            continue
        actual = facts.get(key)
        if not actual or actual == "unknown":
            unknown = True
            continue
        values = actual if isinstance(actual, list) else [actual]
        if key == "shape":
            if not isinstance(expected, list):
                unknown = True
                continue
            for shape in values:
                if len(shape) != len(expected):
                    return "conflict"
                for size, bound in zip(shape, expected):
                    if bound is None:
                        continue
                    if isinstance(bound, int) and not isinstance(bound, bool):
                        if size != bound:
                            return "conflict"
                    elif isinstance(bound, dict) and bound and set(bound) <= {"min", "max"}:
                        if any(not isinstance(v, int) or isinstance(v, bool) for v in bound.values()):
                            unknown = True
                        elif size < bound.get("min", size) or size > bound.get("max", size):
                            return "conflict"
                    else:
                        unknown = True
        else:
            allowed = expected if isinstance(expected, list) else [expected]
            if not allowed or any(not isinstance(v, (str, int)) or isinstance(v, bool) for v in allowed):
                unknown = True
                continue
            allowed = [_canonical(key, v) for v in allowed]
            if any(_canonical(key, v) not in allowed for v in values):
                return "conflict"
    return "unknown" if unknown else "compatible"


def evidence_matches(facts, requested):
    """Count evidence in the current measured domain, not a union of devices.

    Shape transfer is governed by explicit conditions, not exact full-task
    matching. Known source hardware/dtype/layout cannot be transferred to an
    unknown or incompatible target; omitted facts never establish a match.
    """
    for key in ("hardware", "dtype", "layout"):
        source, target = facts.get(key), requested.get(key)
        if not source or not target:
            return False
        if check({key: source}, {key: target}) != "compatible":
            return False
    return True
