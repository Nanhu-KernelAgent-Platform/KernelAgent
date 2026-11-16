"""Explicit, operator-independent correctness and timing case contracts.

This file is also copied beside isolated verification/benchmark scripts. No
coverage is inferred from prose or fabricated for historical experiments.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from statistics import median

try:
    from .kernel_call_binding import bind_kernel_call
except ImportError:
    from kernel_call_binding import bind_kernel_call

RESULT_PREFIX = "KERNELAGENT_CASE_RESULTS="
SUITE_MARKER = "# KERNELAGENT_EXPLICIT_CASE_SUITE_V1"

INT32_LIMIT = 2 ** 31 - 1

# Verification-gate tolerance for the implicit ``default`` case. Without an
# override it inherits torch's dtype defaults (fp32: rtol=1.3e-6, atol=1e-5),
# which a long fp32 reduction cannot meet even when it is numerically correct: a
# summation order that differs from the eager reference alone costs ~12-14 ULP
# (KernelBench L1 #4, a 1M-element fp32 dot product, lands at 1.4-1.7e-6).
# KernelBench scores correctness at atol=rtol=1e-2
# (KernelBench/src/kernelbench/eval.py), so the bridge supplies that standard for
# KernelBench sources only and restores the previous values afterwards.
TOLERANCE_ENV = {"rtol": "KERNELAGENT_VERIFY_RTOL", "atol": "KERNELAGENT_VERIFY_ATOL"}
TOLERANCE_POLICY_ENV = "KERNELAGENT_VERIFY_TOLERANCE_POLICY"


def default_tolerance():
    """Tolerance override for cases that declare none; ``{}`` when unset."""
    overrides = {}
    for key, name in TOLERANCE_ENV.items():
        raw = os.environ.get(name)
        if raw is None or not str(raw).strip():
            continue
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be a number, got {raw!r}") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number, got {raw!r}")
        overrides[key] = value
    return overrides


def tolerance_policy(overrides):
    """Provenance label recorded in the contract next to the tolerance in force."""
    if not overrides:
        return "torch_dtype_default"
    return os.environ.get(TOLERANCE_POLICY_ENV) or "env_explicit"


def is_all_zero(value):
    """True when every tensor in ``value`` is entirely zero (int32-safe, chunked)."""
    import torch
    found = tensors(value)
    if not found:
        return False
    for tensor in found:
        if tensor.numel() == 0:
            continue
        if tensor.is_floating_point() or tensor.is_complex():
            if chunk_max(tensor) != 0.0:
                return False
        elif bool(tensor.any().item()):
            return False
    return True


def split_rows(tensor, count):
    """Split dim 0 of ``tensor`` into ``count`` near-equal row ranges."""
    rows = tensor.shape[0]
    step = max(1, math.ceil(rows / max(1, count)))
    return [slice(start, min(start + step, rows)) for start in range(0, rows, step)]


def chunk_max(tensor):
    """max(|x|) evaluated slice-wise along dim 0 (int32-safe); 0.0 for empty tensors."""
    import torch
    best = 0.0
    if tensor.numel():
        for sl in split_rows(tensor, max(1, math.ceil(tensor.numel() / INT32_LIMIT))):
            part = tensor[sl]
            if part.numel():
                best = max(best, float(part.abs().max().item()))
    return best


def reference_slices(model, inputs, budget=INT32_LIMIT):
    """Decide the dim-0 split for the reference and the comparison from the OUTPUT size.

    MUSA (muDNN) kernels index with int32: an output of >= 2**31 elements makes the
    eager reference return zeros silently and makes ``assert_close``'s mismatch-report
    mask fill raise ``FillOp MUDNN failed in: RunMask``. The output size is unknown
    before running the model, so probe one leading-dim row on a throwaway copy of the
    model (no state mutation, no autograd) and split so each slice's output stays under
    ``budget``. Small cases keep the single-slice path and are unaffected.
    """
    import copy
    import torch
    lead = inputs[0] if inputs and torch.is_tensor(inputs[0]) else None
    if lead is None or lead.dim() == 0 or lead.shape[0] <= 1 or lead.numel() <= 4096:
        return [slice(None)]
    probe_model = copy.deepcopy(model)
    was_training = probe_model.training
    probe_model.eval()

    def probe(rows):
        sliced = [t[0:rows] if torch.is_tensor(t) and t.dim() > 0 and t.shape[0] == lead.shape[0] else t
                  for t in inputs]
        with torch.no_grad():
            return probe_model(*sliced)

    try:
        probe_out = probe(1)
        # A second probe tells an output that grows with inputs[0] rows apart from
        # one whose row count is fixed by another input. moe_unpermute_mask
        # returns row_id_map.shape[1] rows regardless of the activation rows, so
        # the single-probe estimate (fixed-size output x 32768 leading rows)
        # split the reference into 512 sinks and the concat collapsed to
        # all-zero, failing every round as "reference-unsupported".
        scales = True
        if lead.shape[0] > 1:
            try:
                probe_out2 = probe(min(2, lead.shape[0]))
                scales = (sum(t.shape[0] for t in tensors(probe_out2))
                          != sum(t.shape[0] for t in tensors(probe_out)))
            except Exception:
                scales = True
    except Exception:
        return [slice(None)]
    finally:
        probe_model.train(was_training)
    per_sample = sum(t.numel() for t in tensors(probe_out))
    if not per_sample:
        return [slice(None)]
    total = per_sample * lead.shape[0] if scales else per_sample
    if total <= budget:
        return [slice(None)]
    return split_rows(lead, max(2, math.ceil(total / budget)))


def slice_inputs(inputs, sl):
    """Slice every input sharing the leading dim of ``inputs[0]``; pass others through."""
    import torch
    if sl == slice(None) or not inputs:
        return list(inputs)
    lead_rows = inputs[0].shape[0] if torch.is_tensor(inputs[0]) and inputs[0].dim() > 0 else None
    return [t[sl] if torch.is_tensor(t) and t.dim() > 0 and t.shape[0] == lead_rows else t for t in inputs]


def concat_outputs(parts):
    import torch
    first = parts[0]
    if torch.is_tensor(first):
        return torch.cat(parts, dim=0)
    if isinstance(first, (tuple, list)):
        return type(first)(torch.cat([part[index] for part in parts], dim=0) for index in range(len(first)))
    raise ValueError(f"unsupported reference output type for chunked evaluation: {type(first).__name__}")


def assert_close_chunked(actual, expected, slices, tolerance):
    import torch
    if len(slices) == 1:
        torch.testing.assert_close(actual, expected, equal_nan=False, **tolerance)
        return
    actual_seq = actual if isinstance(actual, (tuple, list)) else (actual,)
    expected_seq = expected if isinstance(expected, (tuple, list)) else (expected,)
    for actual_tensor, expected_tensor in zip(actual_seq, expected_seq):
        for sl in slices:
            torch.testing.assert_close(actual_tensor[sl], expected_tensor[sl], equal_nan=False, **tolerance)


def register_torch_musa_alias():
    """Make a bare ``import torch.musa`` valid on MUSA hosts.

    ``torch_musa`` exposes the device API through the ``torch.musa`` *attribute*,
    but it is not a real ``torch`` submodule, so generated tests that write
    ``import torch.musa`` or ``from torch.musa import ...`` fail with
    ``ModuleNotFoundError``. Importing the backend and registering it under
    ``sys.modules['torch.musa']`` (plus the attribute) makes every form work.
    No-op when torch_musa is unavailable (e.g. non-MUSA hosts).
    """
    try:
        import torch_musa
    except ImportError:
        return
    import sys
    import torch
    if not hasattr(torch, "musa"):
        torch.musa = torch_musa
    sys.modules.setdefault("torch.musa", torch_musa)


def valid_samples(values, count=None):
    """Positive finite numeric windows; count stays explicit at each stage."""
    return bool(values) and (count is None or len(values) == count) and all(
        not isinstance(value, bool) and isinstance(value, (int, float))
        and math.isfinite(value) and value > 0 for value in values)


def run_guarded_test(path, device="cpu"):
    """Detect changed framework callables even when a test exits successfully.

    A bounded integrity check, not an isolation/security sandbox.
    """
    import runpy
    import sys
    import torch
    if device == "musa":
        register_torch_musa_alias()
    owners = (torch, torch.Tensor, torch.testing, torch.autograd, torch.nn.functional)
    baseline = [(owner, name, value) for owner in owners
                for name, value in vars(owner).items() if callable(value)]
    arguments = sys.argv
    try:
        sys.argv = [str(path)]
        runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.argv = arguments
        changed = [name for owner, name, value in baseline if getattr(owner, name, None) is not value]
        if changed:
            raise RuntimeError("Kernel/test modified framework verification behavior: " + ", ".join(changed))


def describe(value):
    import torch
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype),
                "stride": list(value.stride()), "requires_grad": value.requires_grad}
    if isinstance(value, dict):
        return {str(key): describe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [describe(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return {"type": type(value).__name__, "value": value}
    raise ValueError(f"unsupported case value: {type(value).__name__}")


def load_cases(problem):
    """Default input plus explicitly supplied cases, with bounded unique IDs."""
    default = {"case_key": "default", "inputs": problem.get_inputs(),
               "init_inputs": problem.get_init_inputs() if hasattr(problem, "get_init_inputs") else []}
    cases = [default]
    if hasattr(problem, "get_test_cases"):
        extra = problem.get_test_cases()
        if not isinstance(extra, (list, tuple)) or len(extra) > 15:
            raise ValueError("get_test_cases must return at most 15 explicit case dictionaries")
        cases.extend(extra)
    overrides = default_tolerance()
    policy = tolerance_policy(overrides)
    seen = set()
    normalized = []
    for raw in cases:
        if not isinstance(raw, dict):
            raise ValueError("each test case must be a dictionary")
        key = raw.get("case_key")
        if not isinstance(key, str) or not key.strip() or key in seen:
            raise ValueError("case_key must be unique and nonempty; 'default' is reserved")
        seen.add(key)
        case = dict(raw)
        for name in ("inputs", "init_inputs"):
            values = case.get(name, default[name])
            if name == "inputs" and name not in case:
                raise ValueError("each explicit case requires inputs")
            case[name] = list(values) if isinstance(values, (tuple, list)) else [values]
        # A case that declares no tolerance inherits the caller-supplied standard;
        # a case that declares its own keeps it and reports that provenance.
        declared = any(name in case for name in ("rtol", "atol"))
        for name, value in overrides.items():
            case.setdefault(name, value)
        case["tolerance_policy"] = "explicit_case" if declared else policy
        for name in ("rtol", "atol"):
            if name in case and (isinstance(case[name], bool) or not isinstance(case[name], (int, float))
                                 or not math.isfinite(case[name]) or case[name] < 0):
                raise ValueError(f"{name} must be a finite nonnegative number")
        normalized.append(case)
    return normalized


def move(value, device, *, gradients=False):
    """Preserve explicit strides, including sliced/noncontiguous test inputs."""
    import torch
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise ValueError("explicit case suite currently requires strided tensors")
        moved = torch.empty_strided(value.shape, value.stride(), dtype=value.dtype, device=device)
        with torch.no_grad():
            moved.copy_(value)
        moved.requires_grad_(value.requires_grad or (gradients and (value.is_floating_point() or value.is_complex())))
        return moved
    if isinstance(value, dict):
        return {key: move(item, device, gradients=gradients) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move(item, device, gradients=gradients) for item in value)
    return value


def tensors(value):
    import torch
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in tensors(item)]
    if isinstance(value, (list, tuple)):
        return [tensor for item in value for tensor in tensors(item)]
    return []


def named_tensors(value, name):
    import torch
    if isinstance(value, torch.Tensor):
        return [(name, value)]
    if isinstance(value, dict):
        return [pair for key, item in value.items() for pair in named_tensors(item, f"{name}/{key}")]
    if isinstance(value, (tuple, list)):
        return [pair for index, item in enumerate(value) for pair in named_tensors(item, f"{name}/{index}")]
    return []


def gradient_call(output, inputs, model, case):
    import torch
    outputs = tuple(tensor for tensor in tensors(output) if tensor.requires_grad)
    targets = [(name, tensor) for i, value in enumerate(inputs)
               for name, tensor in named_tensors(value, f"input:{i}") if tensor.requires_grad]
    targets += [(f"parameter:{name}", tensor) for name, tensor in model.named_parameters()
                if tensor.requires_grad]
    names = case.get("gradient_targets")
    if names is not None:
        if not isinstance(names, (list, tuple)) or len(names) != len(set(names)):
            raise ValueError("gradient_targets must contain unique input:N / parameter:NAME names")
        lookup = dict(targets)
        if any(name not in lookup for name in names):
            raise ValueError("unknown or non-differentiable gradient target")
        targets = [(name, lookup[name]) for name in names]
    if not outputs or not targets:
        raise ValueError("backward case requires differentiable outputs and requested inputs/parameters")
    supplied = case.get("grad_outputs")
    if supplied is None:
        # Nonuniform upstream gradients exercise indexing and reduction paths.
        upstream = tuple(torch.linspace(0.25, 1.25, tensor.numel(), device=tensor.device,
                                       dtype=tensor.dtype).reshape(tensor.shape) for tensor in outputs)
    else:
        upstream = tuple(move(supplied, outputs[0].device)) if isinstance(supplied, (list, tuple)) else (move(supplied, outputs[0].device),)
        if len(upstream) != len(outputs) or any(a.shape != b.shape or a.dtype != b.dtype for a, b in zip(upstream, outputs)):
            raise ValueError("grad_outputs must match every differentiable output shape and dtype")
    def invoke():
        return torch.autograd.grad(outputs, tuple(tensor for _, tensor in targets),
                                   grad_outputs=upstream, retain_graph=True, allow_unused=True)
    return invoke, {"upstream_gradients": describe(upstream),
                    "requested_gradients": [{"name": name, "tensor": describe(tensor)} for name, tensor in targets]}


def prepare(problem, case, device, direction):
    import torch
    if str(device).split(":")[0] == "musa":
        register_torch_musa_alias()
    torch.manual_seed(0)
    model = problem.Model(*case["init_inputs"]).to(device)
    inputs = move(case["inputs"], device, gradients=direction in {"backward", "joint"})
    return model, inputs


def contract(model, inputs, output, *, direction, backward=None, saved_state=None):
    payload = {"direction": direction, "inputs": describe(inputs), "outputs": describe(output),
               "parameters": {name: describe(tensor) for name, tensor in model.named_parameters()},
               "buffers": {name: describe(tensor) for name, tensor in model.named_buffers()},
               "training": model.training}
    if backward is not None:
        payload["backward"] = {**backward, "saved_state": saved_state or [],
                               "saved_state_source": "reference_autograd_hooks"}
    payload["digest"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
    return payload


def verify_cases(problem, kernel, device, direction="forward"):
    import torch
    results = []
    for case in load_cases(problem):
        result = {"case_key": case["case_key"], "status": "failed", "contract": {}, "error": ""}
        try:
            model, inputs = prepare(problem, case, device, direction)
            candidate_model = copy.deepcopy(model)
            candidate_inputs = move(inputs, device)
            saved = []
            def pack(tensor):
                saved.append(describe(tensor))
                return tensor
            slices = reference_slices(model, inputs) if direction != "backward" else [slice(None)]
            result["reference_chunked"] = len(slices) > 1
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                if len(slices) == 1:
                    expected = model(*inputs)
                else:
                    expected = concat_outputs([model(*slice_inputs(inputs, sl)) for sl in slices])
            # Reference sanity: an all-zero reference with non-zero inputs has two
            # very different meanings.
            #   (a) The graph is degenerate by construction: KernelBench L2 #80
            #       reduces a (1024, 8192) GEMM over dim=1 to (1024, 1) and then
            #       subtracts that row's own mean, which is exactly 0.0 in IEEE-754
            #       for every row -- the true output IS all zero. It must be
            #       compared like any other case: only status "passed" can satisfy
            #       the suite, so skipping it fails every round unconditionally.
            #   (b) The reference was evaluated at full shape beyond the int32
            #       element index limit and silently returned zeros. That reference
            #       is not authoritative, so no kernel can be judged against it.
            expected_numel = sum(t.numel() for t in tensors(expected))
            reference_all_zero = bool(expected_numel) and all(
                chunk_max(t) == 0.0 for t in tensors(expected)) and any(
                chunk_max(t) > 0.0 for t in tensors(inputs) if t.numel())
            if reference_all_zero and len(slices) == 1 and expected_numel > INT32_LIMIT:
                result["status"] = "reference-unsupported"
                result["unsupported"] = True
                result["error"] = (
                    f"reference-unsupported-shape: reference output is all-zero at numel={expected_numel}, "
                    f"which exceeds the int32 element index limit (2**31-1={INT32_LIMIT}) while the inputs "
                    "are non-zero; the full-shape eager reference overflows its int32 offsets and returns "
                    "zeros at this size, so it is not authoritative. This is not a kernel defect; do not "
                    "rewrite the kernel.")
                result["contract"] = contract(model, inputs, expected, direction=direction)
                results.append(result)
                continue
            result["reference_all_zero"] = reference_all_zero
            actual = bind_kernel_call(kernel, candidate_model, candidate_inputs)()
            # A collapsed (all-zero) candidate must never be excused by a loose
            # tolerance: atol=1e-2 would otherwise accept a kernel that returns
            # zeros for every element.
            if is_all_zero(actual) and not is_all_zero(expected):
                raise AssertionError(
                    "candidate output is all-zero while the reference is not; a collapsed output cannot be "
                    f"excused by the verification tolerance (policy: {case.get('tolerance_policy')})")
            tolerance = ({"rtol": case.get("rtol", 1e-4), "atol": case.get("atol", 1e-5)}
                         if "rtol" in case or "atol" in case else {})
            if reference_all_zero:
                # x - x.mean(dim=1) over a size-1 reduced dim is exact (a - a == 0.0),
                # so a zero reference must be matched as zero: the benchmark's 1e-2
                # tolerance would otherwise excuse a collapsed kernel.
                tolerance = {"rtol": 0.0, "atol": min(float(tolerance.get("atol", 1e-5)), 1e-6)}
            assert_close_chunked(actual, expected, slices, tolerance)
            backward = None
            if direction in {"backward", "joint"}:
                ref_grad, backward = gradient_call(expected, inputs, model, case)
                kernel_grad, actual_backward = gradient_call(actual, candidate_inputs, candidate_model, case)
                expected_grads, actual_grads = ref_grad(), kernel_grad()
                for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                    grad_slices = (split_rows(expected_grad, max(1, math.ceil(expected_grad.numel() / INT32_LIMIT)))
                                   if expected_grad.numel() > INT32_LIMIT else [slice(None)])
                    assert_close_chunked(actual_grad, expected_grad, grad_slices, tolerance)
                backward["gradient_outputs"] = describe(expected_grads)
                if actual_backward["requested_gradients"] != backward["requested_gradients"]:
                    raise ValueError("candidate gradient target contract differs")
            result["contract"] = contract(model, inputs, expected, direction=direction,
                                          backward=backward, saved_state=saved)
            result["contract"]["verification_tolerance"] = (
                {**tolerance, "policy": case.get("tolerance_policy") or "torch_dtype_default"}
                if tolerance else {"policy": "torch_dtype_default"})
            result["status"] = "passed"
        except Exception as exc:
            root, chain = exc, []
            while getattr(root, "__cause__", None) is not None:
                root = root.__cause__
                chain.append(f"{type(root).__name__}: {root}")
            result["error"] = f"{type(exc).__name__}: {exc}" + (f"\n[cause] " + " <- ".join(chain) if chain else "")
            if not result["contract"]:
                result["contract"] = {"direction": direction, "inputs": describe(case["inputs"])}
        results.append(result)
    return results

def benchmark_cases(problem, kernel, device, direction, measure):
    """One warmup/three windows per case; equal-weight suite mean is explicit."""
    import torch
    rows = []
    for case in load_cases(problem):
        model, inputs = prepare(problem, case, device, direction)
        invoke = bind_kernel_call(kernel, model, inputs) if kernel else lambda: model(*inputs)
        ref_model, ref_inputs = copy.deepcopy(model), move(inputs, device)
        saved = []
        def pack(tensor):
            saved.append(describe(tensor))
            return tensor
        # Runtime interface is always derived from the reference, not candidate implementation.
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            output = ref_model(*ref_inputs)
        backward = None
        if direction == "backward":
            ref_grad, backward = gradient_call(output, ref_inputs, ref_model, case)
            backward["gradient_outputs"] = describe(ref_grad())
            invoke, _ = gradient_call(invoke(), inputs, model, case)
        samples = list(measure(invoke))
        if not valid_samples(samples, 3):
            raise ValueError("each case needs three positive finite samples")
        rows.append({"case_key": case["case_key"], "time_ms": median(samples),
                     "samples_ms": samples, "inputs": contract(ref_model, ref_inputs, output,
                     direction=direction, backward=backward, saved_state=saved)})
    aggregate = [sum(row["samples_ms"][i] for row in rows) / len(rows) for i in range(3)]
    return {"time_ms": median(aggregate), "samples_ms": aggregate,
            "case_measurements": rows, "aggregation": "equal_weight_mean_windows_v1"}


def paired_cases(parent, candidate):
    lookup = {row["case_key"]: row for row in parent.get("case_measurements", [])}
    rows = []
    for row in candidate.get("case_measurements", []):
        before = lookup.get(row["case_key"])
        paired = before is not None and before.get("inputs") == row.get("inputs")
        rows.append({"case_key": row["case_key"], "inputs": row.get("inputs", {}),
                     "kernel_ms": row["time_ms"], "samples_ms": row["samples_ms"],
                     "baseline_ms": before["time_ms"] if paired else None,
                     "baseline_samples_ms": before["samples_ms"] if paired else [],
                     "benchmark": {"paired": paired, "source": "explicit_case_windows_v1",
                         "comparison": candidate.get("comparison", {}).get("cases", {}).get(row["case_key"], {}),
                         "measurement_schedule": candidate.get("measurement_schedule", [])}})
    return rows


def same_contract(checked, measured):
    ignored = {"digest", "verification_tolerance"}
    return ({key: value for key, value in checked.items() if key not in ignored} ==
            {key: value for key, value in measured.items() if key not in ignored})


def measurement_is_covered(measurements, results):
    """Missing/different coverage must not acquire a verified timing label."""
    if not measurements:
        return not results  # Legacy single aggregate, not explicit coverage.
    checked = {row["case_key"]: row for row in results}
    measured_keys = [row["case_key"] for row in measurements]
    return (len(checked) == len(measurements) and set(measured_keys) == set(checked) and all(
        row["case_key"] in checked and checked[row["case_key"]]["status"] == "passed"
        and same_contract(checked[row["case_key"]].get("contract", {}), row.get("inputs", {}))
        for row in measurements))


def reference_contracts(problem, direction):
    import torch
    rows = []
    for case in load_cases(problem):
        model, inputs = prepare(problem, case, "cpu", direction)
        saved = []
        def pack(tensor):
            saved.append(describe(tensor))
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            output = model(*inputs)
        backward = None
        if direction in {"backward", "joint"}:
            invoke, backward = gradient_call(output, inputs, model, case)
            backward["gradient_outputs"] = describe(invoke())
        rows.append({"case_key": case["case_key"], "contract": contract(model, inputs, output,
                     direction=direction, backward=backward, saved_state=saved)})
    return rows


def suite_test(source, direction="forward", device="cuda"):
    """Generate a deterministic companion test; LLM-authored tests remain intact."""
    import ast
    tree = ast.parse(source)
    definitions = {node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    if not {"Model", "get_inputs"} <= definitions:
        return None
    if device != "musa" and direction not in {"backward", "joint"} and not any(
        isinstance(node, ast.FunctionDef) and node.name == "get_test_cases" for node in tree.body
    ):
        return None
    return (SUITE_MARKER + "\nimport json, types\nfrom kernel_case_suite import verify_cases, RESULT_PREFIX\n"
            "from kernel import kernel_function\nproblem = types.ModuleType('case_reference')\n"
            f"exec(compile({source!r}, '<case_reference>', 'exec'), problem.__dict__)\n"
            f"results = verify_cases(problem, kernel_function, {device!r}, {direction!r})\n"
            "print(RESULT_PREFIX + json.dumps(results, allow_nan=False))\n"
            "assert results and all(row['status'] == 'passed' for row in results), results\n")
