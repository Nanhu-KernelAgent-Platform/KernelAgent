"""Golden-parity tests for ``runtime_bindability_probe``.

Context: ``kernel_signature_bindability`` builds its bindable-name set from
Model source only, so it under-approximates the runtime state for buffers a
module constructor creates implicitly (``BatchNorm*d`` running statistics,
``register_buffer`` inside custom modules, ...). On 2026-11-14 that made a
correct BatchNorm3d-inference kernel for KernelBench L2 problem 77
unpassable: every repair round declared ``running_mean``/``running_var`` in
some form and the static gate rejected all of them at seed time, before any
verification could run.

The probe arbitrates a *statically rejected* seed by replaying the exact fill
decision of ``bind_kernel_call`` against a meta-device ``Model``. These tests
pin the probe to the real binder (golden parity) across signature styles and
problem shapes instead of pinning it to one problem's spelling, so any future
divergence between the two binding paths fails a test.
"""
import pytest
import torch

from triton_kernel_agent.kernel_call_binding import (
    KernelCallBindingError,
    bind_kernel_call,
    kernel_signature_bindability,
    runtime_bindability_probe,
)

# ---------------------------------------------------------------------------
# Problem fixtures (all instantiate on the meta device, no storage allocated)
# ---------------------------------------------------------------------------

BN3D_PROBLEM = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x * self.scale_factor
        x = self.batch_norm(x)
        return self.global_avg_pool(x)

def get_inputs():
    return [torch.rand(2, 4, 4, 4, 4)]

def get_init_inputs():
    return [4, 8, 5, 2.0]
"""

# Same shape of contract as BatchNorm3d but 1-D/2-D: the implicit-buffer
# defect is dimension-independent, so the fix must be too.
BN2D_PROBLEM = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.batch_norm(self.conv(x))

def get_inputs():
    return [torch.rand(2, 4, 8, 8)]

def get_init_inputs():
    return [4]
"""

BN1D_PROBLEM = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(channels)

    def forward(self, x):
        return self.batch_norm(x)

def get_inputs():
    return [torch.rand(2, 8, 16)]

def get_init_inputs():
    return [8]
"""

GROUPNORM_PROBLEM = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, channels, num_groups):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(num_groups, channels)

    def forward(self, x):
        return self.norm(self.conv(x))

def get_inputs():
    return [torch.rand(2, 4, 8, 8)]

def get_init_inputs():
    return [4, 2]
"""

LAYERNORM_PROBLEM = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape)

    def forward(self, x):
        return self.norm(x)

def get_inputs():
    return [torch.rand(2, 8)]

def get_init_inputs():
    return [(8,)]
"""

# No Model class: the probe must abstain instead of rejecting.
NO_MODEL_PROBLEM = """
import torch

def get_inputs():
    return [torch.rand(2, 4)]
"""


def _kernel(parameters: str) -> str:
    return f"def kernel_function({parameters}):\n    return x\n"


def _runtime_accepts(problem_source: str, kernel_source: str):
    """Golden truth: bind the signature with the real runtime binder."""
    namespace = {}
    with torch.device("meta"):
        exec(compile(problem_source, "<problem>", "exec"), namespace)
        model = namespace["Model"](*namespace["get_init_inputs"]())
        inputs = namespace["get_inputs"]()
    kernel_namespace = {}
    exec(kernel_source, kernel_namespace)
    try:
        bind_kernel_call(kernel_namespace["kernel_function"], model, inputs)
        return True, ""
    except KernelCallBindingError as exc:
        return False, str(exc)


def _probe_accepts(problem_source: str, kernel_source: str):
    reason = runtime_bindability_probe(
        kernel_source, problem_source, static_reason="static rejected"
    )
    return reason is None, reason


# ---------------------------------------------------------------------------
# Golden parity: probe verdict == real bind_kernel_call verdict
# ---------------------------------------------------------------------------

CASES = [
    # (label, problem, kernel parameters, expected accept)
    # -- problem 77 regression: signatures that the static gate killed ------
    # Qualified names plus the ``gamma``/``beta`` aliases: BN3D_PROBLEM has
    # three modules owning the leaf ``weight``, so the alias is ambiguous and
    # the binder refuses it (the probe replays the same rejection). See
    # test_runtime_probe_confirms_ambiguous_alias_rejection in
    # tests/test_kernel_call_binding.py.
    (
        "bn3d_qualified_plus_ambiguous_aliases_rejected",
        BN3D_PROBLEM,
        "x, conv_transpose_weight, conv_transpose_bias, gamma, beta, running_mean, running_var",
        False,
    ),
    (
        "bn3d_bare_weight_bias_and_running_stats",
        BN3D_PROBLEM,
        "x, weight, bias, running_mean, running_var",
        True,
    ),
    (
        "bn3d_num_batches_tracked",
        BN3D_PROBLEM,
        "x, running_mean, running_var, num_batches_tracked",
        True,
    ),
    # -- forms the runtime binder genuinely cannot fill ---------------------
    (
        "bn3d_terse_conv_bn_names",
        BN3D_PROBLEM,
        "x, conv_weight, conv_bias, bn_weight, bn_bias, bn_running_mean, bn_running_var",
        False,
    ),
    (
        "bn3d_module_prefixed_running_stats",
        BN3D_PROBLEM,
        "x, batch_norm_weight, batch_norm_bias, batch_norm_running_mean, batch_norm_running_var",
        False,
    ),
    # -- generality across batch-norm dimensions ---------------------------
    ("bn2d_running_stats", BN2D_PROBLEM, "x, weight, bias, running_mean, running_var", True),
    ("bn1d_running_stats", BN1D_PROBLEM, "x, weight, bias, running_mean, running_var", True),
    # -- modules without implicit buffers (static gate already handles) ----
    ("groupnorm_args", GROUPNORM_PROBLEM, "x, weight, bias, num_groups", True),
    ("layernorm_args", LAYERNORM_PROBLEM, "x, weight, bias, normalized_shape", True),
    ("bogus_names_rejected", GROUPNORM_PROBLEM, "x, foo, bar", False),
    # -- binder escape hatches: defaults / varargs / keyword-only ----------
    ("scale_factor_default", BN3D_PROBLEM, "x, scale_factor=2.0", True),
    ("kwonly_default", BN3D_PROBLEM, "x, *, scale_factor=2.0", True),
    ("varargs", BN3D_PROBLEM, "*args", True),
]


@pytest.mark.parametrize(
    "label,problem,parameters,expected",
    CASES,
    ids=[case[0] for case in CASES],
)
def test_probe_matches_runtime_binder(label, problem, parameters, expected):
    kernel_source = _kernel(parameters)
    runtime_ok, runtime_error = _runtime_accepts(problem, kernel_source)
    probe_ok, probe_reason = _probe_accepts(problem, kernel_source)
    # The probe is an arbitration layer: its verdict must equal the binder's.
    assert runtime_ok == probe_ok, (
        f"{label}: runtime accepted={runtime_ok}, probe accepted={probe_ok}; "
        f"runtime error={runtime_error!r}, probe reason={probe_reason!r}"
    )
    # And the golden truth itself must match the case table.
    assert runtime_ok == expected, (
        f"{label}: expected accept={expected}, runtime said {runtime_ok} "
        f"({runtime_error})"
    )


# ---------------------------------------------------------------------------
# Explicit problem-77 regression
# ---------------------------------------------------------------------------

# Qualified weight/bias names instead of the ambiguous ``gamma``/``beta``
# aliases, so the static rejection stays about the implicit BatchNorm buffers.
P77_KERNEL = _kernel(
    "x, conv_transpose_weight, conv_transpose_bias, running_mean, running_var"
)


def test_p77_static_gate_rejected_valid_signature_but_probe_accepts():
    """The exact signature style that died at seed time on 2026-11-14."""
    static_reason = kernel_signature_bindability(P77_KERNEL, BN3D_PROBLEM)
    assert static_reason is not None, "static gate must keep its old behavior..."
    assert "running_mean" in static_reason
    # ...but the arbitration probe overrules it because the binder can fill it.
    probe = runtime_bindability_probe(P77_KERNEL, BN3D_PROBLEM, static_reason=static_reason)
    assert probe is None
    runtime_ok, _ = _runtime_accepts(BN3D_PROBLEM, P77_KERNEL)
    assert runtime_ok


def test_probe_rejection_names_missing_parameters_and_available_state():
    kernel_source = _kernel("x, batch_norm_running_mean, batch_norm_running_var")
    reason = runtime_bindability_probe(kernel_source, BN3D_PROBLEM)
    assert reason is not None
    assert "batch_norm_running_mean" in reason
    assert "batch_norm_running_var" in reason
    # The available-state list lets the repair round converge in one attempt.
    assert "batch_norm.running_mean" in reason
    assert "running_mean" in reason


# ---------------------------------------------------------------------------
# Abstain, never reject on probe malfunction
# ---------------------------------------------------------------------------

def test_probe_abstains_when_problem_has_no_model():
    kernel_source = _kernel("x, whatever_a_kernel_wants")
    assert runtime_bindability_probe(kernel_source, NO_MODEL_PROBLEM) is None


def test_probe_abstains_when_kernel_is_unparseable():
    assert runtime_bindability_probe("def kernel_function(:", BN3D_PROBLEM) is None


def test_probe_abstains_when_problem_source_does_not_compile():
    assert runtime_bindability_probe(P77_KERNEL, "def broken(:") is None
