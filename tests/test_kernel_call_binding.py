import pytest
from triton_kernel_agent.kernel_call_binding import (
    KernelCallBindingError,
    bind_kernel_call,
    collect_model_values,
)


class FakeModel:
    stride = 2

    def __init__(self):
        self.weight = object()
        self.running_mean = object()

    def named_parameters(self):
        return iter((("weight", self.weight),))

    def named_buffers(self):
        return iter((("running_mean", self.running_mean),))

    def named_modules(self):
        return iter((("", self),))


def test_binds_top_level_parameter_buffer_and_module_attribute_by_name():
    model = FakeModel()
    x = object()

    def kernel_function(x, weight, running_mean, stride):
        return x, weight, running_mean, stride

    invoke = bind_kernel_call(kernel_function, model, [x])
    assert invoke() == (x, model.weight, model.running_mean, 2)


def test_binds_nested_parameter_leaf_and_alias():
    weight = object()

    class Nested(FakeModel):
        def named_parameters(self):
            return iter((("linear.weight", weight),))

        def named_buffers(self):
            return iter(())

    values, ordered, owners = collect_model_values(Nested())
    assert values["linear.weight"] is weight
    assert values["weight"] is weight
    assert values["w"] is weight
    assert ordered == [weight]
    assert owners["weight"] == ["linear.weight"]


def test_never_guesses_missing_kernel_arguments_from_constructor_inputs():
    def kernel_function(x, dimensions):
        return x, dimensions

    with pytest.raises(KernelCallBindingError, match="dimensions"):
        bind_kernel_call(kernel_function, FakeModel(), [object()])


def test_varargs_receive_forward_inputs_then_parameters():
    model = FakeModel()
    x = object()

    def kernel_function(*args):
        return args

    assert bind_kernel_call(kernel_function, model, [x])() == (x, model.weight)


def test_binds_composite_kernel_state_to_named_submodules():
    conv_weight, conv_bias, fc_weight, fc_bias = (object() for _ in range(4))

    class Composite(FakeModel):
        def named_parameters(self):
            return iter((
                ("conv.weight", conv_weight), ("conv.bias", conv_bias),
                ("fc1.weight", fc_weight), ("fc1.bias", fc_bias),
            ))

        def named_buffers(self):
            return iter(())

    x = object()

    def kernel_function(x, conv_weight, conv_bias, fc_weight, fc_bias):
        return x, conv_weight, conv_bias, fc_weight, fc_bias

    assert bind_kernel_call(kernel_function, Composite(), [x])() == (
        x, conv_weight, conv_bias, fc_weight, fc_bias,
    )


def test_missing_parameter_error_lists_available_model_values():
    """Fix 3-1: refine feedback depends on the stable bindability error text."""
    def kernel_function(x, dim):
        return x, dim

    with pytest.raises(KernelCallBindingError) as excinfo:
        bind_kernel_call(kernel_function, FakeModel(), [object()])
    message = str(excinfo.value)
    assert "Cannot bind required kernel parameter(s): dim" in message
    assert "available model values:" in message
    assert "weight" in message
    assert "stride" in message


def test_numbered_submodule_alias_must_be_unambiguous():
    class Composite(FakeModel):
        def named_parameters(self):
            return iter((("fc1.weight", object()), ("fc2.weight", object())))

        def named_buffers(self):
            return iter(())

    def kernel_function(x, fc_weight):
        return x, fc_weight

    with pytest.raises(KernelCallBindingError, match="Ambiguous model state"):
        bind_kernel_call(kernel_function, Composite(), [object()])


class TwoModuleModel(FakeModel):
    """A fused reference: Linear + GroupNorm, i.e. two ``.weight`` leaves."""

    def __init__(self):
        self.gemm_weight = object()
        self.gemm_bias = object()
        self.gn_weight = object()
        self.gn_bias = object()

    def named_parameters(self):
        return iter((
            ("gemm.weight", self.gemm_weight), ("gemm.bias", self.gemm_bias),
            ("group_norm.weight", self.gn_weight), ("group_norm.bias", self.gn_bias),
        ))

    def named_buffers(self):
        return iter(())


def test_ambiguous_alias_is_rejected_instead_of_binding_first_match():
    """``gamma`` must not silently bind the Linear weight (8192x8192)."""

    def kernel_function(x, weight, bias, gamma, beta):
        return x, weight, bias, gamma, beta

    with pytest.raises(KernelCallBindingError) as excinfo:
        bind_kernel_call(kernel_function, TwoModuleModel(), [object()])
    message = str(excinfo.value)
    assert "gamma" in message
    assert "gemm.weight" in message and "group_norm.weight" in message
    assert "group_norm_weight" in message


def test_ambiguous_alias_does_not_fall_through_to_positional_input():
    def kernel_function(x, extra, gamma):
        return x, extra, gamma

    with pytest.raises(KernelCallBindingError):
        bind_kernel_call(kernel_function, TwoModuleModel(), [object(), object()])


def test_qualified_names_bind_on_two_module_model():
    model = TwoModuleModel()
    x = object()

    def kernel_function(x, gemm_weight, gemm_bias, group_norm_weight, group_norm_bias):
        return x, gemm_weight, gemm_bias, group_norm_weight, group_norm_bias

    assert bind_kernel_call(kernel_function, model, [x])() == (
        x, model.gemm_weight, model.gemm_bias, model.gn_weight, model.gn_bias,
    )


def test_single_module_alias_still_binds():
    """A lone ``weight``/``bias`` leaf keeps the alias behaviour (regression)."""
    weight, bias = object(), object()

    class Single(FakeModel):
        def named_parameters(self):
            return iter((("linear.weight", weight), ("linear.bias", bias)))

        def named_buffers(self):
            return iter(())

    x = object()

    def kernel_function(x, weight, bias, gamma, beta):
        return x, weight, bias, gamma, beta

    assert bind_kernel_call(kernel_function, Single(), [x])() == (
        x, weight, bias, weight, bias,
    )


# ---------------------------------------------------------------------------
# Static seed gate + runtime probe (arbitration)
# ---------------------------------------------------------------------------

PROBLEM_TWO_MODULES = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_features, out_features, num_groups):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)

    def forward(self, x):
        return self.group_norm(self.gemm(x))

def get_inputs():
    return [torch.rand(8, 16)]

def get_init_inputs():
    return [16, 32, 2]
"""

PROBLEM_BN = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        return self.bn(x)

def get_inputs():
    return [torch.rand(8, 32)]

def get_init_inputs():
    return [32, 32]
"""


def test_static_gate_rejects_ambiguous_alias_signature():
    from triton_kernel_agent.kernel_call_binding import kernel_signature_bindability

    kernel = "def kernel_function(x, weight, bias, gamma, beta):\n    return x\n"
    reason = kernel_signature_bindability(kernel, PROBLEM_TWO_MODULES)
    assert reason is not None
    assert "gamma" in reason and "beta" in reason
    assert "group_norm_weight" in reason


def test_static_gate_accepts_qualified_signature():
    from triton_kernel_agent.kernel_call_binding import kernel_signature_bindability

    kernel = (
        "def kernel_function(x, gemm_weight, gemm_bias, "
        "group_norm_weight, group_norm_bias):\n    return x\n"
    )
    assert kernel_signature_bindability(kernel, PROBLEM_TWO_MODULES) is None


def test_runtime_probe_confirms_ambiguous_alias_rejection():
    from triton_kernel_agent.kernel_call_binding import runtime_bindability_probe

    kernel = "def kernel_function(x, weight, bias, gamma, beta):\n    return x\n"
    reason = runtime_bindability_probe(kernel, PROBLEM_TWO_MODULES, static_reason="x")
    assert reason is not None
    assert "gamma" in reason
    assert "group_norm_weight" in reason


def test_runtime_probe_overrules_static_false_negative():
    """Static names miss implicit BatchNorm buffers; the probe accepts the seed."""
    from triton_kernel_agent.kernel_call_binding import (
        kernel_signature_bindability,
        runtime_bindability_probe,
    )

    kernel = (
        "def kernel_function(x, weight, bias, running_mean, running_var):\n"
        "    return x\n"
    )
    static_reason = kernel_signature_bindability(kernel, PROBLEM_BN)
    assert static_reason is not None
    assert runtime_bindability_probe(kernel, PROBLEM_BN, static_reason=static_reason) is None


# ---------------------------------------------------------------------------
# Full-path underscore resolution for multi-word leaves
# ---------------------------------------------------------------------------

class ConvTransposeBNModel(FakeModel):
    """Reference shape with two weight owners and implicit BN buffers."""

    def __init__(self):
        self.conv_w, self.conv_b = object(), object()
        self.bn_w, self.bn_b = object(), object()
        self.rm, self.rv, self.nbt = object(), object(), object()

    def named_parameters(self):
        return iter((
            ("conv_transpose.weight", self.conv_w),
            ("conv_transpose.bias", self.conv_b),
            ("batch_norm.weight", self.bn_w),
            ("batch_norm.bias", self.bn_b),
        ))

    def named_buffers(self):
        return iter((
            ("batch_norm.running_mean", self.rm),
            ("batch_norm.running_var", self.rv),
            ("batch_norm.num_batches_tracked", self.nbt),
        ))


def test_module_qualified_buffer_names_bind_after_full_path_match():
    """A fully module-qualified buffer signature must bind."""
    model = ConvTransposeBNModel()
    x = object()

    def kernel_function(x, weight, bias, batch_norm_weight, batch_norm_bias,
                        batch_norm_running_mean, batch_norm_running_var):
        return (x, weight, bias, batch_norm_weight, batch_norm_bias,
                batch_norm_running_mean, batch_norm_running_var)

    assert bind_kernel_call(kernel_function, model, [x])() == (
        x, model.conv_w, model.conv_b, model.bn_w, model.bn_b,
        model.rm, model.rv,
    )


def test_bare_buffer_leaf_names_still_bind():
    """Regression guard: bare running_mean/running_var keep binding."""
    model = ConvTransposeBNModel()
    x = object()

    def kernel_function(x, running_mean, running_var):
        return x, running_mean, running_var

    assert bind_kernel_call(kernel_function, model, [x])() == (
        x, model.rm, model.rv,
    )


def test_abbreviated_module_prefix_still_rejected():
    """The old template taught `bn_weight`; the binder must keep refusing it."""
    def kernel_function(x, bn_weight):
        return x, bn_weight

    with pytest.raises(KernelCallBindingError, match="bn_weight"):
        bind_kernel_call(kernel_function, ConvTransposeBNModel(), [object()])


def test_ambiguous_gamma_still_rejected_and_message_teaches_buffers():
    """Ambiguous ``gamma`` stays rejected and the message teaches buffer spellings."""
    def kernel_function(x, gamma):
        return x, gamma

    with pytest.raises(KernelCallBindingError) as excinfo:
        bind_kernel_call(kernel_function, ConvTransposeBNModel(), [object()])
    message = str(excinfo.value)
    assert "gamma" in message
    assert "conv_transpose_weight" in message and "batch_norm_weight" in message
    assert "batch_norm_running_mean" in message  # Part B appended guidance


def test_full_path_underscore_collision_is_refused():
    class Colliding(FakeModel):
        def named_parameters(self):
            return iter(())

        def named_buffers(self):
            return iter((
                ("batch_norm.running_mean", object()),
                ("batch.norm_running_mean", object()),
            ))

    def kernel_function(x, batch_norm_running_mean):
        return x, batch_norm_running_mean

    with pytest.raises(KernelCallBindingError, match="Ambiguous model state"):
        bind_kernel_call(kernel_function, Colliding(), [object()])


PROBLEM_CONV_BN = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return self.batch_norm(self.conv_transpose(x))

def get_inputs():
    return [torch.rand(1, 3, 8, 8, 8)]

def get_init_inputs():
    return [3, 4, 3, 2, 1]
"""


def test_probe_accepts_p72_repair3_signature():
    """End-to-end arbitration: the historically fatal signature now passes the probe."""
    from triton_kernel_agent.kernel_call_binding import (
        kernel_signature_bindability,
        runtime_bindability_probe,
    )

    kernel = (
        "def kernel_function(x, weight, bias, batch_norm_weight, batch_norm_bias,\n"
        "                    batch_norm_running_mean, batch_norm_running_var):\n"
        "    return x\n"
    )
    static_reason = kernel_signature_bindability(kernel, PROBLEM_CONV_BN)
    assert static_reason is not None  # static names cannot see implicit BN buffers
    assert runtime_bindability_probe(kernel, PROBLEM_CONV_BN, static_reason=static_reason) is None


def test_probe_message_offers_signature_ready_names():
    """Part B: the rejection text must contain copy-pasteable parameter names."""
    from triton_kernel_agent.kernel_call_binding import runtime_bindability_probe

    kernel = (
        "def kernel_function(x, bn_weight, bn_bias, bn_running_mean):\n"
        "    return x\n"
    )
    reason = runtime_bindability_probe(kernel, PROBLEM_CONV_BN, static_reason="s")
    assert reason is not None
    assert "write as batch_norm_running_mean" in reason
    assert "write as batch_norm_weight" in reason


def test_guidelines_no_longer_teach_abbreviated_prefixes():
    """Part C guard: the template must not re-introduce `bn_weight`."""
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "triton_kernel_agent" / "templates" / "musa_guidelines.j2"
    )
    text = path.read_text(encoding="utf-8")
    # The old advice recommended abbreviated prefixes; only the explicit
    # rejection warning may mention `bn_weight` now.
    assert "name every other module `<module>_<param>`" not in text
    assert "`bn_weight`, `group_norm_weight`, `bn_bias`" not in text
    assert "batch_norm_running_mean" in text
    assert text.count("`bn_weight`") == 1  # the "do NOT bind" warning only
