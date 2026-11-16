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
    """L2 #30: ``gamma`` must not silently bind the Linear weight (8192x8192)."""

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


