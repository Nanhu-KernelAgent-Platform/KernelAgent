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

    values, ordered = collect_model_values(Nested())
    assert values["linear.weight"] is weight
    assert values["weight"] is weight
    assert values["w"] is weight
    assert ordered == [weight]


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
