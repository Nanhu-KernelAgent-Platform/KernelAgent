"""Seed-time static check: kernel_function parameters must be bindable.

Follow-up to fix 3-1 (docs/design_kernelagent_fix_3-1_signature_contract_binding.md).
After the call convention was unified on ``bind_kernel_call``, the *kernel* is
the side that declares the signature, so a parameter that no source can fill
only fails at runtime -- one whole verification round (25-30 min) later. P57
round 6 declared ``stride_h``/``padding_w``/``batch_size``/... on top of the
single tensor input and died with
``Cannot bind required kernel parameter(s): ...``.

``kernel_signature_bindability`` reproduces the binding decision statically, and
is deliberately fail-open: anything it cannot read unambiguously is accepted.
These tests pin both directions plus the fail-open cases.
"""

import pytest

from triton_kernel_agent.agent import TritonKernelAgent
from triton_kernel_agent.kernel_backend import get_kernel_backend
from triton_kernel_agent.kernel_call_binding import kernel_signature_bindability


# P57: conv_transpose2d square -- one tensor input, module-level configuration.
CONV_TRANSPOSE_PROBLEM = '''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, output_padding: int = 0,
                 groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, output_padding=output_padding, groups=groups,
            bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_transpose2d(x)

batch_size = 8
in_channels = 64
out_channels = 64
kernel_size = 3
height = 1024
width = 1024

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
'''

# Max/argmax reduction: the reduction dimension is get_init_inputs state.
REDUCE_PROBLEM = '''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, dim: int):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.max(x, dim=self.dim)

def get_inputs():
    x = torch.randn(16, 4096, 256)
    return [x]

def get_init_inputs():
    return [1]
'''

# Two tensors, no modules: the only sources are the two positional inputs.
MATMUL_PROBLEM = '''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A, B):
        return torch.matmul(A, B)

def get_inputs():
    A = torch.rand(1024, 1024)
    B = torch.rand(1024, 1024)
    return [A, B]

def get_init_inputs():
    return []
'''

# A model built from a container: nested parameter names cannot be bounded, so
# the check must stay silent instead of guessing.
SEQUENTIAL_PROBLEM = '''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential(nn.Linear(64, 64), nn.ReLU())

    def forward(self, x):
        return self.net(x)

def get_inputs():
    return [torch.rand(8, 64)]

def get_init_inputs():
    return []
'''

# Path-dependent arity: unreadable, so the check must stay silent.
CONDITIONAL_INPUTS_PROBLEM = '''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return x

def get_inputs(flag=True):
    if flag:
        return [torch.rand(4)]
    return []

def get_init_inputs():
    return []
'''

P57_ROUND_6_KERNEL = '''
import torch
from torch import Tensor

def kernel_function(
    x: Tensor,
    weight: Tensor,
    stride_h: int,
    stride_w: int,
    padding_h: int,
    padding_w: int,
    output_padding_h: int,
    output_padding_w: int,
    batch_size: int,
    in_channels: int,
    out_channels: int,
    height: int,
    width: int,
    height_out: int,
    width_out: int,
):
    return extension.forward(x, weight)
'''

BINDABLE_CONV_KERNEL = '''
import torch
from torch import Tensor

def kernel_function(x: Tensor, weight: Tensor, bias: Tensor = None):
    return conv_transpose_forward(x, weight, bias)
'''

BINDABLE_CONFIG_KERNEL = '''
from torch import Tensor

def kernel_function(input: Tensor, weight: Tensor, bias: Tensor,
                    stride: int, padding: int, dilation: int, groups: int):
    return extension.forward(input, weight, bias, stride, padding, dilation, groups)
'''

BINDABLE_QUALIFIED_ALIAS_KERNEL = '''
from torch import Tensor

def kernel_function(x: Tensor, conv_transpose2d_weight: Tensor):
    return extension.forward(x, conv_transpose2d_weight)
'''

DIM_KERNEL = '''
from torch import Tensor

def kernel_function(x: Tensor, dim: int):
    return extension.reduce(x, dim)
'''

INIT_INPUT_KERNEL = '''
from torch import Tensor

def kernel_function(x: Tensor, in_channels: int):
    return extension.forward(x, in_channels)
'''

DEFAULTED_UNBINDABLE_KERNEL = '''
from torch import Tensor

def kernel_function(x: Tensor, training: bool = False):
    return extension.forward(x, training)
'''

VARARG_KERNEL = '''
def kernel_function(*args, **kwargs):
    return extension.forward(*args, **kwargs)
'''

MATMUL_KERNEL = '''
def kernel_function(A, B):
    return extension.matmul(A, B)
'''

MATMUL_EXTRA_KERNEL = '''
def kernel_function(A, B, C):
    return extension.matmul(A, B, C)
'''


def test_p57_unbindable_configuration_parameters_are_rejected():
    reason = kernel_signature_bindability(P57_ROUND_6_KERNEL, CONV_TRANSPOSE_PROBLEM)
    assert reason
    for name in ("stride_h", "padding_w", "batch_size", "width_out"):
        assert name in reason


def test_module_state_signature_is_accepted():
    assert kernel_signature_bindability(BINDABLE_CONV_KERNEL, CONV_TRANSPOSE_PROBLEM) is None


def test_module_configuration_signature_is_accepted():
    assert kernel_signature_bindability(
        BINDABLE_CONFIG_KERNEL, CONV_TRANSPOSE_PROBLEM) is None


def test_qualified_weight_alias_is_accepted():
    # bind_kernel_call resolves ``<module>_weight`` against qualified model state.
    assert kernel_signature_bindability(
        BINDABLE_QUALIFIED_ALIAS_KERNEL, CONV_TRANSPOSE_PROBLEM) is None


def test_get_init_inputs_value_is_rejected():
    # ``in_channels`` constructs the model and is never a kernel argument: it
    # must be baked into kernel.py instead of declared.
    reason = kernel_signature_bindability(INIT_INPUT_KERNEL, CONV_TRANSPOSE_PROBLEM)
    assert reason and "in_channels" in reason


def test_configuration_attribute_is_rejected():
    reason = kernel_signature_bindability(DIM_KERNEL, REDUCE_PROBLEM)
    assert reason and "dim" in reason


def test_defaulted_parameter_is_skippable():
    assert kernel_signature_bindability(
        DEFAULTED_UNBINDABLE_KERNEL, CONV_TRANSPOSE_PROBLEM) is None


def test_vararg_kernel_is_skipped():
    # bind_kernel_call feeds every source positionally; it cannot run out.
    assert kernel_signature_bindability(VARARG_KERNEL, CONV_TRANSPOSE_PROBLEM) is None


def test_two_input_problem_binds_two_positional_parameters():
    assert kernel_signature_bindability(MATMUL_KERNEL, MATMUL_PROBLEM) is None


def test_parameter_count_beyond_inputs_and_state_is_rejected():
    reason = kernel_signature_bindability(MATMUL_EXTRA_KERNEL, MATMUL_PROBLEM)
    assert reason and "C" in reason


def test_container_model_fails_open():
    assert kernel_signature_bindability(
        P57_ROUND_6_KERNEL, SEQUENTIAL_PROBLEM) is None


def test_conditional_input_arity_fails_open():
    assert kernel_signature_bindability(
        P57_ROUND_6_KERNEL, CONDITIONAL_INPUTS_PROBLEM) is None


def test_unreadable_problem_fails_open():
    assert kernel_signature_bindability(P57_ROUND_6_KERNEL, "") is None
    assert kernel_signature_bindability(P57_ROUND_6_KERNEL, "not python(") is None


def test_kernel_without_entry_point_is_not_this_check_s_business():
    assert kernel_signature_bindability("def other(x): pass", CONV_TRANSPOSE_PROBLEM) is None


def test_reason_is_actionable():
    reason = kernel_signature_bindability(P57_ROUND_6_KERNEL, CONV_TRANSPOSE_PROBLEM)
    assert "bind_kernel_call" in reason
    assert "get_inputs()" in reason
    assert "Bake" in reason


def _bundle(kernel_py: str) -> str:
    return (
        "FILE: kernel.py\n```python\n" + kernel_py + "\n```\n\n"
        "FILE: binding.cpp\n```cpp\n#include <torch/extension.h>\n```\n\n"
        "FILE: kernel.mu\n```cpp\n"
        "__global__ void k(float* out, const float* in) { out[0] = in[0]; }\n```\n\n"
        "FILE: setup.py\n```python\n"
        "from torch_musa.utils.musa_extension import MUSAExtension, BuildExtension\n"
        "from setuptools import setup\n"
        "setup(name='extension', ext_modules=[MUSAExtension(name='extension', "
        "sources=['binding.cpp', 'kernel.mu'])], "
        "cmdclass={'build_ext': BuildExtension})\n```\n"
    )


def _agent():
    agent = TritonKernelAgent.__new__(TritonKernelAgent)
    agent._kernel_backend = get_kernel_backend("musa")
    return agent


def test_seed_extraction_rejects_unbindable_signature():
    agent = _agent()
    with pytest.raises(ValueError, match="bind_kernel_call cannot"):
        agent._extract_kernel_payload(
            _bundle(P57_ROUND_6_KERNEL), CONV_TRANSPOSE_PROBLEM
        )


def test_seed_extraction_keeps_bindable_signature():
    agent = _agent()
    bundle = agent._extract_kernel_payload(
        _bundle(BINDABLE_CONV_KERNEL), CONV_TRANSPOSE_PROBLEM
    )
    assert sorted(bundle.files) == ["binding.cpp", "kernel.mu", "kernel.py", "setup.py"]


def test_seed_extraction_without_problem_source_is_unchanged():
    # Callers that pass no problem description keep the previous behavior.
    agent = _agent()
    bundle = agent._extract_kernel_payload(_bundle(P57_ROUND_6_KERNEL))
    assert sorted(bundle.files) == ["binding.cpp", "kernel.mu", "kernel.py", "setup.py"]
