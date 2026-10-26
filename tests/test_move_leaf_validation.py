"""Fix 4-1 regression tests: the 2-1 leaf rule must recognize move() leaves.

docs/design_kernelagent_fix_4-1_move_leaf_false_positive.md -- problems 45
and 47 died in the test-generation phase because every repair attempt used
kernel_case_suite.move(...) (which always returns fresh leaf tensors via
empty_strided + no_grad copy_ + requires_grad_(), case_suite.py L122-137)
and the fix 2-1 static rule rejected ``X.requires_grad = True`` on those
runtime-legal values as "derived tensor". These tests pin both directions:
the false positive is gone, and the true positives (clone/arithmetic-derived
requires_grad assignment, shadowed move) stay rejected.
"""

import pytest

from triton_kernel_agent.test_validation import validate_test_autograd


# P45 attempt 1 / P47 form: move(..., gradients=True) result.
MOVE_CALL_TEST = '''
import torch
from kernel_case_suite import move

def main():
    x = torch.randn(3, requires_grad=True)
    x_ref = move(x.detach(), "musa", gradients=True)
    x_ref.requires_grad = True
    out = model(x_ref)
    out.backward()
'''

# P45 attempt 2 form: two-step container element access.
MOVE_TWO_STEP_TEST = '''
import torch
from kernel_case_suite import move

def main():
    inputs_ref = move(get_inputs(), "musa")
    x_ref = inputs_ref[0]
    x_ref.requires_grad = True
    out = model(x_ref)
    out.backward()
'''

# Direct subscript over the move call.
MOVE_DIRECT_SUBSCRIPT_TEST = '''
import torch
from kernel_case_suite import move

def main():
    x_ref = move(get_inputs(), "musa")[0]
    x_ref.requires_grad = True
    out = model(x_ref)
    out.backward()
'''

MOVE_ALIAS_TEST = '''
import torch
from kernel_case_suite import move as mv

def main():
    x_ref = mv(get_inputs(), "musa")[0]
    x_ref.requires_grad = True
'''

MOVE_MODULE_ALIAS_TEST = '''
import torch
import kernel_case_suite as kcs

def main():
    x_ref = kcs.move(get_inputs(), "musa")[0]
    x_ref.requires_grad = True
'''

MOVE_DOTTED_TEST = '''
import torch
import kernel_case_suite

def main():
    x_ref = kernel_case_suite.move(get_inputs(), "musa")[0]
    x_ref.requires_grad = True
'''

CLONE_DERIVED_TEST = '''
import torch

def main():
    y = torch.randn(3, requires_grad=True)
    x = y.clone()
    x.requires_grad = True
'''

ARITHMETIC_DERIVED_TEST = '''
import torch

def main():
    y = torch.randn(3, requires_grad=True)
    x = y + 1
    x.requires_grad = True
'''

MOVE_THEN_CLONE_TEST = '''
import torch
from kernel_case_suite import move

def main():
    x = move(get_inputs(), "musa")[0].clone()
    x.requires_grad = True
'''

MOVE_DEEP_SUBSCRIPT_TEST = '''
import torch
from kernel_case_suite import move

def main():
    x = move(get_inputs(), "musa")[0][1]
    x.requires_grad = True
'''

LOCAL_MOVE_SHADOW_TEST = '''
import torch
from kernel_case_suite import move

def move(value, device):
    return value.clone()

def main():
    y = torch.randn(3, requires_grad=True)
    x = move(y, "musa")
    x.requires_grad = True
'''

ATTRIBUTE_REBIND_SHADOW_TEST = '''
import torch
import kernel_case_suite
from kernel_case_suite import move

def fake(value, device):
    return value.clone()

def main():
    kernel_case_suite.move = fake
    y = torch.randn(3, requires_grad=True)
    x = move(y, "musa")
    x.requires_grad = True
'''

# A move() result is a fresh differentiation root even when its source value
# was computed under no_grad: the copy_ runs under no_grad, so gradients never
# flow back to the blocked source.
NOGRAD_MOVE_ROOT_TEST = '''
import torch
from kernel_case_suite import move

def main():
    x = torch.randn(3, requires_grad=True)
    with torch.no_grad():
        y = model(x)
    leaf = move(y, "musa", gradients=True)
    leaf.requires_grad = True
    out = model2(leaf)
    out.backward()
'''


def test_move_call_result_is_a_leaf():
    # P45 attempt 1 / P47: rejected twice by the old rule, legal at runtime.
    validate_test_autograd(MOVE_CALL_TEST)


def test_move_two_step_element_is_a_leaf():
    # P45 attempt 2: inputs_ref = move(...); x_ref = inputs_ref[0].
    validate_test_autograd(MOVE_TWO_STEP_TEST)


def test_direct_move_subscript_is_a_leaf():
    validate_test_autograd(MOVE_DIRECT_SUBSCRIPT_TEST)


def test_move_alias_form_is_a_leaf():
    validate_test_autograd(MOVE_ALIAS_TEST)


def test_move_module_alias_form_is_a_leaf():
    validate_test_autograd(MOVE_MODULE_ALIAS_TEST)


def test_move_dotted_form_is_a_leaf():
    validate_test_autograd(MOVE_DOTTED_TEST)


def test_clone_true_positive_is_preserved():
    with pytest.raises(ValueError, match="not a derived tensor"):
        validate_test_autograd(CLONE_DERIVED_TEST)


def test_arithmetic_true_positive_is_preserved():
    with pytest.raises(ValueError, match="not a derived tensor"):
        validate_test_autograd(ARITHMETIC_DERIVED_TEST)


def test_method_chain_after_move_stays_derived():
    with pytest.raises(ValueError, match="not a derived tensor"):
        validate_test_autograd(MOVE_THEN_CLONE_TEST)


def test_deep_move_subscript_is_a_leaf():
    # move maps nested containers recursively; every element is a fresh leaf.
    validate_test_autograd(MOVE_DEEP_SUBSCRIPT_TEST)


def test_locally_shadowed_move_is_not_trusted():
    with pytest.raises(ValueError, match="not a derived tensor"):
        validate_test_autograd(LOCAL_MOVE_SHADOW_TEST)


def test_module_attribute_rebind_voids_recognition():
    with pytest.raises(ValueError, match="not a derived tensor"):
        validate_test_autograd(ATTRIBUTE_REBIND_SHADOW_TEST)


def test_move_result_under_no_grad_is_a_usable_root():
    validate_test_autograd(NOGRAD_MOVE_ROOT_TEST)


def test_rejection_message_names_the_legal_patterns():
    with pytest.raises(ValueError) as excinfo:
        validate_test_autograd(CLONE_DERIVED_TEST)
    message = str(excinfo.value)
    assert "move(...) result from kernel_case_suite" in message
    assert ".requires_grad_()" in message
    # The misleading torch.randn-only advice must be gone.
    assert "e.g. torch.randn(...)" not in message


# ---------------------------------------------------------------------------
# Same 2-1 leaf rule, extended the same way as move(): model state iterated out
# of a Module and ``.detach()`` results are leaves too. Observed live on
# 2026-10-25 (agent_20261025_093522.log): a legal
# ``for param in ref_model.parameters(): param.requires_grad = True`` was
# rejected and cost a full regeneration of the test.
# ---------------------------------------------------------------------------

MODEL_PARAMETER_LOOP_TEST = '''
import torch
from kernel import kernel_function
from kernel_case_suite import bind_kernel_call, move

def main():
    ref_model = Model(*get_init_inputs())
    for param in ref_model.parameters():
        param.requires_grad = True
    out = ref_model(*get_inputs())
    out.sum().backward()
'''

MODEL_BUFFER_LOOP_TEST = '''
import torch

def main():
    model = Model()
    for buffer in model.buffers():
        buffer.requires_grad = True
'''

NAMED_PARAMETER_LOOP_TEST = '''
import torch

def main():
    model = Model()
    for name, param in model.named_parameters():
        param.requires_grad = True
'''

SUB_MODULE_PARAMETER_LOOP_TEST = '''
import torch

def main():
    model = Model()
    for param in model.conv.parameters():
        param.requires_grad = True
'''

DETACH_LEAF_TEST = '''
import torch

def main():
    y = torch.randn(3)
    x = y.detach()
    x.requires_grad = True
'''

# ``detach()`` makes an independent differentiation root: no graph survives the
# detach, so differentiating it stays legal even under no_grad.
DETACH_UNDER_NOGRAD_BACKWARD_TEST = '''
import torch

def main():
    x = torch.randn(3, requires_grad=True)
    with torch.no_grad():
        y = model(x)
    z = y.detach()
    z.requires_grad = True
    z.sum().backward()
'''

DETACH_THEN_CLONE_TEST = '''
import torch

def main():
    y = torch.randn(3, requires_grad=True)
    x = y.detach().clone()
    x.requires_grad = True
'''

LOCAL_PARAMETERS_SHADOW_TEST = '''
import torch

def parameters(self):
    return [self.weight.clone()]

def main():
    model = Model()
    for param in model.parameters():
        param.requires_grad = True
'''

NON_STATE_ITERATION_TEST = '''
import torch

def main():
    a = torch.randn(3)
    b = a.clone()
    for tensor in [a, b]:
        tensor.requires_grad = True
'''


def test_parameters_loop_variable_is_a_leaf():
    validate_test_autograd(MODEL_PARAMETER_LOOP_TEST)


def test_buffers_loop_variable_is_a_leaf():
    validate_test_autograd(MODEL_BUFFER_LOOP_TEST)


def test_named_parameters_tuple_form_binds_the_value_as_leaf():
    validate_test_autograd(NAMED_PARAMETER_LOOP_TEST)


def test_submodule_parameters_loop_variable_is_a_leaf():
    validate_test_autograd(SUB_MODULE_PARAMETER_LOOP_TEST)


def test_detach_result_is_a_leaf():
    validate_test_autograd(DETACH_LEAF_TEST)


def test_detach_result_under_no_grad_is_a_usable_root():
    validate_test_autograd(DETACH_UNDER_NOGRAD_BACKWARD_TEST)


def test_detach_chain_stays_derived():
    # Truncation rule from fix 4-1: only the detach() call itself yields a leaf.
    with pytest.raises(ValueError, match="not a derived tensor"):
        validate_test_autograd(DETACH_THEN_CLONE_TEST)


def test_shadowed_parameters_is_not_trusted():
    with pytest.raises(ValueError, match="not a derived tensor"):
        validate_test_autograd(LOCAL_PARAMETERS_SHADOW_TEST)


def test_iteration_over_a_derived_list_is_not_a_leaf_source():
    with pytest.raises(ValueError, match="not a derived tensor"):
        validate_test_autograd(NON_STATE_ITERATION_TEST)


def test_rejection_message_names_the_extended_leaf_patterns():
    with pytest.raises(ValueError) as excinfo:
        validate_test_autograd(CLONE_DERIVED_TEST)
    message = str(excinfo.value)
    assert "parameters()/buffers()" in message
    assert ".detach()" in message
