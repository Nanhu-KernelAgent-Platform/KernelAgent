"""Fix 3-1 regression tests: kernel_function call-contract binding.

docs/design_kernelagent_fix_3-1_signature_contract_binding.md — the main test
is generated BEFORE the kernel exists, so it must invoke kernel_function
through bind_kernel_call (exactly like the deterministic case suite and the
KernelBench evaluator already do) instead of hardcoding a call signature.
P50/P56 burned full verification runs on hardcoded-signature TypeErrors while
the case suite passed.
"""

from unittest.mock import Mock

import pytest

from triton_kernel_agent.agent import TritonKernelAgent
from triton_kernel_agent.platform_config import get_platform
from triton_kernel_agent.prompt_manager import PromptManager
from triton_kernel_agent.test_validation import validate_test_kernel_call_binding


DIRECT_CALL_TEST = '''
import torch
import torch_musa
from kernel import kernel_function

def main():
    x = torch.randn(4)
    out = kernel_function(x)
'''

ALIASED_DIRECT_CALL_TEST = '''
import torch
from kernel import kernel_function as kf

def main():
    x = torch.randn(4)
    out = kf(x)
'''

DOTTED_DIRECT_CALL_TEST = '''
import torch
import kernel as k

def main():
    x = torch.randn(4)
    out = k.kernel_function(x)
'''

BIND_PATTERN_TEST = '''
import copy
import torch
import torch_musa
from kernel import kernel_function
from kernel_case_suite import bind_kernel_call

def main():
    x = torch.randn(4)
    invoke_native = bind_kernel_call(kernel_function, None, [x])
    native_out = invoke_native()
'''

DOTTED_BIND_TEST = BIND_PATTERN_TEST.replace(
    "from kernel_case_suite import bind_kernel_call", "import kernel_case_suite"
).replace(
    "bind_kernel_call(kernel_function, None, [x])",
    "kernel_case_suite.bind_kernel_call(kernel_function, None, [x])",
)

VACUOUS_TEST = '''
import torch
from kernel import kernel_function

def main():
    x = torch.randn(4)
    print(x.shape)
'''

UNIMPORTED_BIND_TEST = '''
import torch
from kernel import kernel_function

def main():
    x = torch.randn(4)
    invoke = bind_kernel_call(kernel_function, None, [x])
    native_out = invoke()
'''

DOTTED_REFERENCE_ONLY_TEST = '''
import kernel

def main():
    f = kernel.kernel_function
'''

PLAIN_TEST = '''
import torch

def main():
    x = torch.randn(4)
    print(x.shape)
'''


def test_direct_kernel_function_call_rejected():
    with pytest.raises(ValueError, match="not called directly"):
        validate_test_kernel_call_binding(DIRECT_CALL_TEST)


def test_direct_call_error_names_the_repair_pattern():
    with pytest.raises(ValueError, match="bind_kernel_call\\(kernel_function"):
        validate_test_kernel_call_binding(DIRECT_CALL_TEST)


def test_direct_call_error_reports_line_number():
    with pytest.raises(ValueError, match="line 8"):
        validate_test_kernel_call_binding(DIRECT_CALL_TEST)


def test_direct_call_via_import_alias_rejected():
    with pytest.raises(ValueError, match="not called directly"):
        validate_test_kernel_call_binding(ALIASED_DIRECT_CALL_TEST)


def test_direct_call_via_module_alias_rejected():
    with pytest.raises(ValueError, match="not called directly"):
        validate_test_kernel_call_binding(DOTTED_DIRECT_CALL_TEST)


def test_bind_kernel_call_pattern_passes():
    # kernel_function appears only as a Name argument to bind_kernel_call,
    # never as a direct Call — the mandated pattern must be accepted.
    validate_test_kernel_call_binding(BIND_PATTERN_TEST)


def test_dotted_bind_call_passes():
    validate_test_kernel_call_binding(DOTTED_BIND_TEST)


def test_vacuous_reference_rejected():
    with pytest.raises(ValueError, match="never exercised"):
        validate_test_kernel_call_binding(VACUOUS_TEST)


def test_unimported_bind_call_reports_missing_exercise():
    # A bare bind_kernel_call call without the kernel_case_suite import is a
    # missing import; it must fall through to the instructive error instead
    # of counting as a valid invocation site.
    with pytest.raises(ValueError, match="import bind_kernel_call"):
        validate_test_kernel_call_binding(UNIMPORTED_BIND_TEST)


def test_dotted_reference_without_bind_rejected():
    with pytest.raises(ValueError, match="never exercised"):
        validate_test_kernel_call_binding(DOTTED_REFERENCE_ONLY_TEST)


def test_test_without_kernel_reference_passes():
    validate_test_kernel_call_binding(PLAIN_TEST)


def make_agent(platform):
    agent = TritonKernelAgent.__new__(TritonKernelAgent)
    agent.provider = object()
    agent.model_name = "test-model"
    agent._platform_config = platform
    agent.prompt_manager = Mock()
    agent.logger = Mock()
    agent._extract_code_from_response = lambda text: text
    return agent


def test_direct_call_test_repaired_before_kernel_generation():
    agent = make_agent(get_platform("musa"))
    agent._call_llm = Mock(side_effect=[DIRECT_CALL_TEST, BIND_PATTERN_TEST])
    assert agent._generate_test("formula") == BIND_PATTERN_TEST
    assert agent._call_llm.call_count == 2


def test_binding_validator_is_musa_scoped():
    # The B2 template mandate lands on the MUSA templates first, so the
    # validator must not reject a non-MUSA test that calls directly.
    agent = make_agent(get_platform("cuda"))
    agent._call_llm = Mock(side_effect=[DIRECT_CALL_TEST])
    assert agent._generate_test("formula") == DIRECT_CALL_TEST
    assert agent._call_llm.call_count == 1


def test_third_attempt_recovers_before_kernel_generation():
    # Fix 4-1 A3: three attempts; the third fires only where the previous
    # behavior hard-failed the whole problem (P47 wasted attempt 1 on a
    # no-code response and attempt 2 on a validation false positive).
    agent = make_agent(get_platform("musa"))
    agent._call_llm = Mock(side_effect=[DIRECT_CALL_TEST, DIRECT_CALL_TEST, BIND_PATTERN_TEST])
    assert agent._generate_test("formula") == BIND_PATTERN_TEST
    assert agent._call_llm.call_count == 3


def test_three_failed_attempts_raise_the_last_contract_error():
    agent = make_agent(get_platform("musa"))
    agent._call_llm = Mock(side_effect=[DIRECT_CALL_TEST, DIRECT_CALL_TEST, DIRECT_CALL_TEST])
    with pytest.raises(ValueError, match="not called directly"):
        agent._generate_test("formula")
    assert agent._call_llm.call_count == 3


def test_repair_message_carries_the_legal_leaf_pattern():
    # Fix 4-1 A2: the repair prompt must teach the concrete move() pattern
    # instead of the misleading torch.randn-only advice (P45/P47 evidence).
    agent = make_agent(get_platform("musa"))
    agent._call_llm = Mock(side_effect=[DIRECT_CALL_TEST, BIND_PATTERN_TEST])
    agent._generate_test("formula")
    repair_message = agent._call_llm.call_args.args[0][-1]["content"]
    assert "move(tensor.detach(), device, gradients=True)" in repair_message
    assert "kernel_case_suite" in repair_message
    assert ".requires_grad_()" in repair_message


def musa_manager():
    return PromptManager(target_platform=get_platform("musa"), kernel_backend="musa")


def test_test_generation_prompt_mandates_call_binding():
    prompt = musa_manager().render_test_generation_prompt("training formula")
    assert "MANDATORY CALL BINDING" in prompt
    assert "bind_kernel_call" in prompt
    assert "copy.deepcopy" in prompt
    assert "NEVER hardcode" in prompt
    # The old permissive wording must be gone.
    assert "as a normal Python function" not in prompt


def test_generation_prompt_documents_binding_contract():
    prompt = musa_manager().render_kernel_generation_prompt("formula", BIND_PATTERN_TEST)
    assert "Evaluation call convention" in prompt
    assert "bind_kernel_call" in prompt
    assert "BY NAME" in prompt
    assert "POSITIONALLY" in prompt
    assert "cannot be baked in" in prompt
    assert "Cannot bind required kernel parameter(s)" in prompt
    # The old wrong convention ("called with exactly the tensors returned by
    # get_inputs()") must be gone.
    assert "called with exactly the tensors" not in prompt


def test_refinement_prompt_documents_binding_contract():
    manager = musa_manager()
    prompt = manager.render_kernel_refinement_prompt(
        "formula", BIND_PATTERN_TEST, "kernel code", {"stdout": "", "stderr": "TypeError"}
    )
    assert "Evaluation call convention" in prompt
    assert "kernel_function(input, dim)" in prompt
    assert "bind_kernel_call" in prompt
    assert "called with exactly the tensors" not in prompt


def test_reduction_variant_inherits_binding_contract():
    template = musa_manager().env.get_template("musa_kernel_generation_reduction.j2")
    prompt = template.render(
        problem_description="formula",
        test_code=BIND_PATTERN_TEST,
        triton_guidelines="",
        kernel_guidance="",
        structured_problem_context="",
        knowledge_context="",
    )
    assert "Evaluation call convention" in prompt
    assert "bind_kernel_call" in prompt
    assert "cannot be baked in" in prompt
