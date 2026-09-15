from unittest.mock import Mock

from triton_kernel_agent.agent import (
    TritonKernelAgent,
    validate_generated_test_input_contract,
)
from triton_kernel_agent.musa_knowledge import MUSAKnowledgePack
from triton_kernel_agent.platform_config import get_platform
from triton_kernel_agent.prompt_context import ProblemContextBuilder
from triton_kernel_agent.prompt_manager import PromptManager


_RMSNORM_PROBLEM = """
import torch
M, N = 1024, 4096
EPS = 1e-6
class Model:
    def forward(self, x, weight):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + EPS) * weight
def get_inputs():
    return [
        torch.randn((M, N), dtype=torch.bfloat16),
        torch.ones((N,), dtype=torch.bfloat16),
    ]
def get_init_inputs():
    return [N, EPS]
"""


def test_problem_context_builder_extracts_contract():
    context = ProblemContextBuilder().build(_RMSNORM_PROBLEM)
    assert context.operator_type == "rmsnorm"
    assert len(context.inputs) == 2
    assert context.inputs[0].shape == "(M, N)"
    assert context.inputs[0].dtype == "torch.bfloat16"
    assert "torch.rsqrt" in context.operations
    assert context.init_inputs == "[N, EPS]"
    summary = context.format_for_prompt()
    assert "Structured problem contract" in summary
    assert "Forward semantics" in summary


def test_generated_test_reuses_exact_problem_input_factories():
    test_code = (
        _RMSNORM_PROBLEM + "\ninputs = get_inputs()\ninit_inputs = get_init_inputs()\n"
    )
    validate_generated_test_input_contract(_RMSNORM_PROBLEM, test_code)


def test_generated_test_rejects_changed_factory_workload():
    test_code = (
        _RMSNORM_PROBLEM.replace("torch.randn((M, N)", "torch.randn((1, N)")
        + "\ninputs = get_inputs()\ninit_inputs = get_init_inputs()\n"
    )
    try:
        validate_generated_test_input_contract(_RMSNORM_PROBLEM, test_code)
    except ValueError as exc:
        assert "exact get_inputs() definition" in str(exc)
    else:
        raise AssertionError("changed input factory was accepted")


def test_generated_test_rejects_uncalled_factory():
    try:
        validate_generated_test_input_contract(_RMSNORM_PROBLEM, _RMSNORM_PROBLEM)
    except ValueError as exc:
        assert "get_inputs() call" in str(exc)
        assert "get_init_inputs() call" in str(exc)
    else:
        raise AssertionError("uncalled input factories were accepted")


def test_generate_test_retries_once_after_input_contract_violation():
    invalid = "def test_kernel():\n    inputs = [123]\n"
    valid = (
        _RMSNORM_PROBLEM + "\ninputs = get_inputs()\ninit_inputs = get_init_inputs()\n"
    )
    responses = iter([invalid, valid])
    agent = TritonKernelAgent.__new__(TritonKernelAgent)
    agent.provider = object()
    agent.model_name = "test-model"
    agent.prompt_manager = Mock()
    agent.prompt_manager.render_test_generation_prompt.return_value = "prompt"
    agent.logger = Mock()
    agent._call_llm = Mock(side_effect=lambda *_args, **_kwargs: next(responses))
    agent._extract_code_from_response = Mock(side_effect=lambda text: text)

    assert agent._generate_test(_RMSNORM_PROBLEM) == valid
    assert agent._call_llm.call_count == 2
    repair_messages = agent._call_llm.call_args_list[1].args[0]
    assert (
        "Copy get_inputs() and get_init_inputs() exactly"
        in repair_messages[-1]["content"]
    )


def test_musa_knowledge_pack_loads_common_and_operator_guidance():
    knowledge = MUSAKnowledgePack().retrieve("rmsnorm")
    assert "Common native MUSA constraints" in knowledge
    assert "Normalization guidance" in knowledge
    assert "Matmul guidance" not in knowledge


def test_prompt_manager_selects_specialized_musa_template():
    manager = PromptManager(target_platform=get_platform("musa"), kernel_backend="musa")
    prompt = manager.render_kernel_generation_prompt(
        problem_description=_RMSNORM_PROBLEM,
        test_code="from kernel import kernel_function",
    )
    assert "Normalization generation strategy" in prompt
    assert "Structured problem contract" in prompt
    assert "Normalization guidance" in prompt
