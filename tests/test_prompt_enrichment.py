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


def test_musa_knowledge_pack_loads_common_and_operator_guidance():
    knowledge = MUSAKnowledgePack().retrieve("rmsnorm")
    assert "Common native MUSA constraints" in knowledge
    assert "Normalization guidance" in knowledge
    assert "Matmul guidance" not in knowledge


def test_prompt_manager_selects_specialized_musa_template():
    manager = PromptManager(
        target_platform=get_platform("musa"), kernel_backend="musa"
    )
    prompt = manager.render_kernel_generation_prompt(
        problem_description=_RMSNORM_PROBLEM,
        test_code="from kernel import kernel_function",
    )
    assert "Normalization generation strategy" in prompt
    assert "Structured problem contract" in prompt
    assert "Normalization guidance" in prompt
